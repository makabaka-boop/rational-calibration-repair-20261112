"""独立验收测试（标准库 unittest，全程 Fraction 精确运算）。

独立性：测试内自行实现 RREF、零空间与 KKT 条件核对，
不导入被测包的 solver 内部函数来“自证”。

运行：
    python3 -I tests/run_acceptance.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from joint_calibration.model import ValidationError, validate  # noqa: E402
from joint_calibration.solver import solve_model  # noqa: E402
from joint_calibration.strict_json import StrictJSONError, loads  # noqa: E402

ZERO = Fraction(0)
ONE = Fraction(1)


# ============================ 独立参考实现 ============================
def ref_rref(mat, col_start, col_end):
    """独立 RREF，返回主元列（全局列号）。"""
    mat = [list(r) for r in mat]
    pivots, pr = [], 0
    for col in range(col_start, col_end):
        cand = next((r for r in range(pr, len(mat)) if mat[r][col] != 0), None)
        if cand is None:
            continue
        mat[pr], mat[cand] = mat[cand], mat[pr]
        piv = mat[pr][col]
        mat[pr] = [v / piv for v in mat[pr]]
        for r in range(len(mat)):
            if r != pr and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[pr])]
        pivots.append(col)
        pr += 1
        if pr == len(mat):
            break
    return mat, pivots


def ref_solution_space(equations, n):
    """equations: [(coef 向量, rhs)]；返回 (特解, 零空间基, RREF 含组合块)。"""
    m = len(equations)
    aug = []
    for i, (coef, rhs) in enumerate(equations):
        row = [ONE if j == i else ZERO for j in range(m)]
        row += coef + [rhs]
        aug.append(row)
    reduced, piv_glob = ref_rref(aug, m, m + n)
    piv = [p - m for p in piv_glob]
    return reduced, piv, m


def ref_particular_null(reduced, piv, m, n):
    particular = [ZERO] * n
    for r, pc in enumerate(piv):
        particular[pc] = reduced[r][m + n]
    pset = set(piv)
    null = []
    for fv in range(n):
        if fv in pset:
            continue
        v = [ZERO] * n
        v[fv] = ONE
        for r, pc in enumerate(piv):
            v[pc] = -reduced[r][m + fv]
        null.append(v)
    return particular, null


def ref_solve_homogeneous(a):
    """独立求 A t = 0 的零空间基底（列向量列表），基于自带 RREF。"""
    if not a:
        return [], []
    k = len(a[0])
    rows = [list(r) + [ZERO] for r in a]
    reduced, piv_glob = ref_rref(rows, 0, k)
    pset = set(piv_glob)
    dirs = []
    for fv in range(k):
        if fv in pset:
            continue
        v = [ZERO] * k
        v[fv] = ONE
        for r, pc in enumerate(piv_glob):
            v[pc] = -reduced[r][fv]
        dirs.append(v)
    return [ZERO] * k, dirs


def ref_inconsistent_row(reduced, m, n):
    for row in reduced:
        if all(row[m + j] == 0 for j in range(n)) and row[m + n] != 0:
            return row
    return None


def transpose(mat):
    return [list(c) for c in zip(*mat)] if mat else []


def mat_mul(a, b):
    bt = transpose(b)
    return [[sum((x * y for x, y in zip(r, c)), ZERO) for c in bt] for r in a]


def mat_vec(a, v):
    return [sum((x * y for x, y in zip(r, v)), ZERO) for r in a]


def build_equations(model):
    """构造独立的方程列表 (E 行, b)，按 测量+守恒 顺序。"""
    n = model.n
    eqs, ids = [], []
    for ob in model.observations:
        row = [ZERO] * n
        for idx, c in ob.coeff.items():
            row[idx] = c
        eqs.append((row, ob.value))
        ids.append(ob.id)
    for bal in model.balances:
        row = [ZERO] * n
        for name, c in bal.terms.items():
            row[model.variables.index(name)] = c
        eqs.append((row, bal.value))
        ids.append(bal.id)
    return eqs, ids


def solve_text(text):
    return solve_model(validate(loads(text)))


def frac(s):
    return Fraction(s)


# ============================ 测试夹具 ============================
CANONICAL = """{
  "variables": ["x", "y"],
  "noises": ["z"],
  "observations": [
    {"id": "o1", "value": "3/1", "terms": {"x": "1/1"}, "noise_terms": {"z": "1/1"}},
    {"id": "o2", "value": "5/1", "terms": {"y": "1/1"}, "noise_terms": {"z": "1/1"}},
    {"id": "o3", "value": "6/1", "terms": {"x": "1/1", "y": "1/1"}, "noise_terms": {}}
  ],
  "balances": []
}"""

# 共享误差：e1 同时出现在 3 个测量中；方程数少于未知量数，解不唯一，
# 两层优化都真正起作用（第一层先压缩共享偏差，第二层再处理缺测自由度）
SHARED = """{
  "variables": ["a", "b"],
  "noises": ["e1", "e2"],
  "observations": [
    {"id": "m1", "value": "10/1", "terms": {"a": "1/1"}, "noise_terms": {"e1": "1/1"}},
    {"id": "m2", "value": "12/1", "terms": {"b": "1/1"}, "noise_terms": {"e1": "1/1"}},
    {"id": "m3", "value": "3/1", "terms": {"a": "1/1", "b": "1/1"}, "noise_terms": {"e2": "1/1"}}
  ],
  "balances": []
}"""

# 冗余等式：b2 是 b1 与 b3 的和，系统仍一致
REDUNDANT = """{
  "variables": ["p", "q"],
  "noises": ["w"],
  "observations": [
    {"id": "n1", "value": "4/1", "terms": {"p": "1/1"}, "noise_terms": {"w": "1/1"}}
  ],
  "balances": [
    {"id": "b1", "terms": {"p": "1/1"}, "value": "2/1"},
    {"id": "b3", "terms": {"q": "1/1"}, "value": "5/1"},
    {"id": "b2", "terms": {"p": "1/1", "q": "1/1"}, "value": "7/1"}
  ]
}"""

# 矛盾守恒：x=1 与 x=2
CONTRA_BAL = """{
  "variables": ["x"],
  "noises": [],
  "observations": [],
  "balances": [
    {"id": "c1", "terms": {"x": "1/1"}, "value": "1/1"},
    {"id": "c2", "terms": {"x": "1/1"}, "value": "2/1"}
  ]
}"""

# 测量内部矛盾（含噪声，需组合消去未知量）
CONTRA_OBS = """{
  "variables": ["u"],
  "noises": ["g"],
  "observations": [
    {"id": "d1", "value": "1/1", "terms": {"u": "1/1"}, "noise_terms": {"g": "1/1"}},
    {"id": "d2", "value": "1/1", "terms": {"u": "1/1"}, "noise_terms": {"g": "1/1"}},
    {"id": "d3", "value": "9/1", "terms": {"u": "1/1"}, "noise_terms": {"g": "1/1"}}
  ],
  "balances": []
}"""

# 缺测变量：c 没有任何等式引用，第二层必须把它拉到 0
MISSING = """{
  "variables": ["a", "b", "c"],
  "noises": ["h"],
  "observations": [
    {"id": "k1", "value": "2/1", "terms": {"a": "1/1"}, "noise_terms": {"h": "1/1"}},
    {"id": "k2", "value": "5/1", "terms": {"b": "1/1"}, "noise_terms": {}}
  ],
  "balances": [
    {"id": "k3", "terms": {"a": "1/1", "b": "1/1"}, "value": "7/1"}
  ]
}"""

# 乱序：标识与数组顺序故意打乱；v0 缺测（无任何等式引用），
# s0 为系数全空的零值测量，系统一致且存在零空间
SHUFFLED = """{
  "noises": ["zeta", "alpha"],
  "variables": ["v2", "v1", "v0"],
  "balances": [
    {"id": "z2", "value": "1/1", "terms": {"v1": "-1/2", "v2": "1/4"}}
  ],
  "observations": [
    {"id": "s2", "value": "0/1", "terms": {"v0": "1/1"}, "noise_terms": {"alpha": "1/1", "zeta": "1/1"}},
    {"id": "s1", "value": "3/2", "terms": {"v0": "1/1", "v2": "1/2"}, "noise_terms": {"zeta": "1/1"}},
    {"id": "s0", "value": "0/1", "terms": {}, "noise_terms": {}}
  ]
}"""

# 两层自由都存在的构造：观测只约束一个线性组合
TWO_LAYER = """{
  "variables": ["x1", "x2", "x3"],
  "noises": ["n1", "n2"],
  "observations": [
    {"id": "t1", "value": "6/1", "terms": {"x1": "1/1", "x2": "1/1"}, "noise_terms": {"n1": "1/1"}},
    {"id": "t2", "value": "4/1", "terms": {"x2": "1/1", "x3": "1/1"}, "noise_terms": {"n2": "1/1"}}
  ],
  "balances": []
}"""

# 空输入：没有任何等式与未知量
EMPTY = '{"variables": [], "noises": [], "observations": [], "balances": []}'

# 零值方程
ZERO_EQ = """{
  "variables": ["r"],
  "noises": [],
  "observations": [
    {"id": "e0", "value": "0/1", "terms": {}, "noise_terms": {}}
  ],
  "balances": [
    {"id": "e1", "terms": {"r": "0/1"}, "value": "0/1"}
  ]
}"""


# ============================ 功能与精确结果 ============================
class CanonicalTests(unittest.TestCase):
    def test_canonical_values(self):
        res = solve_text(CANONICAL)
        self.assertEqual(res["status"], "optimal")
        self.assertEqual(res["x"], {"x": "2/1", "y": "4/1"})
        self.assertEqual(res["z"], {"z": "1/1"})
        self.assertEqual(res["corrected"],
                         {"o1": "2/1", "o2": "4/1", "o3": "6/1"})
        self.assertEqual(res["objectives"],
                         {"sum_z_squared": "1/1", "sum_x_squared": "20/1"})

    def test_all_outputs_are_reduced_positive_denominator(self):
        res = solve_text(CANONICAL)
        def check(s):
            p, q = s.split("/")
            int(p)
            self.assertGreater(int(q), 0)
            self.assertEqual(Fraction(int(p), int(q)).denominator, int(q))
        for d in (res["x"], res["z"], res["corrected"]):
            for v in d.values():
                check(v)
        for v in res["objectives"].values():
            check(v)

    def test_sorting(self):
        res = solve_text(SHUFFLED)
        self.assertEqual(list(res["x"]), sorted(res["x"]))
        self.assertEqual(list(res["z"]), sorted(res["z"]))
        self.assertEqual(list(res["corrected"]), sorted(res["corrected"]))

    def test_empty_input(self):
        res = solve_text(EMPTY)
        self.assertEqual(res, {
            "status": "optimal", "x": {}, "z": {}, "corrected": {},
            "objectives": {"sum_z_squared": "0/1", "sum_x_squared": "0/1"},
        })

    def test_zero_equations(self):
        res = solve_text(ZERO_EQ)
        self.assertEqual(res["status"], "optimal")
        self.assertEqual(res["x"]["r"], "0/1")
        self.assertEqual(res["corrected"]["e0"], "0/1")


# ============================ 独立核对：等式、两层最优性 ============================
def assert_two_layer_optimal(case, text):
    """独立核对：等式成立、解在解空间内、两层 KKT 最优性均满足。"""
    model = validate(loads(text))
    res = solve_model(model)
    case.assertEqual(res["status"], "optimal")
    eqs, ids = build_equations(model)
    n = model.n
    reduced, piv, m = ref_solution_space(eqs, n)
    case.assertIsNone(ref_inconsistent_row(reduced, m, n),
                      "参考实现判定一致，但求解器返回了最优？")
    u0, null = ref_particular_null(reduced, piv, m, n)

    x = [frac(res["x"][v]) for v in model.variables]
    z = [frac(res["z"][v]) for v in model.noises]
    u = x + z

    # 1) 满足全部原始等式
    for (row, rhs), eid in zip(eqs, ids):
        case.assertEqual(sum((a * b for a, b in zip(row, u)), ZERO),
                         rhs, f"等式 {eid} 未被满足")

    # 2) 解确实在仿射解空间内（RREF 回代检验）。
    diff = [a - b for a, b in zip(u, u0)]
    pset = set(piv)
    free = [fv for fv in range(n) if fv not in pset]
    coefs = {fv: diff[fv] for fv in free}
    for r, pc in enumerate(piv):
        expected = sum((-reduced[r][m + fv]) * coefs[fv] for fv in free)
        case.assertEqual(diff[pc], expected, "解不在解空间内")

    if not null:
        return res, model, (u0, null)

    nz, nx = model.nz, model.nx

    # 3) 第一层 KKT：z* 与零空间的 z 分块 Nz 正交，即 Nz^T z* = 0。
    null_t = transpose(null)          # n × k
    nz_block_t = null_t[nx:]          # nz × k（测试独立构造，不复用求解器）
    k = len(null)
    for col in range(k):
        cond = sum((nz_block_t[j][col] * z[j] for j in range(nz)), ZERO)
        case.assertEqual(cond, ZERO, "第一层最优性 KKT 失败")

    # 4) 第二层 KKT：第一层最优集内方向参数 t 满足 (Nz^T Nz) t = 0。
    #    独立求该齐次系统零空间基底 T，得到 u 空间可行方向 N T，
    #    再要求 x* 与每个可行方向的 x 分块正交。
    gram = mat_mul(transpose(nz_block_t), nz_block_t)  # k × k
    _, t_dirs = ref_solve_homogeneous(gram)            # 列向量列表
    first_level_dirs = mat_mul(null_t, transpose(t_dirs))  # n × nfree
    for col in range(len(first_level_dirs[0]) if first_level_dirs else 0):
        d_x = [first_level_dirs[j][col] for j in range(nx)]
        case.assertEqual(
            sum((x[j] * d_x[j] for j in range(nx)), ZERO), ZERO,
            "第二层最优性 KKT 失败")
    return res, model, (u0, null)


class IndependentOptimalityTests(unittest.TestCase):
    def _check_all(self, text):
        return assert_two_layer_optimal(self, text)


    def test_canonical(self):
        self._check_all(CANONICAL)

    def test_shared_error(self):
        res, model, _ = self._check_all(SHARED)
        # 同一误差标识在多个测量中的 z 值只有一份（空间维度为 2）
        self.assertEqual(set(res["z"]), {"e1", "e2"})
        # 逐测量复核含共享误差的完整等式
        for ob in model.observations:
            lhs = sum(
                (c * frac(res["x"][name]) for name, c in ob.x_terms.items()),
                ZERO)
            lhs += sum(
                (c * frac(res["z"][name])
                 for name, c in ob.z_terms.items()),
                ZERO)
            self.assertEqual(lhs, ob.value, f"测量 {ob.id} 等式不成立")
        # e1 被 m1、m2 共享：两处取到的是同一个 z 值（输出只有一个键）
        self.assertEqual(len(res["z"]), 2)

    def test_redundant(self):
        res, model, _ = self._check_all(REDUNDANT)
        self.assertEqual(frac(res["x"]["p"]), frac("2/1"))
        self.assertEqual(frac(res["x"]["q"]), frac("5/1"))
        self.assertEqual(frac(res["z"]["w"]), frac("2/1"))
        self.assertEqual(res["objectives"]["sum_z_squared"], "4/1")

    def test_missing_variable_pulled_to_zero(self):
        res, model, _ = self._check_all(MISSING)
        self.assertEqual(res["x"]["c"], "0/1", "缺测变量必须参与第二层并归零")

    def test_shuffled(self):
        self._check_all(SHUFFLED)

    def test_two_layer(self):
        self._check_all(TWO_LAYER)

    def test_grid_enumeration_two_layer(self):
        """用分数步长网格枚举零空间参数，暴力确认两层目标的全局最优。"""
        model = validate(loads(TWO_LAYER))
        eqs, _ = build_equations(model)
        reduced, piv, m = ref_solution_space(eqs, model.n)
        u0, null = ref_particular_null(reduced, piv, m, model.n)
        res = solve_model(model)
        x_star = [frac(res["x"][v]) for v in model.variables]
        z_star = [frac(res["z"][v]) for v in model.noises]
        z_obj_star = sum((v * v for v in z_star), ZERO)
        x_obj_star = sum((v * v for v in x_star), ZERO)
        k = len(null)
        # 分数步长（含 1/3）以覆盖连续最优点可能出现的分母
        grid = [Fraction(n, 3) for n in range(-6, 7)]

        def rec(idx, acc):
            if idx == k:
                u = [u0[j] + sum((null[t][j] * acc[t] for t in range(k)), ZERO)
                     for j in range(model.n)]
                xx, zz = u[:model.nx], u[model.nx:]
                zo = sum((v * v for v in zz), ZERO)
                xo = sum((v * v for v in xx), ZERO)
                # 第一层：任何可行解的 z 目标都不小于最优值
                self.assertGreaterEqual(zo, z_obj_star)
                if zo == z_obj_star:
                    # 第一层持平的解，第二层目标不小于最优值
                    self.assertGreaterEqual(xo, x_obj_star)
                return
            for g in grid:
                acc.append(g)
                rec(idx + 1, acc)
                acc.pop()
        rec(0, [])
        # 网格必须实际命中最优点（否则枚举无证明力）
        hit = False

        def rec_hit(idx, acc):
            nonlocal hit
            if hit:
                return
            if idx == k:
                u = [u0[j] + sum((null[t][j] * acc[t] for t in range(k)), ZERO)
                     for j in range(model.n)]
                if u[model.nx:] == z_star and u[:model.nx] == x_star:
                    hit = True
                return
            for g in grid:
                acc.append(g)
                rec_hit(idx + 1, acc)
                acc.pop()
        rec_hit(0, [])
        self.assertTrue(hit, "枚举网格未覆盖求解器给出的最优点")


class CorrectedValuesTests(unittest.TestCase):
    def test_corrected_excludes_noise(self):
        model = validate(loads(SHARED))
        res = solve_model(model)
        for ob in model.observations:
            expect = sum(
                (c * frac(res["x"][name]) for name, c in ob.x_terms.items()),
                ZERO)
            self.assertEqual(frac(res["corrected"][ob.id]), expect)

    def test_canonical_corrected(self):
        res = solve_text(CANONICAL)
        self.assertEqual(res["corrected"],
                         {"o1": "2/1", "o2": "4/1", "o3": "6/1"})


# ============================ 不可行与证书 ============================
def assert_valid_certificate(case, text):
    """独立核对不可行证书：原始等式的有理组合，未知量系数全零、右端为 1。"""
    model = validate(loads(text))
    res = solve_model(model)
    case.assertEqual(res["status"], "infeasible")
    case.assertNotIn("corrected", res)
    case.assertNotIn("x", res)
    case.assertNotIn("z", res)
    cert = res["certificate"]
    case.assertTrue(cert, "证书不能为空")

    eqs, ids = build_equations(model)
    by_id = dict(zip(ids, eqs))
    n = model.n
    combo_coef = [ZERO] * n
    combo_rhs = ZERO
    for eid, lambda_s in cert.items():
        lam = frac(lambda_s)
        row, rhs = by_id[eid]
        combo_coef = [a + lam * b for a, b in zip(combo_coef, row)]
        combo_rhs += lam * rhs
    case.assertEqual(combo_coef, [ZERO] * n, "证书未能消去全部未知量系数")
    case.assertEqual(combo_rhs, ONE, "证书右端必须恰为 1")
    case.assertEqual(list(cert), sorted(cert))
    for s in cert.values():
        p, q = s.split("/")
        case.assertGreater(int(q), 0)
        case.assertEqual(Fraction(int(p), int(q)).denominator, int(q))
    return res


class InfeasibleTests(unittest.TestCase):
    def _check_certificate(self, text):
        return assert_valid_certificate(self, text)

    def test_contradictory_balances(self):
        self._check_certificate(CONTRA_BAL)
        res = solve_text(CONTRA_BAL)
        self.assertEqual(res["certificate"],
                         {"c1": "-1/1", "c2": "1/1"})

    def test_contradictory_observations(self):
        self._check_certificate(CONTRA_OBS)

    def test_zero_equals_one(self):
        text = """{
          "variables": ["q"], "noises": [],
          "observations": [
            {"id": "z", "value": "1/1", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self._check_certificate(text)
        res = solve_text(text)
        self.assertEqual(res["certificate"], {"z": "1/1"})

    def test_mixed_infeasibility_certificate(self):
        # 两测量共享噪声 n：o1 与 o2 联立消去 n 得 x=2；
        # 守恒式 b1 要求 x=3。矛盾只能由“测量组合 + 守恒式”共同证出。
        text = """{
          "variables": ["x"], "noises": ["n"],
          "observations": [
            {"id": "o1", "value": "2/1", "terms": {"x": "1/1"},
             "noise_terms": {"n": "1/1"}},
            {"id": "o2", "value": "2/1", "terms": {"x": "1/1"},
             "noise_terms": {"n": "-1/1"}}
          ],
          "balances": [
            {"id": "b1", "terms": {"x": "1/1"}, "value": "3/1"}
          ]
        }"""
        self._check_certificate(text)
        res = solve_text(text)
        # 证书必须同时用到测量与守恒式两类原始等式
        self.assertIn("o1", res["certificate"])
        self.assertIn("o2", res["certificate"])
        self.assertIn("b1", res["certificate"])


# ============================ 严格校验 ============================
class ValidationTests(unittest.TestCase):
    def assert_rejected(self, text, fragment):
        with self.assertRaises((ValidationError, StrictJSONError)) as cm:
            solve_text(text)
        self.assertIn(fragment, str(cm.exception),
                      f"错误信息 {cm.exception!s} 缺少路径片段 {fragment}")

    def test_duplicate_top_level_field(self):
        text = ('{"variables": [], "variables": [], "noises": [], '
                '"observations": [], "balances": []}')
        self.assert_rejected(text, "$.variables")

    def test_duplicate_nested_field(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "id": "o2", "value": "1/1",
             "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "observations[0].id")

    def test_unknown_top_field(self):
        text = ('{"variables": [], "noises": [], "observations": [], '
                '"balances": [], "extra": 1}')
        self.assert_rejected(text, "$.extra")

    def test_unknown_nested_field(self):
        text = """{
          "variables": [], "noises": [],
          "observations": [
            {"id": "o", "value": "1/1", "terms": {},
             "noise_terms": {}, "bogus": 1}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "bogus")

    def test_missing_field(self):
        text = """{
          "variables": [], "noises": [],
          "observations": [
            {"id": "o", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "value")

    def test_bad_number_float(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": 1.5, "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "value")

    def test_bad_number_slash(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": "1/0", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "分母")

    def test_bad_number_garbage(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": "abc", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "p/q")

    def test_dangling_reference_observation(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": "1/1", "terms": {"ghost": "1/1"},
             "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "ghost")

    def test_dangling_noise_reference(self):
        text = """{
          "variables": [], "noises": ["h"],
          "observations": [
            {"id": "o", "value": "1/1", "terms": {},
             "noise_terms": {"phantom": "1/1"}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "phantom")

    def test_balance_reference_noise_rejected(self):
        text = """{
          "variables": [], "noises": ["h"],
          "observations": [],
          "balances": [
            {"id": "b", "terms": {"h": "1/1"}, "value": "1/1"}
          ]
        }"""
        self.assert_rejected(text, "h")

    def test_identifier_rules(self):
        text = ('{"variables": [""], "noises": [], "observations": [], '
                '"balances": []}')
        self.assert_rejected(text, "variables[0]")
        text = ('{"variables": ["a-b"], "noises": [], "observations": [], '
                '"balances": []}')
        self.assert_rejected(text, "variables[0]")

    def test_duplicate_identifier_in_set(self):
        text = ('{"variables": ["a", "a"], "noises": [], '
                '"observations": [], "balances": []}')
        self.assert_rejected(text, "variables[1]")

    def test_overlapping_variable_noise_namespace(self):
        text = ('{"variables": ["s"], "noises": ["s"], '
                '"observations": [], "balances": []}')
        self.assert_rejected(text, "s")

    def test_duplicate_observation_id(self):
        text = """{
          "variables": [], "noises": [],
          "observations": [
            {"id": "dup", "value": "0/1", "terms": {}, "noise_terms": {}},
            {"id": "dup", "value": "0/1", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_rejected(text, "dup")

    def test_globally_unique_balance_id(self):
        text = """{
          "variables": [], "noises": [],
          "observations": [
            {"id": "gid", "value": "0/1", "terms": {}, "noise_terms": {}}
          ],
          "balances": [
            {"id": "gid", "terms": {}, "value": "0/1"}
          ]
        }"""
        self.assert_rejected(text, "gid")

    def test_wrong_type(self):
        text = ('{"variables": "a", "noises": [], "observations": [], '
                '"balances": []}')
        self.assert_rejected(text, "$.variables")

    def test_root_must_be_object(self):
        # JSON 本身合法，但根不是对象：在模式校验阶段拒绝并定位到 $
        with self.assertRaises(ValidationError):
            validate(loads("[1,2]"))
        with self.assertRaises(ValidationError):
            validate(loads('"string"'))
        with self.assertRaises(ValidationError):
            validate(loads("null"))

    def test_json_syntax_error(self):
        with self.assertRaises(StrictJSONError):
            loads('{"a": 1,}')

    def test_non_string_json_object_key(self):
        with self.assertRaises(StrictJSONError):
            loads('{1: 2}')

    def test_trailing_data(self):
        with self.assertRaises(StrictJSONError):
            loads('{} garbage')

    def test_unicode_escape_and_surrogate_pair(self):
        # 合法 unicode 转义与代理对可以解析
        v = loads('"\\u00e9\\uD83D\\uDE00"')
        self.assertEqual(v, "é😀")
        with self.assertRaises(StrictJSONError):
            loads('"\\u12"')
        with self.assertRaises(StrictJSONError):
            loads('"\\uD800"')  # 孤立高代理

    def test_negative_fraction_ok(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": "-3/2", "terms": {"a": "-1/4"},
             "noise_terms": {}}
          ],
          "balances": []
        }"""
        res = solve_text(text)
        self.assertEqual(res["status"], "optimal")
        self.assertEqual(res["x"]["a"], "6/1")  # -a/4 = -3/2 -> a=6

    def test_fractions_are_reduced(self):
        text = """{
          "variables": ["a"], "noises": [],
          "observations": [
            {"id": "o", "value": "2/4", "terms": {"a": "1/2"},
             "noise_terms": {}}
          ],
          "balances": []
        }"""
        res = solve_text(text)  # 输入不要求约分，输出必须约分
        self.assertEqual(res["x"]["a"], "1/1")        # a/2 = 1/2 -> a=1
        self.assertEqual(res["corrected"]["o"], "1/2")  # 校正值 a/2 = 1/2
        # 分母为正且不可再约
        for s in (res["x"]["a"], res["corrected"]["o"]):
            p, q = (int(t) for t in s.split("/"))
            self.assertGreater(q, 0)
            self.assertEqual(Fraction(p, q).denominator, q)


# ============================ 严格 JSON 解析器细节 ============================
class StrictJsonTests(unittest.TestCase):
    def test_basic_roundtrip(self):
        self.assertEqual(loads('{"a": [1, 2.5, true, false, null, "x"]}'),
                         {"a": [1, 2.5, True, False, None, "x"]})

    def test_numbers(self):
        self.assertEqual(loads("0"), 0)
        self.assertEqual(loads("-17"), -17)
        self.assertEqual(loads("1.25e-3"), 1.25e-3)
        with self.assertRaises(StrictJSONError):
            loads("01")
        with self.assertRaises(StrictJSONError):
            loads("1.")
        with self.assertRaises(StrictJSONError):
            loads("1e")
        with self.assertRaises(StrictJSONError):
            loads("+1")

    def test_deep_paths(self):
        text = '{"variables": [], "noises": [], "balances": [], "observations": [{"id": "o", "value": "1/1", "terms": {"k": "1/1"}, "noise_terms": {}}]}'
        with self.assertRaises(ValidationError) as cm:
            solve_text(text)
        self.assertIn("observations[0].terms.k", str(cm.exception))


# ============================ CLI / 原子输出 ============================
class CliTests(unittest.TestCase):
    ENTRY = os.path.join(ROOT, "run_calibration.py")

    def _run(self, input_path, output_path=None):
        cmd = [sys.executable, "-I", self.ENTRY, input_path]
        if output_path is not None:
            cmd.append(output_path)
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    def test_stdout_success(self):
        with tempfile.TemporaryDirectory() as td:
            ip = os.path.join(td, "in.json")
            with open(ip, "w", encoding="utf-8") as fh:
                fh.write(CANONICAL)
            proc = self._run(ip)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["x"]["x"], "2/1")

    def test_atomic_replace_success_and_preserve_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.json")
            bad = os.path.join(td, "bad.json")
            out = os.path.join(td, "out.json")
            with open(good, "w", encoding="utf-8") as fh:
                fh.write(CANONICAL)
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('{"variables": [')  # 语法损坏
            proc = self._run(good, out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as fh:
                good_payload = fh.read()

            proc = self._run(bad, out)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("输入被拒绝", proc.stderr)
            # 旧结果完整保留
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), good_payload)

            # 语义非法同样保留旧结果
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('{"variables": [], "noises": [], '
                         '"observations": [{"id": "o", "value": "1/0", '
                         '"terms": {}, "noise_terms": {}}], "balances": []}')
            proc = self._run(bad, out)
            self.assertNotEqual(proc.returncode, 0)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), good_payload)
            # 没有遗留临时文件
            leftovers = [f for f in os.listdir(td)
                         if f.startswith(".joint_calibration.")]
            self.assertEqual(leftovers, [])

    def test_infeasible_output_has_no_corrections(self):
        with tempfile.TemporaryDirectory() as td:
            ip = os.path.join(td, "in.json")
            op = os.path.join(td, "out.json")
            with open(ip, "w", encoding="utf-8") as fh:
                fh.write(CONTRA_BAL)
            proc = self._run(ip, op)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(op, encoding="utf-8") as fh:
                payload = json.loads(fh.read())
            self.assertEqual(payload["status"], "infeasible")
            self.assertNotIn("corrected", payload)
            self.assertEqual(payload["certificate"]["c2"], "1/1")

    def test_missing_file(self):
        proc = self._run("/nonexistent/path/input.json")
        self.assertNotEqual(proc.returncode, 0)

    def test_bad_utf8_rejected_cleanly_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as td:
            bad = os.path.join(td, "bad_utf8.json")
            out = os.path.join(td, "out.json")
            with open(bad, "wb") as fh:
                fh.write(b"\xff\xfe{not utf8}")
            proc = self._run(bad, out)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("输入被拒绝", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(out))  # 从未产生输出

    def test_bom_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "bom.json")
            with open(p, "wb") as fh:
                fh.write(b"\xef\xbb\xbf{}")
            proc = self._run(p)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("输入被拒绝", proc.stderr)

    def test_usage_error(self):
        proc = subprocess.run(
            [sys.executable, "-I", self.ENTRY],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 2)


# ============================ 随机性质测试 ============================
class RandomizedPropertyTests(unittest.TestCase):
    """随机生成系统，独立核对一致性、两层最优性与证书。"""

    def test_all_free_no_equations(self):
        # 无任何等式：所有 z 与 x 均为自由量，两层最小化把它们全部归零
        text = """{
          "variables": ["f1", "f2", "f3"],
          "noises": ["g1", "g2"],
          "observations": [],
          "balances": []
        }"""
        res = solve_text(text)
        self.assertEqual(res["status"], "optimal")
        self.assertEqual(set(res["x"].values()), {"0/1"})
        self.assertEqual(set(res["z"].values()), {"0/1"})
        self.assertEqual(res["corrected"], {})
        self.assertEqual(res["objectives"],
                         {"sum_z_squared": "0/1", "sum_x_squared": "0/1"})
        assert_two_layer_optimal(self, text)

    def _random_model(self, rng, force_infeasible):
        """生成一个随机模型；force_infeasible 时保证构造出矛盾。"""
        nx = rng.randint(1, 4)
        nz = rng.randint(0, 3)
        nobs = rng.randint(0, 5)
        nbal = rng.randint(0, 3)
        if force_infeasible and nx >= 1:
            nbal = max(nbal, 2)
        variables = [f"x{i}" for i in range(nx)]
        noises = [f"z{j}" for j in range(nz)]

        def coef_obj(names):
            obj = {}
            for nm in names:
                if rng.random() < 0.6:
                    obj[nm] = f"{rng.randint(-3, 3)}/1"
            return obj

        # 随机选取“真实”未知数，由其生成自洽右端，保证一致系统
        true_x = [Fraction(rng.randint(-2, 2), 1) for _ in range(nx)]
        true_z = [Fraction(rng.randint(-2, 2), 1) for _ in range(nz)]
        observations, balances = [], []
        for i in range(nobs):
            xt = coef_obj(variables)
            zt = coef_obj(noises)
            rhs = ZERO
            for nm, c in xt.items():
                rhs += Fraction(c) * true_x[int(nm[1:])]
            for nm, c in zt.items():
                rhs += Fraction(c) * true_z[int(nm[1:])]
            observations.append({
                "id": f"o{i}", "value": f"{rhs.numerator}/{rhs.denominator}",
                "terms": xt, "noise_terms": zt,
            })
        for i in range(nbal):
            xt = coef_obj(variables)
            rhs = sum((Fraction(c) * true_x[int(nm[1:])]
                       for nm, c in xt.items()), ZERO)
            balances.append({
                "id": f"b{i}",
                "terms": xt,
                "value": f"{rhs.numerator}/{rhs.denominator}",
            })
        model_dict = {
            "variables": variables, "noises": noises,
            "observations": observations, "balances": balances,
        }
        if force_infeasible:
            if rng.random() < 0.5:
                # 0=1：未知量系数全空、右端为 1，必然矛盾
                balances.append({
                    "id": "binfeas", "terms": {}, "value": "1/1",
                })
            else:
                # 对同一真实量追加两条互相冲突的守恒式（v=0 与 v=1），
                # 与噪声无关，必然矛盾
                v = variables[0]
                balances.append({"id": "bcfa", "terms": {v: "1/1"},
                                 "value": "0/1"})
                balances.append({"id": "bcfb", "terms": {v: "1/1"},
                                 "value": "1/1"})
        return json.dumps(model_dict, ensure_ascii=False)

    def test_random_consistent_systems(self):
        import random
        rng = random.Random(20261112)
        for _ in range(120):
            text = self._random_model(rng, force_infeasible=False)
            assert_two_layer_optimal(self, text)

    def test_random_infeasible_systems_certificate(self):
        import random
        rng = random.Random(987654321)
        for _ in range(60):
            text = self._random_model(rng, force_infeasible=True)
            assert_valid_certificate(self, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
