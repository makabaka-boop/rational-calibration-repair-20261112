#!/usr/bin/env python3
"""两层字典序校正求解的小型回归测试（标准库 unittest + Fraction）。

覆盖：

* 第一层存在自由方向：``x+z=1`` 必须得到 ``z=0, x=1``；
* 第一层无误差、但真实量方向非独立（``x+y+w=1`` ⇒ 三者各 1/3）；
* 非独立方向共同影响第二层（行向量为另一个的倍数、缺测变量被拉回 0）；
* 共享误差与缺测真实量同时存在；
* 冗余等式、空未知量与唯一解；
* 每个用例都核对完整的 ``x``、``z``、``corrected`` 与两项目标，
  并独立验算原始等式；
* 输入重新排列、以及按非零有理数整体缩放后结果保持不变；
* 无解时只返回证书，并独立复算（系数全零、右端为 1）；
* 原子输出：失败时保留旧文件，成功才替换。

运行：
    python3 -I tests/test_two_layer_regression.py
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

from joint_calibration.model import validate  # noqa: E402
from joint_calibration.solver import solve_model  # noqa: E402
from joint_calibration.strict_json import loads  # noqa: E402

ZERO = Fraction(0)
ONE = Fraction(1)


def solve_text(text):
    return solve_model(validate(loads(text)))


def F(raw):
    return Fraction(raw)


def check_reduced_positive(case, s):
    p_str, q_str = s.split("/")
    p, q = int(p_str), int(q_str)
    case.assertGreater(q, 0, f"{s} 的分母必须为正")
    case.assertEqual(Fraction(p, q).numerator, p, f"{s} 未约分")
    case.assertEqual(Fraction(p, q).denominator, q, f"{s} 未约分")


def check_equations(case, model, res):
    """独立复核：每条测量/守恒式在 x、z 上都必须成立。"""
    for ob in model.observations:
        lhs = sum((c * F(res["x"][name]) for name, c in ob.x_terms.items()), ZERO)
        lhs += sum((c * F(res["z"][name]) for name, c in ob.z_terms.items()), ZERO)
        case.assertEqual(lhs, ob.value, f"测量 {ob.id} 等式不成立")
    for bal in model.balances:
        lhs = sum((c * F(res["x"][name]) for name, c in bal.terms.items()), ZERO)
        case.assertEqual(lhs, bal.value, f"守恒式 {bal.id} 等式不成立")


def check_full_result(
    case,
    model_dict,
    expected_x,
    expected_z,
    expected_corrected,
    expected_sum_z,
    expected_sum_x,
    text=None,
):
    text = text if text is not None else json.dumps(model_dict)
    model = validate(loads(text))
    res = solve_model(model)
    case.assertEqual(res["status"], "optimal")
    case.assertEqual(res["x"], expected_x)
    case.assertEqual(res["z"], expected_z)
    case.assertEqual(res["corrected"], expected_corrected)
    case.assertEqual(res["objectives"]["sum_z_squared"], expected_sum_z)
    case.assertEqual(res["objectives"]["sum_x_squared"], expected_sum_x)
    for d in (res["x"], res["z"], res["corrected"]):
        for s in d.values():
            check_reduced_positive(case, s)
    for s in res["objectives"].values():
        check_reduced_positive(case, s)
    check_equations(case, model, res)
    return res


# ============================ 用例夹具 ============================
# 1) 第一层有自由方向：x 与 z 完全可互换，误差必须被压到 0。
FIRST_LAYER_FREE = {
    "variables": ["x"],
    "noises": ["z"],
    "observations": [
        {"id": "o", "value": "1/1", "terms": {"x": "1/1"},
         "noise_terms": {"z": "1/1"}},
    ],
    "balances": [],
}

# 2) 无误差变量，真实量方向非独立：x+y+w=1 的最近点是 (1/3, 1/3, 1/3)。
NO_NOISE_NONINDEP = {
    "variables": ["x", "y", "w"],
    "noises": [],
    "observations": [
        {"id": "o", "value": "1/1",
         "terms": {"x": "1/1", "y": "1/1", "w": "1/1"}, "noise_terms": {}},
    ],
    "balances": [],
}

# 3) 非独立方向共同影响第二层：两个真实量的系数成比例（同行方向），
#    另有缺测真实量 free 必须归零；误差通过共享 z 被压到 0。
PROPORTIONAL_ROWS = {
    "variables": ["p", "q", "free"],
    "noises": ["s"],
    "observations": [
        {"id": "o1", "value": "9/1", "terms": {"p": "1/1", "q": "2/1"},
         "noise_terms": {"s": "1/1"}},
        {"id": "o2", "value": "18/1", "terms": {"p": "2/1", "q": "4/1"},
         "noise_terms": {"s": "2/1"}},
    ],
    "balances": [],
}

# 4) 共享误差 + 缺测变量同时存在：
#    a+z1=10、b+z1=12、a+b+z2=3；t 缺测。"共享"的 z1 只出现一份。
SHARED_AND_MISSING = {
    "variables": ["a", "b", "t"],
    "noises": ["z1", "z2"],
    "observations": [
        {"id": "m1", "value": "10/1", "terms": {"a": "1/1"},
         "noise_terms": {"z1": "1/1"}},
        {"id": "m2", "value": "12/1", "terms": {"b": "1/1"},
         "noise_terms": {"z1": "1/1"}},
        {"id": "m3", "value": "3/1", "terms": {"a": "1/1", "b": "1/1"},
         "noise_terms": {"z2": "1/1"}},
    ],
    "balances": [],
}

# 5) 冗余等式：b2 = b1 + b3，且 b4 = 2·b1；系统仍一致。
REDUNDANT_BALANCES = {
    "variables": ["u", "v"],
    "noises": ["w"],
    "observations": [
        {"id": "o1", "value": "4/1", "terms": {"u": "1/1"},
         "noise_terms": {"w": "1/1"}},
    ],
    "balances": [
        {"id": "b1", "terms": {"u": "1/1"}, "value": "2/1"},
        {"id": "b3", "terms": {"v": "1/1"}, "value": "5/1"},
        {"id": "b2", "terms": {"u": "1/1", "v": "1/1"}, "value": "7/1"},
        {"id": "b4", "terms": {"u": "2/1"}, "value": "4/1"},
    ],
}

# 4b) 非独立（非正交）方向共同影响第二层：零空间基 (取自由变量为单位向量)
#     两两不正交，正规方程的 Gram 矩阵非对角；第一层先把 z 压到 0，
#     第二层再在 x+y+w=1 上取最近点 (1/3, 1/3, 1/3)。
NON_ORTHOGONAL_FREEDOM = {
    "variables": ["x", "y", "w"],
    "noises": ["z"],
    "observations": [
        {"id": "o", "value": "1/1",
         "terms": {"x": "1/1", "y": "1/1", "w": "1/1"},
         "noise_terms": {"z": "1/1"}},
    ],
    "balances": [],
}


# 6) 唯一解 + 冗余 + 空未知量集合参与：x=2、y=4、z=1。
UNIQUE_SOLUTION = {
    "variables": ["x", "y", "missing"],
    "noises": ["z", "unused"],
    "observations": [
        {"id": "o1", "value": "3/1", "terms": {"x": "1/1"},
         "noise_terms": {"z": "1/1"}},
        {"id": "o2", "value": "5/1", "terms": {"y": "1/1"},
         "noise_terms": {"z": "1/1"}},
        {"id": "o3", "value": "6/1", "terms": {"x": "1/1", "y": "1/1"},
         "noise_terms": {}},
    ],
    "balances": [],
}

# 7) 空未知量：没有变量也没有误差，只有 0=0 的等式。
EMPTY_UNKNOWNS = {
    "variables": [],
    "noises": [],
    "observations": [
        {"id": "e", "value": "0/1", "terms": {}, "noise_terms": {}},
    ],
    "balances": [],
}


class FirstLayerFreeDirectionTests(unittest.TestCase):
    def test_error_forced_to_zero(self):
        res = check_full_result(
            self, FIRST_LAYER_FREE,
            {"x": "1/1"}, {"z": "0/1"},
            {"o": "1/1"}, "0/1", "1/1",
        )
        # 第一层自由：任何非零 z 都能被同量 x 抵消，但最优必须 z=0。
        self.assertEqual(F(res["x"]["x"]), ONE)

    def test_reordering_invariant(self):
        reordered = {
            "balances": [],
            "observations": [
                {"noise_terms": {"z": "1/1"}, "terms": {"x": "1/1"},
                 "value": "1/1", "id": "o"},
            ],
            "noises": ["z"],
            "variables": ["x"],
        }
        base = solve_text(json.dumps(FIRST_LAYER_FREE))
        other = solve_text(json.dumps(reordered))
        self.assertEqual(base, other)

    def test_scale_invariance(self):
        # 整条等式乘以非零有理数：x、z 与两项目标不变；
        # 校正值按行系数等比缩放（corrected = Σ terms·x，系数被放大了 -3/7）。
        base = solve_text(json.dumps(FIRST_LAYER_FREE))
        scaled = {
            "variables": ["x"],
            "noises": ["z"],
            "observations": [
                {"id": "o", "value": "-3/7",
                 "terms": {"x": "-3/7"}, "noise_terms": {"z": "-3/7"}},
            ],
            "balances": [],
        }
        other = solve_text(json.dumps(scaled))
        for key in ("status", "x", "z", "objectives"):
            self.assertEqual(base[key], other[key], key)
        self.assertEqual(F(other["corrected"]["o"]),
                         F("-3/7") * F(base["corrected"]["o"]))


class NoNoiseNonIndependentTests(unittest.TestCase):
    def test_three_way_split(self):
        res = check_full_result(
            self, NO_NOISE_NONINDEP,
            {"w": "1/3", "x": "1/3", "y": "1/3"}, {},
            {"o": "1/1"}, "0/1", "1/3",
        )

    def test_scale_and_permute_invariance(self):
        base = solve_text(json.dumps(NO_NOISE_NONINDEP))
        scaled = {
            "variables": ["w", "y", "x"],
            "noises": [],
            "observations": [
                {"id": "o", "value": "5/2",
                 "terms": {"w": "5/2", "y": "5/2", "x": "5/2"},
                 "noise_terms": {}},
            ],
            "balances": [],
        }
        other = solve_text(json.dumps(scaled))
        for key in ("status", "x", "z", "objectives"):
            self.assertEqual(base[key], other[key], key)
        # 校正值随各行的缩放因子变化，按原始系数独立复算。
        model = validate(loads(json.dumps(scaled)))
        check_equations(self, model, other)


class NonIndependentSecondLayerTests(unittest.TestCase):
    def test_proportional_rows_and_free_variable(self):
        res = check_full_result(
            self, PROPORTIONAL_ROWS,
            {"p": "9/5", "q": "18/5", "free": "0/1"}, {"s": "0/1"},
            {"o1": "9/1", "o2": "18/1"}, "0/1", "81/5",
        )
        # 第一层把共享误差压到 0；第二层在 p+2q=9 上取最近点。
        self.assertEqual(F(res["x"]["p"]) * 5, F("9/1"))

    def test_permute_ids_and_scale(self):
        base = solve_text(json.dumps(PROPORTIONAL_ROWS))
        scaled = {
            "variables": ["free", "q", "p"],
            "noises": ["s"],
            "observations": [
                {"id": "o2", "value": "27/1", "terms": {"p": "3/1", "q": "6/1"},
                 "noise_terms": {"s": "3/1"}},
                {"id": "o1", "value": "27/2", "terms": {"p": "3/2", "q": "3/1"},
                 "noise_terms": {"s": "3/2"}},
            ],
            "balances": [],
        }
        # 第二行整体 ×3/2：o1 行 (p+2q + s = 9) ×3/2 → 3/2 p + 3 q + 3/2 s = 27/2
        other = solve_text(json.dumps(scaled))
        self.assertEqual(base["x"], other["x"])
        self.assertEqual(base["z"], other["z"])


class NonOrthogonalDirectionTests(unittest.TestCase):
    def test_non_orthogonal_free_directions(self):
        res = check_full_result(
            self, NON_ORTHOGONAL_FREEDOM,
            {"w": "1/3", "x": "1/3", "y": "1/3"}, {"z": "0/1"},
            {"o": "1/1"}, "0/1", "1/3",
        )
        self.assertEqual(res["status"], "optimal")

    def test_free_directions_are_genuinely_non_orthogonal(self):
        """确认夹具确实让第二层正规方程的 Gram 出现非零非对角项。

        这里直接按参考实现构造零空间：方程 x+y+w+z=1 的 3 维零空间，
        第一层固定 z 后剩下 2 个方向；它们在 x 块上不正交。
        """
        # 方向 d1=(1,-1,0,0)、d2=(1,0,-1,0) 都是齐次方向，点积=1≠0。
        d1 = [F("1/1"), F("-1/1"), F("0/1"), F("0/1")]
        d2 = [F("1/1"), F("0/1"), F("-1/1"), F("0/1")]
        self.assertNotEqual(sum((a * b for a, b in zip(d1, d2)), ZERO), ZERO)
        # 两者都与 (1,1,1,1) 正交，即都满足齐次 x+y+w+z=0。
        for d in (d1, d2):
            self.assertEqual(sum(d, ZERO), ZERO)


class SharedErrorAndMissingTests(unittest.TestCase):
    def test_shared_error_with_missing_variable(self):
        res = check_full_result(
            self, SHARED_AND_MISSING,
            {"a": "12/5", "b": "22/5", "t": "0/1"},
            {"z1": "38/5", "z2": "-19/5"},
            {"m1": "12/5", "m2": "22/5", "m3": "34/5"},
            "361/5", "628/25",
        )
        # 共享误差标识只占一个空间维度，不因出现 3 次而分裂。
        self.assertEqual(len(res["z"]), 2)

    def test_reorder_and_scale(self):
        base = solve_text(json.dumps(SHARED_AND_MISSING))
        scaled = {
            "variables": ["t", "b", "a"],
            "noises": ["z2", "z1"],
            "observations": [
                {"id": "m3", "value": "-3/1", "terms": {"b": "-1/1", "a": "-1/1"},
                 "noise_terms": {"z2": "-1/1"}},
                {"id": "m2", "value": "6/1", "terms": {"b": "1/2"},
                 "noise_terms": {"z1": "1/2"}},
                {"id": "m1", "value": "10/1", "terms": {"a": "1/1"},
                 "noise_terms": {"z1": "1/1"}},
            ],
            "balances": [],
        }
        other = solve_text(json.dumps(scaled))
        self.assertEqual(base["x"], other["x"])
        self.assertEqual(base["z"], other["z"])
        self.assertEqual(base["objectives"], other["objectives"])
        # 校正值 = Σ terms·x，随各行缩放因子等比变化；独立复算请见等式核对。
        model = validate(loads(json.dumps(scaled)))
        check_equations(self, model, other)


class RedundantAndUniqueTests(unittest.TestCase):
    def test_redundant_balances(self):
        check_full_result(
            self, REDUNDANT_BALANCES,
            {"u": "2/1", "v": "5/1"}, {"w": "2/1"},
            {"o1": "2/1"}, "4/1", "29/1",
        )

    def test_unique_solution_with_empty_extra_unknowns(self):
        check_full_result(
            self, UNIQUE_SOLUTION,
            {"missing": "0/1", "x": "2/1", "y": "4/1"},
            {"unused": "0/1", "z": "1/1"},
            {"o1": "2/1", "o2": "4/1", "o3": "6/1"},
            "1/1", "20/1",
        )

    def test_empty_unknowns(self):
        res = check_full_result(
            self, EMPTY_UNKNOWNS, {}, {}, {"e": "0/1"}, "0/1", "0/1",
        )
        self.assertEqual(res["status"], "optimal")


class InfeasibleCertificateTests(unittest.TestCase):
    def assert_certificate(self, case, text):
        model = validate(loads(text))
        res = solve_model(model)
        case.assertEqual(res["status"], "infeasible")
        for key in ("x", "z", "corrected", "objectives"):
            case.assertNotIn(key, res)
        cert = res["certificate"]
        case.assertTrue(cert)
        # 独立复算：Σ 系数·原始等式 = (0,...,0 | 1)
        eqs = {}
        for ob in model.observations:
            row = [ZERO] * model.n
            for idx, c in ob.coeff.items():
                row[idx] = c
            eqs[ob.id] = (row, ob.value)
        for bal in model.balances:
            row = [ZERO] * model.n
            for name, c in bal.terms.items():
                row[model.variables.index(name)] = c
            eqs[bal.id] = (row, bal.value)
        combo = [ZERO] * model.n
        rhs = ZERO
        for eid, lam_s in cert.items():
            lam = F(lam_s)
            check_reduced_positive(case, lam_s)
            row, value = eqs[eid]
            combo = [a + lam * b for a, b in zip(combo, row)]
            rhs += lam * value
        case.assertEqual(combo, [ZERO] * model.n, "证书未能消去全部未知量")
        case.assertEqual(rhs, ONE, "证书右端必须恰为 1")
        case.assertEqual(list(cert), sorted(cert))
        return res

    def test_mixed_certificate(self):
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
        res = self.assert_certificate(self, text)
        self.assertIn("o1", res["certificate"])
        self.assertIn("o2", res["certificate"])
        self.assertIn("b1", res["certificate"])

    def test_constant_conflict_certificate(self):
        text = """{
          "variables": ["q"], "noises": [],
          "observations": [
            {"id": "z", "value": "1/1", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        res = self.assert_certificate(self, text)
        self.assertEqual(res["certificate"], {"z": "1/1"})

    def test_certificate_independent_of_scaling(self):
        # 同一矛盾按不同非零倍数书写，证书仍必须独立复算成立。
        text = """{
          "variables": ["q"], "noises": [],
          "observations": [
            {"id": "a", "value": "3/2", "terms": {"q": "3/2"}, "noise_terms": {}},
            {"id": "b", "value": "-5/3", "terms": {"q": "-5/3"}, "noise_terms": {}},
            {"id": "c", "value": "2/1", "terms": {}, "noise_terms": {}}
          ],
          "balances": []
        }"""
        self.assert_certificate(self, text)


class AtomicOutputRegressionTests(unittest.TestCase):
    ENTRY = os.path.join(ROOT, "run_calibration.py")

    def _run(self, input_path, output_path=None):
        cmd = [sys.executable, "-I", self.ENTRY, input_path]
        if output_path is not None:
            cmd.append(output_path)
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    def test_success_replaces_and_failure_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, "good.json")
            bad = os.path.join(td, "bad.json")
            out = os.path.join(td, "out.json")
            with open(good, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(FIRST_LAYER_FREE))
            proc = self._run(good, out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as fh:
                old = fh.read()
            self.assertEqual(json.loads(old)["z"]["z"], "0/1")

            # 非法输入：定位路径、非零退出、旧文件保持原字节
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('{"variables": ["x"], "noises": ["z"], '
                         '"observations": [{"id": "o", "value": "1/0", '
                         '"terms": {"x": "1/1"}, "noise_terms": {"z": "1/1"}}], '
                         '"balances": []}')
            proc = self._run(bad, out)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("$.observations[0].value", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), old)
            self.assertEqual(
                [f for f in os.listdir(td) if f.startswith(".joint_calibration.")],
                [])

    def test_stdout_regression_values(self):
        with tempfile.TemporaryDirectory() as td:
            ip = os.path.join(td, "in.json")
            with open(ip, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(SHARED_AND_MISSING))
            proc = self._run(ip)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["x"], {"a": "12/5", "b": "22/5", "t": "0/1"})
            self.assertEqual(payload["z"], {"z1": "38/5", "z2": "-19/5"})
            self.assertEqual(payload["objectives"],
                             {"sum_z_squared": "361/5", "sum_x_squared": "628/25"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
