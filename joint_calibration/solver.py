"""Exact rational calibration with a two-layer (lexicographic) optimisation.

求解模型::

    A x + B z = b

其中 ``x`` 是真实量、``z`` 是共享标定误差。目标严格分两层：

1. 先在所有可行解中最小化 ``||z||²``；
2. 再在第一层的**全部**最优解中最小化 ``||x||²``（缺测的真实量同样计入）。

两层绝不加权合并。所有运算使用 :class:`fractions.Fraction`，不调用任何
代数/优化求解库。

实现要点：先由带来源记录的 RREF 得到特解 ``u0`` 与零空间基 ``N``，于是全部
可行解为 ``u(t) = u0 + N t``。把 ``N`` 按 ``z`` / ``x`` 分块：

* 第一层是 ``min_t ||z0 + Nz t||²``，其正规方程 ``(NzᵀNz) t = -Nzᵀz0``；
  取一个精确解 ``t0``，第一层最优解集恰为 ``t0 + null(Nz)``。
* 第二层把 ``t = t0 + T s``（``T`` 是 ``null(Nz)`` 的基）代入，
  得 ``min_s ||x0 + Nx T s||²``，再解一次正规方程即可。

正规方程在有理数域上必然相容（其右端位于矩阵列空间内），若由于程序缺陷
而出现不相容，会抛出 :class:`ArithmeticError` 而不是冒充“无解”。
"""

from __future__ import annotations

from fractions import Fraction

from .model import Balance, Model, Observation, fraction_to_str

ZERO = Fraction(0)
ONE = Fraction(1)


# ---------- 高斯-若尔当消元（返回主元列） ----------
def gauss_jordan(mat: list[list[Fraction]], col_start: int,
                 col_end: int) -> list[int]:
    """原地 RREF。主元只在 ``[col_start, col_end)`` 列中选取。

    主元列之外的前导列（如证书记录块 I_m）随消元一起变换但不参与选主元。
    返回全局列号形式的主元列列表。
    """
    if not mat:
        return []
    row_count = len(mat)
    pivot_row = 0
    pivot_cols: list[int] = []
    for col in range(col_start, col_end):
        candidate = None
        for r in range(pivot_row, row_count):
            if mat[r][col] != 0:
                candidate = r
                break
        if candidate is None:
            continue
        mat[pivot_row], mat[candidate] = mat[candidate], mat[pivot_row]
        pivot = mat[pivot_row][col]
        if pivot != ONE:
            mat[pivot_row] = [v / pivot for v in mat[pivot_row]]
        for r in range(row_count):
            if r == pivot_row:
                continue
            factor = mat[r][col]
            if factor != 0:
                mat[r] = [
                    a - factor * b
                    for a, b in zip(mat[r], mat[pivot_row])
                ]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_cols


# ---------- 精确线性方程组：特解与零空间基 ----------
def _solve_consistent(
    matrix: list[list[Fraction]],
    rhs: list[Fraction],
    ncols: int,
) -> tuple[list[Fraction], list[list[Fraction]], list[int]] | None:
    """解 ``matrix · v = rhs``（方阵或任意行数）。

    返回 ``(特解, RREF 增广矩阵, 主元列)``；特解令自由变量为 0。
    若方程组不相容则返回 ``None``。
    """
    aug = [list(row) + [value] for row, value in zip(matrix, rhs)]
    pivots = gauss_jordan(aug, 0, ncols)
    for row in aug:
        if all(row[j] == 0 for j in range(ncols)) and row[ncols] != 0:
            return None
    solution = [ZERO] * ncols
    for r, pc in enumerate(pivots):
        solution[pc] = aug[r][ncols]
    return solution, aug, pivots


def _null_space(
    reduced: list[list[Fraction]],
    pivots: list[int],
    ncols: int,
) -> list[list[Fraction]]:
    """由 RREF 结果读出零空间的一组基（自由变量取单位向量）。"""
    pivot_set = set(pivots)
    basis: list[list[Fraction]] = []
    for free in range(ncols):
        if free in pivot_set:
            continue
        vector = [ZERO] * ncols
        vector[free] = ONE
        for r, pc in enumerate(pivots):
            vector[pc] = -reduced[r][free]
        basis.append(vector)
    return basis


# ---------- 两层最小二乘 ----------
def _normal_equations(
    basis: list[list[Fraction]],
    offset: list[Fraction],
    indices: range,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    """对 ``min_t ||offset + Σ_q t_q·basis_q||²`` 在给定坐标块上建正规方程。

    ``basis_q`` 是完整的 n 维方向；``indices`` 选出参与本层的坐标（z 块或
    x 块）。返回 ``(Gram, rhs)``，即 ``Gram · t = rhs``，其中
    ``Gram[i][j] = Σ_{k ∈ indices} basis_i[k]·basis_j[k]``、
    ``rhs[i] = -Σ_{k ∈ indices} offset[k]·basis_i[k]``。
    """
    k = len(basis)
    gram = [
        [sum((basis[i][j] * basis[jj][j] for j in indices), ZERO)
         for jj in range(k)]
        for i in range(k)
    ]
    rhs = [
        -sum((offset[j] * basis[i][j] for j in indices), ZERO)
        for i in range(k)
    ]
    return gram, rhs


def _minimise_coefficients(
    offset: list[Fraction],
    basis: list[list[Fraction]],
    indices: range,
) -> list[Fraction]:
    """精确求 ``min ||offset + Σ c_q basis_q||²``（只计 ``indices`` 坐标）。

    返回最优组合系数 ``c``（正规方程有自由参数时取 0）。该坐标块上的最优值
    唯一；返回哪个最优点不影响 ``indices`` 坐标上的数值。
    """
    if not basis:
        return []
    gram, rhs = _normal_equations(basis, offset, indices)
    solved = _solve_consistent(gram, rhs, len(basis))
    if solved is None:  # 正规方程必然相容；触发即程序缺陷
        raise ArithmeticError("内部错误：正规方程不相容（不应发生）")
    coefficients, _, _ = solved
    return coefficients


def _combine(
    base: list[Fraction],
    basis: list[list[Fraction]],
    coefficients: list[Fraction],
) -> list[Fraction]:
    result = list(base)
    for coefficient, vector in zip(coefficients, basis):
        if coefficient != 0:
            for j, value in enumerate(vector):
                result[j] += coefficient * value
    return result


def solve_model(model: Model) -> dict:
    nx, nz, n = model.nx, model.nz, model.n
    x_index = {name: i for i, name in enumerate(model.variables)}
    equations: list[Observation | Balance] = (
        list(model.observations) + list(model.balances)
    )
    m = len(equations)

    # 增广矩阵 [ I_m | E | b ]：I_m 块随消元记录原始等式的线性组合。
    mat: list[list[Fraction]] = []
    for i, eq in enumerate(equations):
        row = [ZERO] * m
        row[i] = ONE
        row.extend(ZERO for _ in range(n))
        if isinstance(eq, Observation):
            for idx, coef in eq.coeff.items():
                row[m + idx] = coef
        else:
            for name, coef in eq.terms.items():
                row[m + x_index[name]] = coef
        row.append(eq.value)
        mat.append(row)

    pivot_cols_global = gauss_jordan(mat, m, m + n)
    pivot_cols = [c - m for c in pivot_cols_global]
    pivot_set = set(pivot_cols)

    # ---- 可行性：存在 [组合 | 0...0 | 非零 RHS] 行即不可行 ----
    bad_row = None
    for row in mat:
        if all(row[m + j] == 0 for j in range(n)) and row[m + n] != 0:
            bad_row = row
            break
    if bad_row is not None:
        rhs = bad_row[m + n]
        certificate: dict[str, str] = {}
        for i, eq in enumerate(equations):
            coef = bad_row[i] / rhs  # 归一化使组合后的右端恰为 1
            if coef != 0:
                certificate[eq.id] = fraction_to_str(coef)
        return {
            "status": "infeasible",
            "certificate": dict(sorted(certificate.items())),
        }

    # ---- 特解 u0 与零空间基 N（按未知量块读取 RREF） ----
    particular = [ZERO] * n
    for r_idx, pc in enumerate(pivot_cols):
        particular[pc] = mat[r_idx][m + n]

    null_basis: list[list[Fraction]] = []
    for free in range(n):
        if free in pivot_set:
            continue
        vector = [ZERO] * n
        vector[free] = ONE
        for r2, pc in enumerate(pivot_cols):
            vector[pc] = -mat[r2][m + free]
        null_basis.append(vector)

    z_indices = range(nx, n)
    x_indices = range(0, nx)

    # ---- 第一层：min Σ z² ----
    # 可行解族 u(t) = u0 + Σ t_q N_q；只在 z 坐标块上做仿射投影。
    t_coefficients = _minimise_coefficients(particular, null_basis, z_indices)
    t_star = _combine(particular, null_basis, t_coefficients)

    # 第一层最优解集：参数 t 可沿 null(Nz) 方向移动（Nz 是 N 的 z 坐标分块）。
    # 求 null(Nz)：把 N 的 z 坐标堆成矩阵，再取零空间。
    z_rows = len(null_basis)
    nz_matrix = [
        [null_basis[q][idx] for q in range(z_rows)]
        for idx in z_indices
    ]
    if nz_matrix:
        nz_pivots = gauss_jordan(nz_matrix, 0, z_rows)
        first_layer_dirs = _null_space(nz_matrix, nz_pivots, z_rows)
    else:
        # 没有任何 z 坐标：第一层无约束，全部零空间方向都留在最优集中。
        first_layer_dirs = [
            [ONE if i == q else ZERO for i in range(z_rows)]
            for q in range(z_rows)
        ]

    # ---- 第二层：在第一层最优集内 min Σ x²（缺测真实量同样计入） ----
    # 由 t = t_coeff + Σ s_j T_j 生成 x 坐标块上的方向 N·T_j。
    x_basis: list[list[Fraction]] = []
    for direction in first_layer_dirs:
        vector = [ZERO] * n
        for q in range(z_rows):
            coefficient = direction[q]
            if coefficient != 0:
                source = null_basis[q]
                for j in range(n):
                    vector[j] += coefficient * source[j]
        x_basis.append(vector)

    s_coefficients = _minimise_coefficients(t_star, x_basis, x_indices)
    solution = _combine(t_star, x_basis, s_coefficients)

    x_sol = solution[:nx]
    z_sol = solution[nx:]

    # ---- 输出（按标识排序，全部为约分正分母有理数） ----
    x_out = {
        name: fraction_to_str(x_sol[i])
        for i, name in enumerate(model.variables)
    }
    z_out = {
        name: fraction_to_str(z_sol[i])
        for i, name in enumerate(model.noises)
    }
    corrected: dict[str, str] = {}
    for obs in model.observations:
        value = sum(
            (coef * x_sol[x_index[name]]
             for name, coef in obs.x_terms.items()),
            ZERO,
        )
        corrected[obs.id] = fraction_to_str(value)

    z_obj = sum((v * v for v in z_sol), ZERO)
    x_obj = sum((v * v for v in x_sol), ZERO)
    return {
        "status": "optimal",
        "x": dict(sorted(x_out.items())),
        "z": dict(sorted(z_out.items())),
        "corrected": dict(sorted(corrected.items())),
        "objectives": {
            "sum_z_squared": fraction_to_str(z_obj),
            "sum_x_squared": fraction_to_str(x_obj),
        },
    }
