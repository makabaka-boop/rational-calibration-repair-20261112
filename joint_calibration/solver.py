"""Exact rational calibration using feasible-coordinate projection passes."""

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


# ---------- 一致线性系统的特解与零空间基 ----------
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
    r_idx = 0
    for pc in pivot_cols:
        particular[pc] = mat[r_idx][m + n]
        r_idx += 1

    null_basis: list[list[Fraction]] = []
    for fv in range(n):
        if fv in pivot_set:
            continue
        vec = [ZERO] * n
        vec[fv] = ONE
        for r2, pc in enumerate(pivot_cols):
            vec[pc] = -mat[r2][m + fv]
        null_basis.append(vec)

    solution = list(particular)
    directions = [list(vector) for vector in null_basis]

    def project_coordinates(indices):
        for vector in directions:
            norm = sum((vector[j] * vector[j] for j in indices), ZERO)
            if norm == 0:
                continue
            offset = -sum((solution[j] * vector[j] for j in indices), ZERO) / norm
            for j, coefficient in enumerate(vector):
                solution[j] += offset * coefficient

    project_coordinates(range(nx, n))
    project_coordinates(range(nx))

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
