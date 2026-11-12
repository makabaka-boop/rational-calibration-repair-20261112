"""输入模式校验与求解数据模型（全部基于精确有理数）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9]+$")
_NUM_RE = re.compile(r"^[+-]?[0-9]+/[0-9]+$")
_TOP_KEYS = ("variables", "noises", "observations", "balances")


class ValidationError(ValueError):
    """输入不合法，message 以 JSONPath 风格路径定位问题。"""


# ---------- 有理数 ----------
def parse_fraction(raw: Any, path: str) -> Fraction:
    """解析 p/q 字符串：p 为整数、q 为正整数；拒绝浮点与一切非法形式。"""
    if not isinstance(raw, str) or not _NUM_RE.fullmatch(raw):
        raise ValidationError(
            f"{path} 必须是 'p/q' 形式的字符串（p 为整数、q 为正整数），"
            f"实际为 {raw!r}"
        )
    p_str, q_str = raw.split("/")
    p = int(p_str)
    q = int(q_str)
    if q <= 0:
        raise ValidationError(f"{path} 的分母必须为正整数，实际为 {q}")
    return Fraction(p, q)


def fraction_to_str(value: Fraction) -> str:
    """Fraction -> 约分后正分母的 'p/q' 字符串。"""
    return f"{value.numerator}/{value.denominator}"


# ---------- 数据模型 ----------
@dataclass
class Observation:
    id: str
    value: Fraction
    # 系数按内部未知量索引存储：x 索引 [0, nx)，z 索引 [nx, nx+nz)
    coeff: dict[int, Fraction]
    x_terms: dict[str, Fraction] = field(default_factory=dict)
    z_terms: dict[str, Fraction] = field(default_factory=dict)


@dataclass
class Balance:
    id: str
    value: Fraction
    terms: dict[str, Fraction]


@dataclass
class Model:
    variables: list[str]
    noises: list[str]
    observations: list[Observation]
    balances: list[Balance]

    @property
    def nx(self) -> int:
        return len(self.variables)

    @property
    def nz(self) -> int:
        return len(self.noises)

    @property
    def n(self) -> int:
        return self.nx + self.nz


# ---------- 校验辅助 ----------
def _require_type(value: Any, path: str, expected: type | tuple[type, ...],
                  label: str) -> None:
    if not isinstance(value, expected):
        names = (
            "/".join(t.__name__ for t in expected)
            if isinstance(expected, tuple) else expected.__name__
        )
        raise ValidationError(f"{path} 必须是 {label}（{names}），实际类型为 "
                              f"{type(value).__name__}")


def _require_object(value: Any, path: str) -> dict[str, Any]:
    _require_type(value, path, dict, "JSON 对象")
    return value


def _require_array(value: Any, path: str) -> list[Any]:
    _require_type(value, path, list, "JSON 数组")
    return value


def _require_string(value: Any, path: str) -> str:
    _require_type(value, path, str, "字符串")
    return value


def _check_identifier(value: Any, path: str, seen: set[str]) -> str:
    ident = _require_string(value, path)
    if not ident or not _ID_RE.fullmatch(ident):
        raise ValidationError(
            f"{path} 必须是非空 ASCII 字母数字串，实际为 {ident!r}"
        )
    if ident in seen:
        raise ValidationError(f"{path} 标识 {ident!r} 在同一集合内重复")
    seen.add(ident)
    return ident


def _check_terms(raw: Any, path: str, known: set[str],
                 namespace_label: str) -> dict[str, Fraction]:
    obj = _require_object(raw, path)
    terms: dict[str, Fraction] = {}
    for key, raw_coef in obj.items():
        if not _ID_RE.fullmatch(key):
            raise ValidationError(
                f"{path}.{key} 的系数标识必须是非空 ASCII 字母数字串"
            )
        if key not in known:
            raise ValidationError(
                f"{path}.{key} 引用的{namespace_label}不存在"
            )
        terms[key] = parse_fraction(raw_coef, f"{path}.{key}")
    return terms


def validate(raw: Any) -> Model:
    """完整模式校验；任何不合法之处都抛出带路径的 :class:`ValidationError`。"""
    root = _require_object(raw, "$")
    extra = set(root) - set(_TOP_KEYS)
    if extra:
        raise ValidationError(f"$.{next(iter(extra))} 是未知字段")
    missing = [k for k in _TOP_KEYS if k not in root]
    if missing:
        raise ValidationError(f"$.{missing[0]} 为必填字段但缺失")

    # ---- 标识集合 ----
    var_seen: set[str] = set()
    noise_seen: set[str] = set()
    variables: list[str] = []
    noises: list[str] = []
    for i, item in enumerate(_require_array(raw["variables"], "$.variables")):
        variables.append(_check_identifier(item, f"$.variables[{i}]", var_seen))
    for i, item in enumerate(_require_array(raw["noises"], "$.noises")):
        noises.append(_check_identifier(item, f"$.noises[{i}]", noise_seen))
    overlap = var_seen & noise_seen
    if overlap:
        raise ValidationError(
            f"$.noises 标识 {next(iter(overlap))!r} 与 variables 中的真实量标识"
            "冲突（两类未知量必须分属不同空间）"
        )

    x_index = {name: i for i, name in enumerate(variables)}
    z_index = {name: nx for nx, name in enumerate(noises)}
    nx, nz = len(variables), len(noises)

    # ---- 测量 ----
    observations: list[Observation] = []
    obs_ids: set[str] = set()
    raw_obs = _require_array(raw["observations"], "$.observations")
    for i, item in enumerate(raw_obs):
        path = f"$.observations[{i}]"
        obj = _require_object(item, path)
        for key in obj:
            if key not in ("id", "value", "terms", "noise_terms"):
                raise ValidationError(f"{path}.{key} 是未知字段")
        for key in ("id", "value", "terms", "noise_terms"):
            if key not in obj:
                raise ValidationError(f"{path}.{key} 为必填字段但缺失")
        obs_id = _require_string(obj["id"], f"{path}.id")
        if not _ID_RE.fullmatch(obs_id):
            raise ValidationError(
                f"{path}.id 必须是非空 ASCII 字母数字串，实际为 {obs_id!r}"
            )
        if obs_id in obs_ids:
            raise ValidationError(f"{path}.id 标识 {obs_id!r} 重复")
        obs_ids.add(obs_id)

        value = parse_fraction(obj["value"], f"{path}.value")
        x_terms = _check_terms(obj["terms"], f"{path}.terms",
                               var_seen, "真实量变量")
        z_terms = _check_terms(obj["noise_terms"], f"{path}.noise_terms",
                               noise_seen, "误差")
        coeff: dict[int, Fraction] = {}
        for name, coef in x_terms.items():
            if coef != 0:
                coeff[x_index[name]] = coef
        for name, coef in z_terms.items():
            if coef != 0:
                coeff[nx + z_index[name]] = coef
        observations.append(Observation(
            id=obs_id, value=value, coeff=coeff,
            x_terms=x_terms, z_terms=z_terms,
        ))

    # ---- 守恒式（id 全局唯一，因此与测量 id 也不可冲突） ----
    balances: list[Balance] = []
    raw_bal = _require_array(raw["balances"], "$.balances")
    for i, item in enumerate(raw_bal):
        path = f"$.balances[{i}]"
        obj = _require_object(item, path)
        for key in obj:
            if key not in ("id", "terms", "value"):
                raise ValidationError(f"{path}.{key} 是未知字段")
        for key in ("id", "terms", "value"):
            if key not in obj:
                raise ValidationError(f"{path}.{key} 为必填字段但缺失")
        bal_id = _require_string(obj["id"], f"{path}.id")
        if not _ID_RE.fullmatch(bal_id):
            raise ValidationError(
                f"{path}.id 必须是非空 ASCII 字母数字串，实际为 {bal_id!r}"
            )
        if bal_id in obs_ids:
            raise ValidationError(
                f"{path}.id 标识 {bal_id!r} 与某测量 id 冲突（id 须全局唯一）"
            )
        if bal_id in {b.id for b in balances}:
            raise ValidationError(f"{path}.id 标识 {bal_id!r} 重复")
        value = parse_fraction(obj["value"], f"{path}.value")
        terms = _check_terms(obj["terms"], f"{path}.terms",
                             var_seen, "真实量变量")
        balances.append(Balance(id=bal_id, value=value,
                                terms={k: v for k, v in terms.items() if v != 0}))

    return Model(variables=variables, noises=noises,
                 observations=observations, balances=balances)
