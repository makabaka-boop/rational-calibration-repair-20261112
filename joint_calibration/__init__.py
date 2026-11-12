"""联合校正（共享标定误差）纯后端求解器。

仅使用 Python 标准库（fractions.Fraction、json 仅用于输出），
核心高斯-若尔当消元、两层最小二乘与不可行证书均自行实现。
"""

from .model import ValidationError, parse_fraction, validate
from .solver import solve_model
from .strict_json import StrictJSONError, loads

__all__ = [
    "ValidationError",
    "StrictJSONError",
    "loads",
    "parse_fraction",
    "validate",
    "solve_model",
]
