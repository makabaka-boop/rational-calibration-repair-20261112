"""严格的 UTF-8 JSON 递归下降解析器。

与标准库 json 的关键区别：
* 对象内重复字段会被拒绝，并给出 JSONPath 风格的定位路径；
* 语法错误给出输入字符位置；
* 不使用任何第三方/代数库。

路径约定（验证阶段也沿用）：
    $                       根
    $.field                 根对象字段
    $.observations[3]       数组下标
    $.observations[3].terms  嵌套字段
"""

from __future__ import annotations

from typing import Any


class StrictJSONError(ValueError):
    """JSON 语法错误或重复字段错误，message 中包含定位路径。"""


_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_WS = " \t\n\r"
_HEX = "0123456789abcdefABCDEF"


class _Parser:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0
        self.n = len(text)

    # ---------- 基础工具 ----------
    def _peek(self) -> str | None:
        return self.s[self.i] if self.i < self.n else None

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in _WS:
            self.i += 1

    def _error(self, msg: str) -> StrictJSONError:
        return StrictJSONError(f"{msg} (位置 {self.i})")

    def _expect_literal(self, literal: str, value: Any) -> Any:
        if not self.s.startswith(literal, self.i):
            raise self._error(f"非法 JSON：期望字面量 {literal}")
        self.i += len(literal)
        return value

    # ---------- 入口 ----------
    def parse_root(self) -> Any:
        self._skip_ws()
        value = self.parse_value("$")
        self._skip_ws()
        if self.i != self.n:
            raise self._error("非法 JSON：根值之后存在多余字符")
        return value

    def parse_value(self, path: str) -> Any:
        ch = self._peek()
        if ch is None:
            raise self._error(f"非法 JSON：{path} 处意外结束")
        if ch == '"':
            return self.parse_string()
        if ch == "{":
            return self.parse_object(path)
        if ch == "[":
            return self.parse_array(path)
        if ch == "t":
            return self._expect_literal("true", True)
        if ch == "f":
            return self._expect_literal("false", False)
        if ch == "n":
            return self._expect_literal("null", None)
        if ch == "-" or "0" <= ch <= "9":
            return self.parse_number(path)
        raise self._error(f"非法 JSON：{path} 处出现非法字符 {ch!r}")

    # ---------- 对象 / 数组 ----------
    def parse_object(self, path: str) -> dict[str, Any]:
        self.i += 1  # '{'
        result: dict[str, Any] = {}
        self._skip_ws()
        if self._peek() == "}":
            self.i += 1
            return result
        while True:
            self._skip_ws()
            if self._peek() != '"':
                raise self._error(f"非法 JSON：{path} 的对象字段名必须是字符串")
            key = self.parse_string()
            field_path = f"{path}.{key}"
            if key in result:
                raise StrictJSONError(f"重复 JSON 字段：{field_path}")
            self._skip_ws()
            if self._peek() != ":":
                raise self._error(f"非法 JSON：{field_path} 后缺少 ':'")
            self.i += 1
            self._skip_ws()
            result[key] = self.parse_value(field_path)
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.i += 1
                continue
            if ch == "}":
                self.i += 1
                return result
            raise self._error(f"非法 JSON：{path} 的对象中缺少 ',' 或 '}}'")

    def parse_array(self, path: str) -> list[Any]:
        self.i += 1  # '['
        result: list[Any] = []
        self._skip_ws()
        if self._peek() == "]":
            self.i += 1
            return result
        while True:
            self._skip_ws()
            result.append(self.parse_value(f"{path}[{len(result)}]"))
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.i += 1
                continue
            if ch == "]":
                self.i += 1
                return result
            raise self._error(f"非法 JSON：{path} 的数组中缺少 ',' 或 ']'")

    # ---------- 字符串 ----------
    def parse_string(self) -> str:
        start = self.i
        self.i += 1  # 开引号
        out: list[str] = []
        s = self.s
        while True:
            if self.i >= self.n:
                raise self._error("非法 JSON：字符串未闭合")
            ch = s[self.i]
            if ch == '"':
                self.i += 1
                return "".join(out)
            if ord(ch) < 0x20:
                raise self._error("非法 JSON：字符串中出现未转义控制字符")
            if ch != "\\":
                out.append(ch)
                self.i += 1
                continue
            self.i += 1
            if self.i >= self.n:
                raise self._error("非法 JSON：转义序列未结束")
            esc = s[self.i]
            self.i += 1
            if esc == "u":
                out.append(self._parse_unicode_escape(start))
            else:
                if esc not in _ESCAPES:
                    raise self._error(f"非法 JSON：未知转义 \\{esc}")
                out.append(_ESCAPES[esc])

    def _parse_unicode_escape(self, _start: int) -> str:
        cp = self._read_hex4()
        # 代理对
        if 0xD800 <= cp <= 0xDBFF:
            if self.s.startswith("\\u", self.i):
                self.i += 2
                low = self._read_hex4()
                if 0xDC00 <= low <= 0xDFFF:
                    return chr(0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00))
                raise self._error("非法 JSON：高代理项后缺少合法低代理项")
            raise self._error("非法 JSON：高代理项缺少配对的低代理项")
        if 0xDC00 <= cp <= 0xDFFF:
            raise self._error("非法 JSON：出现孤立的低代理项")
        return chr(cp)

    def _read_hex4(self) -> int:
        s = self.s
        if self.i + 4 > self.n or any(c not in _HEX for c in s[self.i:self.i + 4]):
            raise self._error("非法 JSON：\\u 后需要 4 位十六进制数")
        value = int(s[self.i:self.i + 4], 16)
        self.i += 4
        return value

    # ---------- 数字（语法层面接受；数值合法性在验证阶段处理） ----------
    def parse_number(self, path: str) -> int | float:
        start = self.i
        s = self.s
        if self._peek() == "-":
            self.i += 1
        if self.i >= self.n:
            raise self._error(f"非法 JSON：{path} 处数字不完整")
        if s[self.i] == "0":
            self.i += 1
        elif "1" <= s[self.i] <= "9":
            while self.i < self.n and "0" <= s[self.i] <= "9":
                self.i += 1
        else:
            raise self._error(f"非法 JSON：{path} 处数字不合法")
        is_float = False
        if self.i < self.n and s[self.i] == ".":
            is_float = True
            self.i += 1
            frac_start = self.i
            while self.i < self.n and "0" <= s[self.i] <= "9":
                self.i += 1
            if self.i == frac_start:
                raise self._error(f"非法 JSON：{path} 处小数部分为空")
        if self.i < self.n and s[self.i] in "eE":
            is_float = True
            self.i += 1
            if self.i < self.n and s[self.i] in "+-":
                self.i += 1
            exp_start = self.i
            while self.i < self.n and "0" <= s[self.i] <= "9":
                self.i += 1
            if self.i == exp_start:
                raise self._error(f"非法 JSON：{path} 处指数部分为空")
        token = s[start:self.i]
        return float(token) if is_float else int(token)


def loads(text: str) -> Any:
    """解析 JSON 文本，重复字段与语法错误抛出 :class:`StrictJSONError`。"""
    if not isinstance(text, str):
        raise StrictJSONError("loads 仅接受 str 输入")
    return _Parser(text).parse_root()
