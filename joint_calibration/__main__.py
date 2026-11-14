"""命令行入口。

用法（隔离模式，推荐）：
    python3 -I run_calibration.py 输入.json [输出.json]
也支持包方式：
    python3 -m joint_calibration 输入.json [输出.json]

* 输入按 UTF-8 解码并做严格校验（重复字段/未知字段/非法数值/非法引用均拒绝）；
* 仅当完整求解成功后，才以“同目录临时文件 + os.replace”原子替换输出；
* 任何失败都保留旧输出，错误信息写入 stderr 并以非零状态退出。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# 同时兼容 `python -m joint_calibration`（有包上下文）与
# `python -I joint_calibration/__main__.py`（无包上下文）两种启动方式。
try:  # pragma: no cover - 两条导入路径只取决于启动方式
    from .model import ValidationError, validate
    from .solver import solve_model
    from .strict_json import StrictJSONError, loads
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joint_calibration.model import ValidationError, validate
    from joint_calibration.solver import solve_model
    from joint_calibration.strict_json import StrictJSONError, loads


def run(input_text: str) -> dict:
    """解析 + 校验 + 求解，返回可 JSON 序列化的结果字典。"""
    raw = loads(input_text)
    model = validate(raw)
    return solve_model(model)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 1 or len(args) > 2:
        print("用法: python3 -I run_calibration.py 输入.json [输出.json]",
              file=sys.stderr)
        return 2
    input_path = args[0]
    output_path = args[1] if len(args) == 2 else None

    try:
        with open(input_path, "r", encoding="utf-8", errors="strict") as fh:
            text = fh.read()
        result = run(text)
    except (StrictJSONError, ValidationError) as exc:
        print(f"输入被拒绝: {exc}", file=sys.stderr)
        return 1
    except ArithmeticError as exc:
        # 精确求解内部出现的不相容（正常输入不会触发）：明确报告求解失败，
        # 而不是当作“无解”或静默写出错误结果。
        print(f"求解失败（内部精确运算异常）: {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print(f"输入被拒绝: 不是合法 UTF-8 文本（{exc.reason}，"
              f"字节位置 {exc.start}）", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"文件错误: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False)
    if output_path is None:
        print(payload)
        return 0

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".joint_calibration.", suffix=".tmp", dir=out_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, output_path)  # 同文件系统内原子替换
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        print(f"写出失败（旧结果已保留）: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
