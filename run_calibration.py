#!/usr/bin/env python3
"""联合校正求解器的顶层入口（支持 `python3 -I` 隔离模式）。

示例：
    python3 -I run_calibration.py examples/canonical.json
    python3 -I run_calibration.py input.json output.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_calibration.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
