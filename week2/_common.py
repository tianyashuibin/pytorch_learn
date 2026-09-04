"""复用 week1 的模型与工具:把 week1 目录加入 import 路径。"""
import os
import sys

_WEEK1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "week1")
if _WEEK1 not in sys.path:
    sys.path.insert(0, _WEEK1)

from model import build_model  # noqa: E402,F401
from timing_utils import get_device, sync, timed  # noqa: E402,F401
