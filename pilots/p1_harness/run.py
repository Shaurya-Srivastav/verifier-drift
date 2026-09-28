"""Run pilot P1: ``python pilots/p1_harness/run.py [--config ...] [--out ...]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p1_harness import run

if __name__ == "__main__":
    main_guard(run, __file__)
