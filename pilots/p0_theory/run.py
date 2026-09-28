"""Run pilot P0: ``python pilots/p0_theory/run.py [--config ...] [--out ...]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p0_theory import run

if __name__ == "__main__":
    main_guard(run, __file__)
