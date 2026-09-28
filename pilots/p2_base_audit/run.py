"""Run pilot P2: ``python pilots/p2_base_audit/run.py [--config ...] [--out ...]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p2_base_audit import run

if __name__ == "__main__":
    main_guard(run, __file__)
