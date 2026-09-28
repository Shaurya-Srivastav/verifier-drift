"""Run pilot P4: ``python pilots/p4_micro_grpo/run.py [--config ...] [--out ...]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p4_micro_grpo import run

if __name__ == "__main__":
    main_guard(run, __file__)
