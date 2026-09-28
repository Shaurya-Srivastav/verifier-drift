"""Run pilot P3: ``python pilots/p3_estimator/run.py [--config ...] [--out ...]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p3_estimator import run

if __name__ == "__main__":
    main_guard(run, __file__)
