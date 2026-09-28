"""Run pilot P6: ``python pilots/p6_natural/run.py [--out results]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p6_natural import run

if __name__ == "__main__":
    main_guard(run, __file__)
