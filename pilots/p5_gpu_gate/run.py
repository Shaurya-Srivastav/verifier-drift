"""Run pilot P5: ``python pilots/p5_gpu_gate/run.py [--set mode=full]``."""

from vfates.pilots.common import main_guard
from vfates.pilots.p5_gpu_gate import run

if __name__ == "__main__":
    main_guard(run, __file__)
