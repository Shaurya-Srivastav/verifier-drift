"""verifier-drift: does a fixed verifier's error drift as an RLVR policy's
answer style changes, and does that bias GRPO training?

The analysis core (forms, verifier, oracle, audit, metrics) is pure-Python and
runs on CPU. Real GRPO training on an HF model lives in `grpo.py` and is meant
to run on a GPU / Apple-Silicon box; it reuses this same core so the drift
metrics are computed identically in the smoke pilot and the real run.
"""

__version__ = "0.0.1"
