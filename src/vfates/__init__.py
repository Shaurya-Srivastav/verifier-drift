"""vfates: which imperfect-verifier errors transfer in RL with verifiable rewards.

The package is organised as a pipeline:

    tasks      -> problems, prompts, answer parsing, oracle grading
    verifiers  -> fixed and resampled verifier arms, calibrated to step-0 rates
    gen        -> sampling backends (MLX, Hugging Face, vLLM)
    features   -> per-sample representations (hidden states, LoRA-gradient sketches)
    forecast   -> step-0 confusion statistics, transfer precision, locked forecasts
    train      -> GRPO (MLX on Apple silicon, TRL on CUDA)
    analysis   -> curves, calibration of forecasts against outcomes, figures
    theory     -> exact small-world dynamics used to validate the theory
    pilots     -> the staged validation ladder (P0-P6)
"""

__version__ = "0.1.0"
