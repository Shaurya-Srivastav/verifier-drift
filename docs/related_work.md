# Related work

Each cluster is what that work established, then the point of contact with this repository. Numerical claims below are either taken from the cited note or measured in this repository (pilot P0, `pilots/p0_theory/results/report.md`). Nothing else is filled in.

## Random verifier noise and Youden's $J$

Rad et al., *Rate or Fate* (arXiv:2601.04411, ICML 2026 spotlight), study random verifier noise and a phase transition in Youden's $J = 1 - \mathrm{FPR} - \mathrm{FNR}$. Lemma 1 and the resampled-noise identity $\mathbb{E}[g_V] = J g_O$ are the part of that picture this repository keeps: when errors are redrawn, the shared update stays in the oracle direction and the scale is $J$ (P0.3). P0.5 holds $J$ fixed across a shared false positive, a prompt-specific twin, and matched resampled noise. Held-out accuracy separates the shared arm from the other two. Matched $J$ does not determine the fate once the errors are fixed. The screen for that split is transfer precision (Theorem 3, `vfates.forecast.transfer`).

## Systematic verifier errors on chain sums

Egashira et al., *Delay, Plateau, or Collapse* (arXiv:2605.02909, COLM 2026), inject systematic verifier errors on Reasoning Gym `decimal_chain_sum`. Their conclusion is that the pattern of the error matters, and the rates in that study are not matched. A trigram rule is the concrete case they use: frequent triggers with negative advantage collapse, and positive ones plateau. This repository uses that task (`vfates.tasks.chain_sum`, Egashira configuration by default) and a trigger-phrase false positive (`fp_trigger` in `configs/arms/chain_sum.yaml`). The arms are calibrated to shared step-0 pooled FPR and FNR targets, and the resampled arms redraw noise at those achieved rates. The predictor is the step-0 velocity of the shared update, with the tilt and mode limits in `docs/theory.md`.

## The controlled FPR/FNR study

Plesner et al., *An Imperfect Verifier is Good Enough* (arXiv:2604.07666), end with a request for the missing experiment: "A systematic controlled study varying FPR and FNR independently, including fixed noise, would strengthen our conclusions." The arm matrix in `configs/arms/chain_sum.yaml` is that study: false-positive and false-negative rules are separate, each is calibrated to its own target rate, fixed rules and resampled noise sit at the same rates, and prompt-specific rules sit next to shared ones.

## Natural false positives on code

Zhang (arXiv:2607.11022) compares MBPP with MBPP+. The natural code false positives are described there as selection, not exploitation. A step-0 audit predicts false-positive mass with Spearman correlation 0.80, the held-out gap is about 0.2 points, and that comparison has no random arm. That pattern is the prompt-specific case of Theorem 3: an error that does not load on a feature shared with held-out prompts does not move $g_V$ off the oracle direction. `vfates.tasks.mbpp` grades MBPP+ (base tests and extra tests) and can score the original MBPP tests as the natural verifier. P0's `fpI` arm is the toy version of the same split, matched in $J$ to a shared false positive and to resampled noise.

## Concurrent work on imperfect verifiers

Xin (arXiv:2609.06386) is a closely related concurrent study of imperfect verifiers in RLVR.

## Reward hacking by special-casing tests

Countdown-Code (arXiv:2603.07084) shows special-casing and hardcoding of tests as a shared exploit in code RL. In the decomposition here, that behaviour is an $\mathrm{FP}$ aligned with a feature that is present on held-out prompts, so it contributes a nonzero $g_{\mathrm{FP}}$. The chain-sum analogues are the tolerance and trigger false positives; the code task's natural false positives are the MBPP versus MBPP+ gap above. Model-written programs are executed only inside `vfates.tasks.sandbox`, which is not a security boundary (`SECURITY.md`).

## Judges as verifiers

Zhao et al., *One Token to Fool LLM-as-a-Judge* (arXiv:2507.08794), show master-key false positives against an LLM judge. Huang et al. (arXiv:2505.22203) study model-based verifier hacking during RL. A master key is a shared surface feature, the same role as `TriggerFP` / `fp_trigger`. `vfates.tasks.judges` provides the content judges used when the IFEval oracle is the constraint checker conjoined with a content check (`vfates.tasks.ifeval`).

## Keyword stuffing under IFEval RLVR

Tulu 3 (Lambert et al., arXiv:2411.15124) reports keyword stuffing when IFEval is used as an RLVR reward. Stuffing is a shared surface behaviour that the verifier accepts and the oracle need not. `BehaviorFN` and `TriggerFP` are the controlled versions of a phrase rule, and `fn_behavior` in the chain-sum arm list prefers phrases with positive accuracy lift so that a rejection is a tax on a behaviour that predicts correctness. The natural IFEval verifier in this repository is the constraint checker; the oracle adds the content judge.

## Tilts and GRPO theory

Mroueh (arXiv:2503.06639) and Dorner et al. (arXiv:2507.12399, Proposition 4) give the exponential-tilt geometry that Lemma 1 quotes: updates $\pi_t \propto \pi_0 \exp(g_t(x, V))$ preserve $\pi(\cdot \mid V = v, x)$. This repository uses that corollary as the repeated-prompt baseline (P0.1, P0.4b) and then leaves it. Proposition 2 is the Euclidean softmax flow, whose $\beta = 0$ limit is modal precision and whose $\beta > 0$ equilibrium is the finite tilt $\pi_0 \exp(V/\beta)$ (P0.4a, P0.4c). Theorem 3 is the part that is not in those papers: fixed errors add $g_{\mathrm{FP}}$ and $g_{\mathrm{FN}}$, and held-out prompts move only through the shared inner product with $g_V$.

## Imperfect verifiers at inference

Stroebl et al. (arXiv:2411.17501) study best-of-$n$ selection with imperfect verifiers at inference. The object here is the training-time counterpart. The verifier is held fixed and used as the GRPO reward, and the question is which of its errors the shared update carries to held-out prompts. Mean precision and modal precision are the two terminal accuracies of that training dynamics (Lemma 1, Proposition 2), not inference-time selection rules. The audit estimators are `mean_precision_forecast` and `modal_precision_forecast` in `vfates.forecast.confusion`.

## Gradient features

LESS (arXiv:2402.04333) uses random projections of LoRA gradients for data selection. TRAK (Park et al., arXiv:2303.14186) uses random projections of per-sample gradients for influence estimates. This repository uses the same kind of object for a different purpose. At standard LoRA initialisation ($B = 0$) the trainable step-0 feature is $\partial L / \partial B$, already a random projection of the full weight gradient, and CountSketch sketches it so that inner products stay unbiased with variance at most $(\lVert a \rVert^2 \lVert b \rVert^2 + \langle a, b \rangle^2) / D$ (`vfates.features.lora_grad`, `vfates.features.sketch`). The sketched vector is $\phi$ in the velocity $s(y) = \langle \phi(x', y), g_V \rangle$. It is not used to choose training data.

## Sparse RL updates

*RL fine-tunes small subnetworks* (arXiv:2505.11711) and *RL's Razor* (arXiv:2509.04259) study RL updates that concentrate on a small part of the network. That is the regime in which a single step-0 direction $g_V$ can be a summary of what training will reinforce. The knife-edge world (P0.8) is the limitation that comes with the summary: $\tau$ is first-order at $\pi_0$, and a shared flow can leave that neighbourhood. On the committed seeds, transfer precision is about $0.73$ and terminal held-out accuracy is $0.0226$.

## GRPO and its variants

Shao et al., DeepSeekMath (arXiv:2402.03300), introduce GRPO. `vfates.train.grpo_core` is the numpy reference both trainers are tested against. Advantages are `std` (TRL `scale_rewards="group"`) and `mean`, which the module names as Dr.GRPO (Liu et al., arXiv:2503.20783; TRL `scale_rewards="none"`). Reductions are `seq_mean` (GRPO), `token_mean`, which the module names as DAPO (Yu et al., arXiv:2503.14476), and `dr_constant` (Dr.GRPO). The advantage identity in `docs/theory.md` is for the mean-centred advantage; standard-deviation scaling rescales it and does not flip its sign (P0.3c). `MLXGRPOTrainer` and `build_trl_trainer` are the Apple-silicon and CUDA implementations of that objective.

## Tasks

Reasoning Gym (Stojanovski et al., arXiv:2505.24760) supplies `decimal_chain_sum`, with a bit-identical built-in generator when the package is absent. EvalPlus (Liu et al., arXiv:2305.01210) supplies MBPP+. IFEval (Zhou et al., arXiv:2311.07911) is the instruction-following task, loaded here as `allenai/RLVR-IFeval`. Those three are `vfates.tasks.chain_sum`, `vfates.tasks.mbpp`, and `vfates.tasks.ifeval`.
