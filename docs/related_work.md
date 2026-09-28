# Related work

Each cluster is what that work established, then the point of contact with this repository. Numerical claims below come either from the cited paper or from this repository (pilot P0, `pilots/p0_theory/results/report.md`). Nothing else is filled in. Where a connection to the theory here is an interpretation rather than a result of the cited paper, the text says so.

## Random verifier noise and Youden's $J$

Rad et al., *Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards* (arXiv:2601.04411), model GRPO under false positives and false negatives as a replicator flow over reasoning modes. The drift of the incorrect mass depends only on Youden's $J = \mathrm{TPR} - \mathrm{FPR} = 1 - \mathrm{FPR} - \mathrm{FNR}$, which gives a phase transition at $J = 0$; a KL penalty turns it into an interior equilibrium. Their noise is resampled and their prediction is summarised by the two population rates. Lemma 1 and the resampled-noise identity $\mathbb{E}[g_V] = J g_O$ are the part of that picture this repository keeps: when errors are redrawn, the shared update stays in the oracle direction and the scale is $J$ (P0.3).

The departure is fixed errors. P0.5 holds $J$ fixed across three arms: a shared false positive, a prompt-specific twin, and matched resampled noise. Held-out accuracy separates the shared arm from the other two, so matched $J$ does not determine the fate once the errors are fixed. The screen for that split is transfer precision (Theorem 3, `vfates.forecast.transfer`). E1 tests the same claim on a language model at matched step-0 channel rates.

## Systematic verifier errors on chain sums

Egashira et al., *Delay, Plateau, or Collapse* (arXiv:2605.02909), inject systematic verifier errors on Reasoning Gym `decimal_chain_sum` (OLMo3-7B, Qwen3-1.7B-Base, and Qwen2.5-1.5B-Instruct in an appendix; single-pass data; DAPO in TRL). They report that systematic false negatives behave like random noise (training is delayed and then recovers), while systematic false positives plateau or collapse depending on the pattern. For word-triggered false positives, the outcome follows two step-0 quantities: the trigger's initial frequency and its conditional advantage $C(t)$, the mean oracle advantage of outputs containing the trigger. Their random-noise arms flip at fixed rates, and their systematic rules are not calibrated to those rates, so rate and structure vary together.

This repository uses that task and its configuration (`vfates.tasks.chain_sum`) and adds three things:

- **Matched controls.** Every fixed rule has a resampled control at the same step-0 channel rate, so a difference in fate is not a difference in rate.
- **One forecast across rule types.** `tau_pool` is defined for any verifier from the step-0 update. The prompt-specific arms (`fp_hash`, `fn_prompt_subset`) are the case that separates the two views. Their errors are frequent and wrong by construction, which the frequency and conditional-advantage reading would flag, but Theorem 3 predicts that held-out prompts do not move.
- **The false-negative question, sharpened.** `fn_behavior` rejects correct answers carrying a phrase whose within-prompt association with correctness is positive. That is a tax on a behaviour that predicts success, which neither of their false-negative rules (a LaTeX display token and English text) is selected to be.

The trigger arm's phrase sign uses the within-prompt association, which matches their advantage-based definition. Their conditional advantage, with the initial frequency, is reported for every arm with a behaviour rule, as a baseline forecast next to `tau_pool` (`PREREGISTRATION.md`, H3 baselines).

## Controlled FPR and FNR studies

Plesner et al., *An Imperfect Verifier is Good Enough* (arXiv:2604.07666), find that symmetric resampled noise up to 15% leaves peak validation accuracy within 2 points of the clean baseline, across Qwen3, GLM4 and Llama 3.1 at 4B to 9B. Their noise is symmetric and redrawn every epoch. They close with: "A systematic controlled study varying FPR and FNR independently, including fixed noise, would strengthen our conclusions about the importance of precision over recall."

The arm matrix in `configs/arms/chain_sum.yaml` addresses part of that request. It does not vary FPR and FNR over a grid. Instead:

- False-positive and false-negative rules are separate, each at one calibrated rate (FPR 0.15, FNR 0.30), and one arm combines them.
- Each fixed rule has resampled noise at the same rate.
- Prompt-specific rules sit next to shared ones at the same rate.

The question it answers is whether fixed structure changes the fate at a fixed rate, not how the fate varies with the rate.

## Natural false positives on code

Zhang, *When the Reward Suite Is Leaky* (arXiv:2607.11022), runs a preregistered two-arm GRPO contrast on MBPP with Qwen2.5-Coder-1.5B-Instruct (5 seeds × 400 steps), replicated on two more model families. The leaky arm is rewarded by the original MBPP tests. The hardened arm is rewarded by the MBPP+ extra tests alone. Three findings are relevant here:

- The held-out gap is 0.20 points, non-inferior under a 1.5-point margin.
- The false-positive mass rewarded during training tracks a static step-0 leakiness audit (Spearman 0.80).
- The mechanism evidence points to selection of pre-existing wrong programs rather than learned exploitation.

The comparison has no rate-matched random arm. One reading, which is an interpretation and not a claim of that paper, is that these natural false positives are the prompt-specific case of Theorem 3: they are per-task programs that do not load on a feature shared with held-out tasks, so they do not move $g_V$ off the oracle direction. E2 tests that reading by adding the missing arm: resampled noise at the natural verifier's step-0 channel rates, with the direction forecast by locked `tau_pool`. `vfates.tasks.mbpp` grades MBPP+ and can score the original tests as the natural verifier. P0's `fpI` arm is the toy version of the same split, matched in $J$ to a shared false positive and to resampled noise.

## Correlated verifier errors within a group

Xin, *Are Verifier Errors Independent Within a GRPO Group?* (arXiv:2609.06386), measures verifier errors on 24,998 groups of eight Qwen2.5-1.5B completions (MATH, GSM8K, DeepMath-103K). The pooled within-group error correlation is 0.530 (95% interval 0.500 to 0.560), which is an effective group size of about 1.70 under an exchangeable model. It varies by answer form, and advantage signs disagree across four rule-based verifiers in up to 0.83% of groups. The paper does not train under the errors and does not separate shared answer form from shared prompt difficulty.

Correlation inside a group is the prompt-level half of the structure this repository cares about. A prompt-specific rule (`fp_hash`, `fn_prompt_subset`) is strongly correlated within a group and still predicted to stay local. A shared rule is correlated across prompts, which is what moves held-out accuracy. The prompt-centred statistics here (`tau_pool`, the within-prompt phrase association) remove the shared prompt-difficulty component that paper leaves unresolved.

## Reward hacking by special-casing tests

Countdown-Code (arXiv:2603.07084) shows special-casing and hardcoding of tests as a shared exploit in code RL. In the decomposition here, that behaviour is an $\mathrm{FP}$ aligned with a feature that is present on held-out prompts, so it contributes a nonzero $g_{\mathrm{FP}}$. The chain-sum analogues are the tolerance and trigger false positives; the code task's natural false positives are the MBPP versus MBPP+ gap above. Model-written programs are executed only inside `vfates.tasks.sandbox`, which is not a security boundary (`SECURITY.md`).

## Judges as verifiers

Zhao et al., *One Token to Fool LLM-as-a-Judge* (arXiv:2507.08794), show master-key false positives against an LLM judge. Huang et al. (arXiv:2505.22203) study model-based verifier hacking during RL. A master key is a shared surface feature, the same role as `TriggerFP` / `fp_trigger`. `vfates.tasks.judges` provides the content judges used when the IFEval oracle is the constraint checker conjoined with a content check (`vfates.tasks.ifeval`).

## Keyword stuffing under IFEval RLVR

Tulu 3 (Lambert et al., arXiv:2411.15124) reports keyword stuffing when IFEval is used as an RLVR reward. Stuffing is a shared surface behaviour that the verifier accepts and the oracle need not. `TriggerFP` and `BehaviorFN` are the controlled versions of a phrase rule. In the chain-sum arm list, `fp_trigger` rewards a phrase that is not enriched for correct answers within a prompt, and `fn_behavior` rejects correct answers carrying a phrase that is, so that the rejection is a tax on a behaviour that predicts correctness. The natural IFEval verifier in this repository is the constraint checker; the oracle adds the content judge.

## Tilts and GRPO theory

Mroueh (arXiv:2503.06639) and Dorner et al. (arXiv:2507.12399, Proposition 4) give the exponential-tilt geometry that Lemma 1 quotes: updates $\pi_t \propto \pi_0 \exp(g_t(x, V))$ preserve $\pi(\cdot \mid V = v, x)$. This repository uses that corollary as the repeated-prompt baseline (P0.1, P0.4b) and then leaves it. Proposition 2 is the Euclidean softmax flow, whose $\beta = 0$ limit is modal precision and whose $\beta > 0$ equilibrium is the finite tilt $\pi_0 \exp(V/\beta)$ (P0.4a, P0.4c). Theorem 3 is the part that is not in those papers: fixed errors add $g_{\mathrm{FP}}$ and $g_{\mathrm{FN}}$, and held-out prompts move only through the shared inner product with $g_V$.

## Imperfect verifiers at inference

Stroebl et al. (arXiv:2411.17501) study best-of-$n$ selection with imperfect verifiers at inference. The object here is the training-time counterpart. The verifier is held fixed and used as the GRPO reward, and the question is which of its errors the shared update carries to held-out prompts. Mean precision and modal precision are the two terminal accuracies of that training dynamics (Lemma 1, Proposition 2), not inference-time selection rules. The audit estimators are `mean_precision_forecast` and `modal_precision_forecast` in `vfates.forecast.confusion`.

## Gradient features

LESS (arXiv:2402.04333) uses random projections of LoRA gradients for data selection. TRAK (Park et al., arXiv:2303.14186) uses random projections of per-sample gradients for influence estimates. This repository uses the same kind of object for a different purpose. At standard LoRA initialisation ($B = 0$) the trainable step-0 feature is $\partial L / \partial B$, already a random projection of the full weight gradient through the fixed $A$. CountSketch then sketches it so that inner products stay unbiased with variance at most $(\lVert a \rVert^2 \lVert b \rVert^2 + \langle a, b \rangle^2) / D$ (`vfates.features.lora_grad`, `vfates.features.sketch`). The audit saves that $A$ and every training run loads it, so the forecast and the trained adapter share one parameterisation. The sketched vector is $\phi$ in the velocity $s(y) = \langle \phi(x', y), g_V \rangle$. It is not used to choose training data.

## Sparse RL updates

*RL fine-tunes small subnetworks* (arXiv:2505.11711) and *RL's Razor* (arXiv:2509.04259) study RL updates that concentrate on a small part of the network. That is the regime in which a single step-0 direction $g_V$ can be a summary of what training will reinforce. The knife-edge world (P0.8) is the limitation that comes with the summary: $\tau$ is first-order at $\pi_0$, and a shared flow can leave that neighbourhood. On the committed seeds, transfer precision is about $0.73$ while terminal held-out accuracy is $0.023$. E1 labels any run whose curve rises and then falls by the preregistered amounts.

## GRPO and its variants

Shao et al., DeepSeekMath (arXiv:2402.03300), introduce GRPO. `vfates.train.grpo_core` is the numpy reference both trainers are tested against. There are two advantage modes and three loss reductions:

- **Advantages:** `std` (TRL `scale_rewards="group"`) and `mean`, the unscaled advantage of Dr.GRPO (Liu et al., arXiv:2503.20783; TRL `scale_rewards="none"`).
- **Reductions:**
  - `seq_mean`, the per-sequence average of GRPO;
  - `token_mean`, the token-level average used in DAPO's loss (Yu et al., arXiv:2503.14476), which is only that reduction and not the rest of DAPO;
  - `dr_constant`, the constant normaliser of Dr.GRPO.

The advantage identity in `docs/theory.md` is for the mean-centred advantage; standard-deviation scaling rescales it and does not flip its sign (P0.3c). With `seq_mean`, the step-0 direction weights each completion by the inverse of its length, and the audit applies that weight to train rows when it builds $g_V$. `MLXGRPOTrainer` and `build_trl_trainer` are the Apple-silicon and CUDA implementations of that objective.

## Tasks

Reasoning Gym (Stojanovski et al., arXiv:2505.24760) supplies `decimal_chain_sum`, with a bit-identical built-in generator when the package is absent. EvalPlus (Liu et al., arXiv:2305.01210) supplies MBPP+. IFEval (Zhou et al., arXiv:2311.07911) is the instruction-following task, loaded here as `allenai/RLVR-IFeval`. Those three are `vfates.tasks.chain_sum`, `vfates.tasks.mbpp`, and `vfates.tasks.ifeval`.
