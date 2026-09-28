# Related work

Each cluster is what that work established, then the point of contact with this repository. Numerical claims below come either from the cited paper (arXiv records checked on 28 September 2026) or from this repository (pilot P0, `pilots/p0_theory/results/report.md`). Where a connection to the theory here is an interpretation rather than a result of the cited paper, the text says so.

## What this study adds, and what it does not claim

The contribution is a test, at matched step-0 rates, of whether a statistic of the step-0 update ranks training outcomes:

- every synthetic verifier in a domain is calibrated to the same step-0 channel (FPR 0.15, FNR 0.30), with prompt-specific, shared-behaviour and resampled arms side by side;
- `tau_pool`, locked before training, is tested as a rank predictor of the within-pair accuracy difference across five domains at 1.7B and 4B, then again, prospectively, on 8B to 32B models and two other families;
- two shared phrase rules with the same rate and structure, whose phrase predicts correct or wrong answers, test whether the screen reads the direction of a shared error and not only its structure, which is what a frequency-and-advantage reading or a structure label would need to explain;
- the screen's label cost is traced from 32 to 192 oracle labels per verifier;
- at an equal patch budget, fixes chosen by the screen are compared with fixes chosen by the error count.

It does not claim to be the first to show that the pattern of verifier errors, not their rate, decides the outcome (Egashira et al.), the first pre-training predictor of an RLVR outcome (conditional advantage, leakiness audits and hack probes exist), the first gradient-based reward-hacking method (GRIFT), the first master-key judge failure (Zhao et al.), the first natural false-positive contrast on code (Zhang), or the first evidence that reward hacking generalises across tasks (MacDiarmid et al., Taylor et al.). It makes no prediction that a resampled arm matches the clean verifier.

## Systematic verifier errors: pattern, not rate

Egashira et al., *Delay, Plateau, or Collapse* (arXiv:2605.02909, COLM 2026), inject systematic verifier errors on Reasoning Gym arithmetic (Qwen3-1.7B-Base, OLMo3-7B, Qwen2.5-1.5B-Instruct and OLMo3-32B; GRPO, with DAPO, Dr. GRPO and SAPO preserving the qualitative split). Their conclusion: "these outcomes are not determined by the overall error rate but by the specific pattern of introduced errors", and they are "difficult to predict from aggregate measures of verifier quality, such as overall error rate or Youden's J statistic". For word-triggered false positives, the outcome follows the trigger's initial frequency and its conditional advantage $C(t)$, the mean oracle advantage of outputs containing the trigger. Their arms are not calibrated to one rate, so rate and structure vary together. They close by asking for code, formal and rubric-based domains, and for "diagnostics and mitigation strategies that remain effective without full oracle access".

This repository takes that task and configuration (`vfates.tasks.chain_sum`) as one of five domains and adds:

- **Matched controls.** Every fixed rule has a resampled control at the same step-0 channel rate, so a difference in fate is not a difference in rate. Any correction that depends only on the rates treats a pair identically.
- **One forecast across rule types.** `tau_pool` is defined for any verifier. The prompt-specific arms (`fp_hash`, `fn_prompt_subset`) separate it from a frequency-and-advantage reading: their errors are frequent and wrong by construction, yet Theorem 3 predicts that held-out prompts do not move.
- **Few oracle labels.** `g_V` uses only the verifier's verdicts, so the oracle is needed on the 192 held-out samples the update favours most (H9 traces 32 to 192).
- **The false-negative question, sharpened.** `fn_behavior` taxes a phrase whose within-prompt association with correctness is positive, which neither of their false-negative rules is selected to be.

Their conditional advantage, with the initial frequency, is a preregistered baseline for every arm with a behaviour rule (`PREREGISTRATION.md`, H3 baselines). X6 reruns their base model and task with the kit at matched rates.

## Random verifier noise and Youden's $J$

Rad et al., *Rate or Fate? RLVεR* (arXiv:2601.04411), model GRPO under class-conditional false positives and false negatives as a replicator flow: "when J>0, the incorrect mass is driven toward extinction (learning); when J=0, the process is neutral; and when J<0, incorrect modes amplify", so in the learning regime "noise primarily rescales convergence time". Lemma 1 and the identity $\mathbb{E}[g_V] = J g_O$ are the part of that picture kept here (P0.3). The study's channel has $J = 0.55$, so their result predicts that the resampled arms still learn.

Other evidence on random noise is mixed, which is why the design compares arms with each other and not with the oracle. Plesner et al. (arXiv:2604.07666) find that symmetric noise up to 15% leaves peak validation accuracy within 2 points of clean training (Qwen3, GLM-4, Llama 3.1; 4B to 9B), and ask for "a systematic controlled study varying FPR and FNR independently, including fixed noise". Mitsuhashi et al. (arXiv:2605.25252) find that the accuracy gap under controlled noise "persists under substantial compute scaling", on Qwen2.5-0.5B and 1.5B. Zhu and Kang (arXiv:2603.16140) report that truly incorrect annotations cost 8 to 10 points after removing contamination, against earlier results that 40% flips still learn (Lv et al., arXiv:2505.22653). Shao et al., *Spurious Rewards* (arXiv:2506.10947), show random rewards raise Qwen2.5-Math-7B on MATH-500 by 21.4 points through GRPO's clipping bias; the reported `random` arm in the math cell is that reference point.

## Natural false positives

Zhang, *When the Reward Suite Is Leaky* (arXiv:2607.11022), runs a preregistered GRPO contrast on MBPP between the original tests and the hardened MBPP+ tests, replicated on two more families. The held-out effect is "non-inferior under a preregistered 1.5-pt margin (gap 0.20 pt)", and the "rewarded false-positive mass tracks a cheap static leakiness audit computed before training (Spearman 0.80)". The comparison has no rate-matched random arm, and the audit predicts false-positive mass, not accuracy. One reading, which is an interpretation and not a claim of that paper, is that those natural false positives are the prompt-specific case of Theorem 3. H7 tests natural verifiers in four domains against resampled noise at their own rates, with the direction forecast by the locked screen.

## Verifier implementations as systematic errors

Xin (arXiv:2609.01354) audits math verifiers by error category over 307,420 verdicts: two configurations of one library "disagree on 49.9% of pairs", and a relative tolerance of $10^{-4}$ accepts every off-by-one answer at magnitude $10^4$ and above. That tolerance is the math domain's lenient natural verifier, and strip-string exact match is its strict one. Xin (arXiv:2609.06386) measures a within-group error correlation of 0.530 on Qwen2.5-1.5B; a prompt-specific rule is correlated within a group and is still predicted to stay local, while the prompt-centred statistics here remove the shared prompt-difficulty component. Ray (arXiv:2606.01066) fuzzes RLVR verifiers before training, and Zhou et al. (arXiv:2608.20362) document language-dependent exact-match false negatives on Qwen3 and Llama 3.1. None of the three forecasts a training outcome.

## Model judges as verifiers

Zhao et al., *One Token to Fool LLM-as-a-Judge* (arXiv:2507.08794), show master-key false positives (a colon, "Let's solve this problem step by step") against model judges and train a robust reward model. Huang et al. (arXiv:2505.22203) find model-based math verifiers "highly susceptible to reward hacking during RL". The qa domain trains against a real judge, TIGER-Lab's general-verifier (the reward of General-Reasoner), with the robust judge of Zhao et al. as a reported second natural verifier, and a strict rule as the oracle. A master key is a shared surface feature, the same role as `fp_trigger`; one of the qa patch categories is exactly the master-key opener.

## Predicting or detecting reward hacking

Beigi et al. (arXiv:2606.09711) show probes during training "forecast later hack onset and severity even when the visible hack rate is still low". GRIFT (Wang et al., arXiv:2604.16242) detects reward hacking from compressed gradients of the chain of thought. Bergen et al. (arXiv:2609.19101) detect hacks with difference-of-means vectors. M. and Karmakar (arXiv:2609.09776) refit a frozen judge on a small on-policy label stream during GRPO. These methods act during or after training. The screen here is computed once, at step 0, before any training, and it is locked; the held-out decodability probe (H4) is its detection-style baseline.

## Reward hacking across tasks

MacDiarmid et al. (arXiv:2511.18397) show that reward hacks learned in production coding environments generalise to broader misalignment, Taylor et al. (arXiv:2508.17511) that supervised training on harmless hacks generalises, and Daniels et al. (arXiv:2609.06649) that iterative preference optimisation can induce it. X1 asks a narrower, measurable question: whether an adapter trained against a leaky verifier in one domain changes oracle accuracy in the others.

## Rate-based corrections

Cai et al. (arXiv:2510.00915) and Mansouri et al. (arXiv:2510.18924) correct the policy gradient for known false-positive and false-negative rates. A correction that uses only the rates is identical for an arm and its rate-matched control, so it cannot explain or remove a difference between them. With group-standardised advantages, an affine reward correction also leaves the GRPO update unchanged.

## Instruction following and keyword stuffing

Tulu 3 (Lambert et al., arXiv:2411.15124) reports keyword stuffing when IFEval is the RLVR reward. The ifeval domain uses AllenAI's multi-constraint data with the IFEval and IFBench checkers (Pyatkin et al., arXiv:2507.02833) as the natural verifier and the checker plus a content judge as the oracle. Salmani-Zarchi et al. (arXiv:2606.06058) study GRPO instability on IFEval, which the per-cell oracle-gain rule guards against.

## Tilts and GRPO theory

Mroueh (arXiv:2503.06639) and Dorner et al. (arXiv:2507.12399, Proposition 4) give the exponential-tilt geometry that Lemma 1 quotes: updates $\pi_t \propto \pi_0 \exp(g_t(x, V))$ preserve $\pi(\cdot \mid V = v, x)$. This repository uses that corollary as the repeated-prompt baseline (P0.1, P0.4b) and does not claim it as new. Proposition 2 is the Euclidean softmax flow, whose $\beta = 0$ limit is modal precision and whose $\beta > 0$ equilibrium is the finite tilt $\pi_0 \exp(V/\beta)$ (P0.4a, P0.4c). Theorem 3 is the part that is not in those papers: fixed errors add $g_{\mathrm{FP}}$ and $g_{\mathrm{FN}}$, and held-out prompts move only through the shared inner product with $g_V$.

## Imperfect verifiers at inference

Stroebl et al. (arXiv:2411.17501) study best-of-$n$ selection with imperfect verifiers at inference. The object here is the training-time counterpart: the verifier is held fixed and used as the GRPO reward. Mean precision and modal precision are the two terminal accuracies of that training dynamics (Lemma 1, Proposition 2), not inference-time selection rules (`vfates.forecast.confusion`).

## Gradient features

LESS (arXiv:2402.04333) uses random projections of LoRA gradients for data selection, and TRAK (Park et al., arXiv:2303.14186) random projections of per-sample gradients for influence. At standard LoRA initialisation ($B = 0$) the trainable step-0 feature is $\partial L / \partial B$, already a random projection of the full weight gradient through the fixed $A$. CountSketch keeps inner products unbiased with variance at most $(\lVert a \rVert^2 \lVert b \rVert^2 + \langle a, b \rangle^2) / D$ (`vfates.features.lora_grad`, `vfates.features.sketch`). The audit saves that $A$ and every training run loads it, so the forecast and the trained adapter share one parameterisation. X5 checks the fate contrast under full-parameter training.

## Sparse RL updates

*RL fine-tunes small subnetworks* (arXiv:2505.11711) and *RL's Razor* (arXiv:2509.04259) study RL updates that concentrate on a small part of the network, the regime in which one step-0 direction can summarise what training reinforces. The knife-edge world (P0.8) is the limitation that comes with the summary: $\tau$ is first-order at $\pi_0$, and a shared flow can leave that neighbourhood (transfer precision about 0.73, terminal held-out accuracy 0.023 on the committed seeds). Every run whose curve rises and then falls by the preregistered amounts is labelled knife-edge, and X7 extends three arms to 600 steps.

## GRPO and its variants

Shao et al., DeepSeekMath (arXiv:2402.03300), introduce GRPO. `vfates.train.grpo_core` is the NumPy reference both trainers are tested against:

- **Advantages:** `std` (TRL `scale_rewards="group"`) and `mean`, the unscaled advantage of Dr. GRPO (Liu et al., arXiv:2503.20783; TRL `scale_rewards="none"`).
- **Reductions:** `seq_mean`, the per-sequence average of GRPO; `token_mean`, the token-level average of DAPO's loss (Yu et al., arXiv:2503.14476), which is only that reduction and not the rest of DAPO; and `dr_constant`, the constant normaliser of Dr. GRPO.

With `seq_mean`, the step-0 direction weights each completion by the inverse of its length, and the audit applies that weight to train rows when it builds $g_V$. X4 repeats three arms under the Dr. GRPO and token-mean objectives, each with an audit that uses its own weighting.

## Tasks and models

Reasoning Gym (Stojanovski et al., arXiv:2505.24760) supplies `decimal_chain_sum`. Big-Math (Albalak et al., arXiv:2502.17387) supplies the math pool, TACO (Li et al., arXiv:2312.14852) the code pool, AllenAI's multi-constraint data (arXiv:2507.02833) the instruction-following pool, and WebInstruct-verified with General-Reasoner's verifier (Ma et al., arXiv:2505.14652) the judged QA pool. The policies are Qwen3 (arXiv:2505.09388) in non-thinking mode, OLMo 3 at its SFT checkpoint, whose training data is public, and Llama 3.1.
