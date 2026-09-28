# Theory

This note records the statements the paper uses, the assumptions they need, and the check in this repository that verifies each one.

## Summary

| Result | In one sentence | Checked by |
| --- | --- | --- |
| Lemma 1 (tilt invariance) | Reweighting outputs by a function of the verifier value alone never changes the distribution of outputs within a verdict, so repeated-prompt accuracy moves between the false-omission rate and mean precision. | P0.1, P0.4b |
| Proposition 2 (mode seeking) | Plain policy gradient without a KL penalty concentrates each prompt's accepted mass on its single most likely accepted output (modal precision); with a KL penalty it settles on a finite tilt instead. | P0.4a, P0.4c |
| Theorem 3 (which errors transfer) | The update splits as $g_V = g_O - g_{\mathrm{FN}} + g_{\mathrm{FP}}$: resampled noise only rescales the clean direction, a fixed error adds its own direction, and that direction reaches unseen prompts only through features they share. | P0.2, P0.3, P0.5 |
| Transfer precision | The oracle accuracy of the held-out outputs the step-0 update favours most tracks trained held-out accuracy (Spearman 0.86 on the P0 grid) better than step-0 $J$ does (0.64). | P0.6, P0.7 |
| The knife-edge | Transfer precision is first order: a world exists where it predicts success and the long-horizon flow still collapses. | P0.8 |

Read the sections in order: Notation, the three results, then the estimators (Transfer precision), the GRPO advantage identity, the step-0 LoRA features, and the counterexample. The last section maps every statement to the code and test that checks it.

## Setting

The objects are a fixed verifier $V \in \{0,1\}$, an oracle $O \in \{0,1\}$ that scores correctness, and a policy trained against $V$. Youden's $J$ is

$$
J = 1 - \mathrm{FPR} - \mathrm{FNR}.
$$

Rates are pooled over samples drawn from the step-0 policy on the training prompts. Held-out prompts are never used to fit a verifier or a forecast. The numpy implementations are `vfates.theory` (exact finite worlds) and `vfates.forecast.transfer` (the same estimators on audit samples). The validated toy run is pilot P0: `python pilots/p0_theory/run.py`, report at `pilots/p0_theory/results/report.md`.

## Notation

A prompt $x$ has a finite output set. The step-0 policy is $\pi_0(\cdot \mid x) = \mathrm{softmax}(\mathrm{base}(x))$. A shared parameter $\theta$ enters through a feature map $\phi(x,y)$:

$$
\mathrm{logits}(x) = \mathrm{base}(x) + \phi(x)\,\theta + \mathrm{tab}(x),
$$

with $\mathrm{tab}$ prompt-specific (`vfates.theory.dynamics.logits`). Held-out prompts are scored with $\theta$ only. The shared update at $\pi_0$, prompts weighted equally, is

$$
g_V = \mathbb{E}_x \,\mathrm{Cov}_{\pi_0(\cdot\mid x)}(\phi, V).
$$

`shared_update` in `vfates.theory.dynamics` is this expectation when $\pi_0$ is known. `velocity_direction` in `vfates.forecast.transfer` is the same functional on audit rows: each prompt contributes the mean of $(\text{prompt-centred reward}) \times \phi$, and prompts are averaged. One Euler step of the population flow on $\theta$ equals $\eta g_V$ (`tests/test_theory_dynamics.py`, `test_one_euler_step_matches_the_covariance`).

Pathwise, with $\mathrm{FN} = O(1-V)$ and $\mathrm{FP} = (1-O)V$,

$$
V = O - \mathrm{FN} + \mathrm{FP}.
$$

## Lemma 1 (L1). Tilt invariance

**Statement.** An update of the form

$$
\pi_t(y \mid x) \propto \pi_0(y \mid x) \exp\bigl(g_t\bigl(x, V(x,y)\bigr)\bigr)
$$

preserves the conditional law of the output given the verifier value: $\pi_t(\cdot \mid V = v, x) = \pi_0(\cdot \mid V = v, x)$, on every prompt where $\pi_0(V = v \mid x) > 0$. This is a corollary of the exponential-tilt geometry in Mroueh (arXiv:2503.06639) and of Dorner et al. (arXiv:2507.12399, Proposition 4). The repository checks the finite-world case; it does not re-prove those papers.

**Assumptions.** The output space is finite, so the normaliser is a sum. The multiplier depends on the output only through $V(x,y)$. The base $\pi_0$ is held fixed (the reference policy in the KL term, not a moving snapshot). $V$ itself does not change with $t$.

**Derivation.** Write $Z_t(x) = \sum_y \pi_0(y \mid x) \exp(g_t(x, V(x,y)))$. On the level set $\{y : V(x,y) = v\}$ the factor $\exp(g_t(x,v))$ does not depend on $y$, so it cancels in the conditional:

$$
\pi_t(y \mid V = v, x)
= \frac{\pi_0(y \mid x) \mathbf{1}_{V(x,y)=v}}{\pi_0(V = v \mid x)}
= \pi_0(y \mid V = v, x).
$$

Specialise to the replicator family $g_t(x,v) = t v$, i.e. $\pi_t \propto \pi_0 \exp(t V)$, which is `tilt` in `vfates.theory.dynamics`. For binary $V$ and a prompt that puts positive $\pi_0$-mass on both values, let $\alpha_t(x) = \pi_t(V = 1 \mid x)$,

$$
\mathrm{Prec}_0(x) = \mathbb{E}_{\pi_0}[O \mid V = 1, x], \qquad
\mathrm{FOR}_0(x) = \mathbb{E}_{\pi_0}[O \mid V = 0, x].
$$

$\mathrm{FOR}_0$ is the false-omission rate. Invariance gives

$$
\mathbb{E}_{\pi_t}[O \mid x] = \alpha_t(x)\,\mathrm{Prec}_0(x) + \bigl(1 - \alpha_t(x)\bigr)\,\mathrm{FOR}_0(x).
$$

So each such prompt's accuracy lies on the segment joining its own $\mathrm{FOR}_0(x)$ and $\mathrm{Prec}_0(x)$. As $t \to +\infty$, $\alpha_t \to 1$ and accuracy tends to mean precision. As $t \to -\infty$, $\alpha_t \to 0$ and accuracy tends to the false-omission rate. Prompt-averaged accuracy therefore has those two prompt-means as its endpoints. It is not, in general, the straight line between them at a common mixing weight, because $\alpha_t(x)$ depends on the prompt's base odds of acceptance.

If a prompt has no $\pi_0$-mass on $\{V = 1\}$ (respectively $\{V = 0\}$), that endpoint is undefined. The forecasts then substitute the prompt's base accuracy $\mathbb{E}_{\pi_0}[O \mid x]$. That convention is `mean_precision` in `vfates.theory.forecasts` and `mean_precision_forecast` / `for_prompt_mean` in `vfates.forecast.confusion`.

**Checks.** P0.1 runs `check_tilt_invariance` (`vfates.theory.identities`) at horizons $t \in \{1, 5, 20\}$ on a hand-featured world and a random-feature world, for arms `fpF` and `fnC+fpI`. The committed report's maximum absolute conditional error is $7.22 \times 10^{-16}$, against a gate of $10^{-10}$. The identity does not use the sign of $t$; the pilot evaluates positive horizons. P0.4b evaluates the $t \to +\infty$ endpoint: a tilt at horizon 80 matches prompt-mean precision to $1.11 \times 10^{-16}$ (gate $10^{-6}$).

## Proposition 2 (P2). Euclidean policy gradient is mode-seeking

**Statement.** Under a tabular softmax parameterisation and the objective $\mathbb{E}_\pi[V] - \beta \,\mathrm{KL}(\pi \| \pi_0)$, Euclidean gradient ascent on the logits is mode-seeking inside a level set of $V$. For $\beta = 0$ and two outputs with the same verifier value $v$,

$$
\frac{d}{dt} \log \frac{\pi_y}{\pi_{y'}} = \eta \, a_t \, (\pi_y - \pi_{y'}),
\qquad a_t = v - \mathbb{E}_\pi[V].
$$

With $\beta = 0$ the limit, when the mode is unique, is modal precision: the oracle label of the highest-probability accepted output. For every $\beta > 0$ the unique equilibrium is the finite tilt $\pi_0 \exp(V / \beta)$. Accuracy there lies between base accuracy and mean precision when $J > 0$. Modal precision is the $\beta = 0$ trajectory's endpoint. It is not an equilibrium of any $\beta > 0$ objective.

**Assumptions.** One prompt, or a tabular logit $\mathrm{tab}(x)$ that can realise every distribution on that prompt (`population_flow(..., param="tabular")`). The gradient is Euclidean in the logits, which is the update implemented: the logit step is $\eta\,\pi$ times the centred advantage (`vfates.theory.dynamics`). $V$ is binary for the limit statements. The $\beta > 0$ equilibrium uses a policy class that contains every distribution; a shared linear head (Theorem 3) cannot realise an arbitrary per-prompt tilt. P0.4a and P0.4c are tabular.

**Derivation of the pairwise identity.** $\pi = \mathrm{softmax}(z)$. The Euclidean gradient of expected reward is $\partial \mathbb{E}_\pi[V] / \partial z_y = \pi_y (V_y - \mathbb{E}_\pi[V])$. With $\beta = 0$,

$$
\dot z_y = \eta \, \pi_y \bigl(V_y - \mathbb{E}_\pi[V]\bigr).
$$

Softmax log-odds are logit differences, $\log(\pi_y / \pi_{y'}) = z_y - z_{y'}$. If $V_y = V_{y'} = v$, then $a_t = v - \mathbb{E}_\pi[V]$ is shared by the pair and

$$
\frac{d}{dt} \log \frac{\pi_y}{\pi_{y'}} = \eta \, a_t \, (\pi_y - \pi_{y'}).
$$

On the accepted set, $v = 1$ and $a_t = \pi(V = 0) \ge 0$, and $a_t > 0$ while any rejected mass remains. The sign of the derivative equals the sign of $\pi_y - \pi_{y'}$, so the $\pi_0$-ordering inside $\{V = 1\}$ is preserved and strict gaps grow. Starting from $\pi_0$, mass in the accepted set concentrates on the unique $\pi_0$-mode of that set, and rejected mass goes to zero. The terminal accuracy is the mean over prompts of $O$ at that mode. `modal_precision` breaks exact ties toward the lowest output index; under the flow, exact ties have derivative zero and stay ties. The audit analogue clusters accepted samples by answer string (`modal_precision_forecast` in `vfates.forecast.confusion`).

**Equilibrium for $\beta > 0$.** The objective is strictly concave in $\pi$. Stationarity of $\mathbb{E}[V] - \beta \,\mathrm{KL}(\pi \| \pi_0)$ under $\sum_y \pi_y = 1$ gives

$$
V(y) - \beta \log \frac{\pi(y)}{\pi_0(y)} = \lambda,
$$

hence $\pi \propto \pi_0 \exp(V / \beta)$. The temperature of the tilt is $t = 1/\beta$. For $J > 0$, precision exceeds prevalence, so this positive-temperature tilt sits strictly between base accuracy ($t = 0$) and mean precision ($t \to +\infty$), on each prompt's Lemma 1 segment. The $\beta = 0$ mode is a different point: it is where the unconstrained Euclidean flow ends after it has left the exponential family. It is not stationary for any $\beta > 0$.

**Checks.** P0.4a: tabular flow at $\beta = 0$, arm `fpF`, 800 steps, terminal accuracy within $0.0012$ of modal precision (gate $0.03$). P0.4b: the replicator endpoint, above. P0.4c: tabular flows at $\beta \in \{0.5, 0.1\}$ against `tilt` at $1/\beta$. The committed report has a maximum gap of $0.0120$ (gate $0.02$). At $\beta = 0.5$ the equilibrium accuracy is $0.3701$, mean precision is $0.5028$, and modal precision is $0.3000$. If the step budget is short of the equilibrium the check is recorded as `documented` rather than failed (`src/vfates/pilots/p0_theory.py`). The same $\beta = 0.5$ and $\beta = 0$ limits are unit-tested in `tests/test_theory_dynamics.py` (`test_beta_equilibrium_is_the_tilt_and_beta0_is_the_mode`).

## Theorem 3 (T3). Which errors move held-out prompts

**Statement.** Because $V = O - \mathrm{FN} + \mathrm{FP}$ holds on every sample,

$$
g_V = g_O - g_{\mathrm{FN}} + g_{\mathrm{FP}}.
$$

Class-conditional noise with rates $\rho_0 = \mathbb{P}(V = 1 \mid O = 0)$ and $\rho_1 = \mathbb{P}(V = 0 \mid O = 1)$, redrawn independently of $\phi$ given $O$, satisfies $\mathbb{E}[g_V] = J g_O$ with $J = 1 - \rho_0 - \rho_1$. Fixed errors add the bias vectors $g_{\mathrm{FP}}$ and $g_{\mathrm{FN}}$, whose directions are the features those errors correlate with. A held-out prompt $x'$ moves only through $\theta$. To first order,

$$
s(y) = \langle \phi(x', y),\, g_V \rangle
$$

is the growth rate of $\log \pi(y \mid x')$.

**Assumptions.** Covariance is linear in the reward, which it is. The resampled draw uses $\mathbb{E}[V \mid O] = \rho_0 + J O$ and does not depend on $\phi$ except through $O$. The first-order velocity freezes $g_V$ at $\pi_0$; it does not follow $g$ as $\pi$ moves. Prompt-specific logits $\mathrm{tab}$ are not applied on held-out prompts (`test_fp_directions_and_heldout_ignores_tabular_head`).

**Decomposition.** Substitute $\mathrm{FN} = O(1-V)$ and $\mathrm{FP} = (1-O)V$:

$$
O - O(1-V) + (1-O)V = O - O + OV + V - OV = V.
$$

Taking $\mathrm{Cov}_{\pi_0}(\phi, \cdot)$ and then $\mathbb{E}_x$ yields the split. P0.2: `check_decomposition` on arms `fpF`, `fnC+fpI`, and `fnI`, maximum absolute error $4.86 \times 10^{-17}$ (gate $10^{-10}$).

**Resampled noise.** Conditioning,

$$
\mathbb{E}[V \mid O] = (1-\rho_1) O + \rho_0 (1-O) = \rho_0 + J O.
$$

Independence of the coins and $\phi$ given $O$ pulls the expectation inside the covariance. Covariance with the constant $\rho_0$ is zero, so $\mathbb{E}[g_V] = J g_O$. The direction matches the oracle update; the scale is $J$. P0.3a and P0.3b: `check_j_scaling` with $\rho_0 = 0.12$, $\rho_1 = 0.18$, 200 draws. Cosine between the Monte Carlo mean of $g_V$ and $J g_O$ is $1.0000$ (gate $0.999$); the least-squares multiplier differs from 1 by $0.0017$ (gate $0.02$).

**Fixed errors.** $g_{\mathrm{FP}}$ is the covariance of $\phi$ with a bit that is on only for accepted incorrect outputs. Two cases matter.

- **Shared.** The bit is a function of a shared behaviour (a numeric tolerance, a trigger phrase, a surface form), and the same behaviour has a coordinate in $\phi$ on held-out prompts. Then $g_{\mathrm{FP}}$ has a component along that coordinate, and $s(y)$ is large for held-out outputs that carry it.
- **Prompt-specific.** The bit depends on the prompt (a hash of the prompt id and the answer, or a random subset of prompts) and is independent of the shared coordinates of $\phi$. Then the prompt-averaged covariance is zero: there is no feature of the completion that the error selects on every prompt.

In the toy worlds (`vfates.theory.worlds`) the shared false positive `fpF` accepts group `H`, the only group with the exploit coordinate, and `shared_update` loads positively on that coordinate. The twin `fpI` accepts group `I`, which has the same base logits as `H` and a zero exploit coordinate; the exploit component of its update is negative (`test_fp_directions_and_heldout_ignores_tabular_head`).

The chain-sum arms in `configs/arms/chain_sum.yaml` follow the same partition: `fp_hash` and `fn_prompt_subset` are prompt-specific; `fp_tolerance`, `fp_trigger`, `fn_form`, and `fn_behavior` are shared; `noise_fp`, `noise_fn` and `noise_fn_form` redraw class-conditional noise at the rates of `fp_tolerance`, `fn_behavior` and `fn_form`.

**Same $J$, different held-out accuracy.** P0.5 builds `fpF`, `fpI`, and class-conditional noise matched to `fpF`'s pooled step-0 rates (`matched_resampled`). On `hand_noisy` and `rf05` the three arms share $J$ to a spread of $0$ (gate $10^{-8}$). Held-out accuracy of the resampled arm exceeds `fpF` by $0.9938$ and $0.9913$ (gate $0.5$). The resampled arm stays within $7.55 \times 10^{-5}$ and $8.02 \times 10^{-5}$ of `fpI` (gate $0.05$). Shared structure is not synonymous with collapse: on `hand_clean`, where the skill coordinate is pure, `fpF` still has the same $J$ as `fpI` but exact transfer precision remains $1$ and trained held-out accuracy is $0.9949$. The screen below is what distinguishes these cells. $J$ does not.

## Transfer precision

### Exact $\tau$

The exact statistic (`exact_tau` in `vfates.theory.forecasts`) scores every held-out output, not a sample. With $s = \phi g_V$, each held-out prompt contributes the $\pi_0$-weighted oracle accuracy of the outputs that attain the maximum of $s$ (ties within the same relative tolerance as `tau_hard`, $10^{-9}$). That is the oracle accuracy of the output whose log-probability grows fastest.

P0.6 compares this number with trained held-out accuracy on 4 worlds $\times$ 7 arms (28 cells), each cell a mean of 3 sampled-GRPO seeds. In the committed report, Spearman correlation with held-out accuracy is $0.8600$ (gate $0.8$), against $0.6353$ for step-0 $J$ and $0.5526$ for step-0 mean precision (`report.json`, key `tables.rank`). The fraction of cells with $|\tau - \text{observed}| \le 0.10$ is $0.9643$ (27/28; gate $0.90$). The cell outside the band is `rf05` / `fnC`: exact $\tau = 0.3642$, observed held-out accuracy $0.1903$.

### Soft tilt and the hard limit

Audit samples are drawn from $\pi_0$. After a tilt of strength $\kappa$ along $s$, self-normalised importance sampling gives

$$
\hat\tau(\kappa) = \mathbb{E}_{x'} \frac{\sum_i O_i e^{\kappa s_i}}{\sum_i e^{\kappa s_i}}.
$$

This is `tau_soft`. The implementation divides $s$ by the pooled within-prompt root-mean-square so that $\kappa$ is unit-free. $\kappa = 0$ returns base accuracy. $\kappa \to +\infty$ is `tau_hard`: the mean over prompts of the oracle label of the highest-velocity sample, ties averaged. Both endpoints are unit-tested (`tests/test_forecast_transfer.py`, `test_tau_soft_endpoints`).

### Coverage and inflation

With $n$ samples per prompt, the sample selected by `tau_hard` is correct only if that prompt has a correct sample and the argmax picked it. Hence

$$
\tau_{\mathrm{hard}} \le \text{coverage},
$$

where coverage is the fraction of held-out prompts with at least one correct sample (`TransferForecast.coverage`). The inequality is an audit ceiling. It is not the quantity of interest.

`tau_hard` is also inflated on prompts that never sampled the favoured behaviour. If the high-velocity output is an incorrect exploit and the audit missed it, the argmax among the samples that were drawn can be a correct output. The statistic then reports success on a prompt where the true top-velocity output is wrong. `tau_rel` divides `tau_hard` by `tau_hard` computed in the oracle direction $g_O$. The two ceilings match, so the ratio cancels the coverage cap. It does not cancel the inflation: the oracle direction and the verifier direction miss different outputs. In the committed P0 report the Spearman correlation of 16-sample `tau_rel` with trained held-out accuracy is $0.7769$. P0.7 records that number as `documented`. It is not a gate.

`test_tau_pool_recovers_a_shared_exploit_missing_on_most_prompts` builds a shared exploit that is present on every training prompt and on only a quarter of held-out prompts. There `tau_pool < 0.05` while `tau_hard > 0.4`.

### Pooled tail

The primary finite-sample statistic is `tau_pool(q)` (`PRIMARY_Q = 1/32` in `vfates.forecast.transfer`). Prompt-centre the velocities, pool them across held-out prompts, and take the oracle accuracy of the top fraction $q$. The tail has $k = \max(1, \mathrm{round}(q N))$ samples. Samples tied with the $k$-th value share the remaining slots equally, so the value does not depend on row order (`test_tau_pool_splits_boundary_ties_evenly`).

Pooling is the reason to prefer this over a per-prompt argmax. A behaviour carried by a shared feature is favoured on every prompt, including prompts whose $n$ samples happened to miss it. The global tail can still be filled by the prompts that did sample it. $q = 1/32$ was fixed on these toy worlds before any language-model run; the other fractions in `DEFAULT_QS` are sensitivity checks, stored as `tau_pool_curve`.

**Prompt-mean centring.** The pooled tail uses velocities with the per-prompt mean removed (`prompt_centered`). A raw velocity contains an additive prompt-level offset. Across prompts those offsets dominate the ranking, and the global top fraction selects whole prompts rather than the favoured behaviour inside each prompt. `test_tau_pool_takes_the_pooled_prompt_centred_tail` shifts one prompt's scores by $+100$. After centring, the top eighth is the within-prompt outlier, whose label is 0. Without centring that tail would be taken from the shifted prompt.

**P0.7.** On the same 28 cells, with 16 $\pi_0$ samples per prompt and one audit draw shared by every arm of a world (seed $50000 + \text{data seed}$):

| Statistic | Committed value | Role |
| --- | --- | --- |
| Spearman of `tau_pool(1/32)` with held-out accuracy | 0.8617 | Gate, threshold 0.8. Also required to beat $J$ and mean precision. |
| Fraction of cells with $\lvert \tau_{\mathrm{pool}} - \text{observed} \rvert \le 0.10$ | 1.0000 (28/28) | Gate, threshold 0.9. |
| Spearman of `tau_rel` | 0.7769 | Documented. Not a gate. |
| Mean split-half cosine of $g_V$ | 0.9871 | Gate, threshold 0.9 (`split_half_reliability`). |

The `rf05` / `fnC` cell that exact $\tau$ misses by $0.1739$ has `tau_pool` $0.1217$, which is inside $0.10$ of the observed accuracy $0.1903$. That is a fact about this grid. It is not a claim that the sampled tail dominates the exact statistic in general.

## GRPO advantage identity

Inside a group of size $G$, rewards are centred by their group mean. Let $O_1,\ldots,O_G$ be i.i.d. Bernoulli($p$), and conditional on $O_i$ let $V_i$ be the class-conditional verifier, independent across $i$ given the oracle bits, with $\mathbb{E}[V_i \mid O_i] = \rho_0 + J O_i$. The mean-centred advantage of member 1 is

$$
A_1 = V_1 - \frac{1}{G}\sum_{j=1}^{G} V_j = \Bigl(1 - \frac{1}{G}\Bigr) V_1 - \frac{1}{G}\sum_{j=2}^{G} V_j.
$$

Take $\mathbb{E}[\,\cdot \mid O_1]$. The other members have expectation $\mathbb{E}[V] = \rho_0 + J p$, and $\mathbb{E}[V_1 \mid O_1] = \rho_0 + J O_1$, so

$$
\begin{aligned}
\mathbb{E}[A_1 \mid O_1]
&= \Bigl(1 - \frac{1}{G}\Bigr)(\rho_0 + J O_1) - \Bigl(1 - \frac{1}{G}\Bigr)(\rho_0 + J p) \\
&= \Bigl(1 - \frac{1}{G}\Bigr) J (O_1 - p).
\end{aligned}
$$

The false-positive rate cancels. The identity holds for any $(\rho_0, \rho_1)$ with the stated $J$, not only for symmetric noise. The factor $1 - 1/G$ is the effect of including the sample in the mean it is centred by.

Dividing by the group's standard deviation, when that standard deviation is positive, multiplies every advantage in the group by the same positive constant (the implementations add $10^{-8}$ in the identity check and $10^{-4}$ in `vfates.train.grpo_core`, and they set the advantage to 0 when the standard deviation is 0). A positive scale does not change the sign of a nonzero advantage. A zero-variance group already has centred advantage 0. P0.3c draws groups at $G \in \{4,8,16\}$, $p \in \{0.2, 0.5\}$, $J \in \{0.4, 0.9\}$, with symmetric rates $\mathrm{FPR} = \mathrm{FNR} = (1-J)/2$, and compares the mean-centred advantage with the identity using a prompt-free cluster standard error. The maximum $\lvert z \rvert$ in the committed report is $1.5915$ (gate $4$), and the fraction of nonzero advantages whose sign flips under standard-deviation scaling is $0$.

`grpo_core.group_advantages` is the trainer reference: unbiased standard deviation matches TRL's `scale_rewards="group"`, and `mode="mean"` is Dr.GRPO (centre only). The sign statement does not depend on which of these positive scales is used. The conditional-expectation identity above is for the mean-centred advantage, before that scale.

## LoRA features at step 0

Standard LoRA initialises $A$ at random and $B = 0$ (`init_lora_weights=True` in `lora_grad_features`). The adapter output is $Y = s \, B(Ax)$ with scaling $s = \alpha / r$. At $B = 0$, $Y = 0$.

Let $u = Ax$. Then $Y = s B u$, so

$$
\frac{\partial L}{\partial B} = s \,\frac{\partial L}{\partial Y}\, u^{\top} = s \,\frac{\partial L}{\partial Y}\, (Ax)^{\top},
$$

and

$$
\frac{\partial L}{\partial u} = s B^{\top} \frac{\partial L}{\partial Y} = 0 \quad\text{at}\quad B = 0,
$$

hence $\partial L / \partial A = 0$. The first optimiser step moves only $B$.

The gradient of a full linear map $Y = Wx$ is $(\partial L / \partial Y) x^{\top}$. The $B$ gradient is that expression with $x$ replaced by $Ax$. $A$ is a fixed random projection of the layer input, so $\partial L / \partial B$ is a random projection of the full weight gradient. Sketching $B$, and not $A$, is the step-0 feature of the parameters GRPO trains. `lora_grad_features` raises if the maximum absolute $A$ gradient on the first sample exceeds $10^{-5}$ in float32 ($10^{-2}$ in reduced precision).

`normalize="sum"` differentiates the sum of completion-token log-probabilities. `normalize="token_mean"` differentiates their mean, which is the sum-gradient divided by the completion length. `velocity_direction(..., lengths=)` divides features by length again. Pass `lengths` only for the sum features.

### CountSketch

For vectors $a, b$ and a CountSketch $S$ of dimension $D$ that uses the same buckets $h$ and signs $\sigma(i) \in \{\pm 1\}$ on both,

$$
\langle Sa, Sb \rangle = \sum_{i,j} a_i b_j \, \sigma(i)\sigma(j) \, \mathbf{1}_{h(i)=h(j)}.
$$

If $i = j$, the summand is $a_i b_i$. If $i \ne j$, $\mathbb{E}[\sigma(i)\sigma(j)] = 0$ and the hash is independent of the signs, so the expectation vanishes. Therefore $\mathbb{E}[\langle Sa, Sb \rangle] = \langle a, b \rangle$.

For the variance, the cross term $Z = \sum_{i \ne j} a_i b_j \sigma(i)\sigma(j) \mathbf{1}_{h(i)=h(j)}$ has mean zero. In $\mathbb{E}[Z^2]$ a pair of summands survives only when the two unordered pairs of indices coincide, and $\mathbb{P}(h(i)=h(j)) = 1/D$, which gives

$$
\mathrm{Var}(\langle Sa, Sb \rangle) = \frac{1}{D}\Bigl(\lVert a \rVert^2 \lVert b \rVert^2 + \langle a,b \rangle^2 - 2 \sum_i a_i^2 b_i^2\Bigr).
$$

The sum is nonnegative, so

$$
\mathrm{Var}(\langle Sa, Sb \rangle) \le \frac{\lVert a \rVert^2 \lVert b \rVert^2 + \langle a,b \rangle^2}{D}.
$$

`CountSketch` hashes each tensor with `numpy.random.Generator` seeded by `stable_seed(seed, tensor_index, numel)`, so the buckets do not depend on Python's hash randomisation. `tests/test_sketch.py` (`test_monte_carlo_unbiased_and_variance_bound`) checks that the Monte Carlo mean of the sketched inner product sits within 4 standard errors of $\langle a,b \rangle$, and that the empirical variance does not exceed the bound beyond a slack for the error in the variance estimate. Linearity and bitwise CPU agreement with the torch path are tested in the same file.

## The knife-edge (P0.8)

$\tau$ uses $g_V$ at $\pi_0$. It does not describe a later $g$ evaluated at a policy that has already moved. `knife_edge_world` is a four-atom counterexample, identical base masses on every prompt:

| Atom | Oracle | Verifier | Feature | Mass |
| --- | --- | --- | --- | --- |
| $C$ | 1 | 1 | skill, amplitude 1 | 0.15 |
| $H$ | 0 | 1 | exploit, amplitude 1.10 | 0.12 |
| $I$ | 0 | 0 | $0.99$ of the skill direction | $0.02 \times 0.15 = 0.003$ |
| $W$ | 0 | 0 | noise only | 0.727 |

Mean precision is $0.15 / 0.27 = 0.5556$. The only false positives are $H$, so $\mathrm{FPR} = 0.12 / 0.85$ and $J = 0.8588$, with false-negative rate 0. At $\pi_0$ the rejected impurity is 2% of the correct mass, so the skill coordinate is still strongly associated with $V$. The top-velocity output is usually $C$: the three committed seeds give $\tau \in \{0.7325, 0.7400, 0.7325\}$. The initial slope of held-out accuracy is $0.0119 > 0$, and held-out accuracy rises from $0.15$ to a peak between $0.5465$ and $0.5768$.

Along the shared flow the skill coordinate also raises the logit of $I$, which is rejected. As $I$ gains mass, $\mathrm{Cov}(\text{skill}, V)$ falls. The exploit coordinate is carried only by $H$, which remains accepted, so it does not pick up that penalty. The flow leaves the neighbourhood where the step-0 linearisation was valid and collapses onto $H$. Mean terminal held-out accuracy in the committed run is $0.0226$.

`impurity_amp = 0.99` rather than 1 is deliberate. At amplitude 1 the correct atom and the impure atom tie, and the $\pi_0$-weighted tie average is about $0.98$. That world also collapses. The one-percent detune, plus feature noise of scale $0.008$, makes the top-velocity output unique and lands near the values above. P0.8 is `documented`. A failure of the knife-edge to rise and then collapse would mean the counterexample had not been reproduced; it would not mean the screen had passed a gate.

## Where each statement is checked

| Statement | Code | Pilot or test |
| --- | --- | --- |
| Tilt leaves $\pi(\cdot \mid V=v, x)$ unchanged | `identities.check_tilt_invariance` | P0.1 |
| $t \to +\infty$ accuracy is mean precision | `forecasts.mean_precision`, `dynamics.tilt` | P0.4b |
| $\beta = 0$ tabular limit is modal precision | `forecasts.modal_precision` | P0.4a; `test_beta_equilibrium_is_the_tilt_and_beta0_is_the_mode` |
| $\beta > 0$ equilibrium is $\pi_0 \exp(V/\beta)$ | `population_flow(..., beta=)` | P0.4c |
| $g_V = g_O - g_{\mathrm{FN}} + g_{\mathrm{FP}}$ | `identities.check_decomposition` | P0.2 |
| $\mathbb{E}[g_V] = J g_O$ under resampled noise | `identities.check_j_scaling` | P0.3a, P0.3b |
| $\mathbb{E}[A \mid O] = (1-1/G) J (O-p)$; std scaling keeps the sign | `identities.check_group_advantage_scaling` | P0.3c |
| Same $J$, separated held-out fates | `worlds.matched_resampled`, `sampled_grpo` | P0.5 |
| Exact $\tau$ tracks held-out accuracy | `forecasts.exact_tau` | P0.6 |
| 16-sample `tau_pool(1/32)` is the finite-sample gate | `transfer.tau_pool` | P0.7 |
| Per-prompt `tau_rel` is inflated by coverage | `transfer.tau_hard`, `TransferForecast.tau_rel` | P0.7, documented; `test_tau_pool_recovers_a_shared_exploit_missing_on_most_prompts` |
| Prompt-mean centring, not raw velocity | `transfer.prompt_centered` | `test_tau_pool_takes_the_pooled_prompt_centred_tail` |
| First-order screen can miss a later collapse | `worlds.knife_edge_world` | P0.8, documented; `test_knife_edge_rises_then_collapses` |
| $\partial L/\partial A = 0$ and $\partial L/\partial B = s (\partial L/\partial Y)(Ax)^\top$ at $B = 0$ | `features.lora_grad` | Runtime check on the first sample inside `lora_grad_features` |
| CountSketch unbiasedness and variance bound | `features.sketch.CountSketch` | `tests/test_sketch.py` |
