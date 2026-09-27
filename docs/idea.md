# The idea and its delta

## One sentence

A verifier that is fixed *in code* is not fixed *in effect*: as GRPO changes a
policy's answer-form distribution, a real rule verifier's false-negative/false-
positive rates drift within those forms, flipping the group-relative advantage
sign versus ground truth and biasing training — even though RLVR tolerates
random verifier noise.

## Why this is the framing that survives review

The strongest objection to any "verifier is imperfect" paper is *"An Imperfect
Verifier is Good Enough"* (arXiv:2604.07666): RLVR shrugs off ~15% noise. That
result is about **random, symmetric, resampled** noise. Our claim is precisely
the complementary case it leaves open (and its own limitations section flags):
**systematic, persistent, policy-correlated** error that *moves with training*.
So the "who cares" objection becomes the motivation.

## Closest prior work and the exact gap

| Paper | What it did | Gap we fill |
|---|---|---|
| An Imperfect Verifier is Good Enough (2604.07666) | RLVR robust to ~15% random symmetric noise, across domains | Random ≠ systematic drift; asymmetric persistent bias is their future work |
| Delay, Plateau, or Collapse (2605.02909) | Systematic error harms RLVR — but errors are **injected** triggers | Ours is a **real unmodified** verifier's natural error |
| Xin, within-group verifier error (2609.06386) | Form-conditional error + advantage-sign flips on **frozen** rollouts; no training; difficulty not separated | Ours is **over a GRPO trajectory**, difficulty-**controlled** |

## The measurement

Per checkpoint over a GRPO run: (1) the policy's answer-form mix, (2) the
verifier's FP/FN within each form vs an oracle, (3) the fraction of groups whose
advantage sign under the verifier disagrees with the oracle — all stratified by
problem difficulty so the effect is attributable to form drift, not to the
policy solving easier problems.

## Honest odds

Thin but real. The effect may be small (that is exactly what the pilot tests).
Route: mock pilot (this repo) → real GRPO pilot (Mac Studio) → if the drift is
real, a workshop paper first, then ICML/NeurIPS; if null, report it.
