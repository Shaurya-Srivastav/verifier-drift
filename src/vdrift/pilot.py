"""End-to-end pilot runner.

`run_pilot` simulates a training trajectory with MockPolicy (CPU, seconds) and
runs the full measurement core at each checkpoint, writing a summary JSON and a
mechanical GO/NO-GO against the pre-registered thresholds. This proves the
harness works; the real result comes from swapping MockPolicy for a GRPO'd HF
model on the Mac Studio (see grpo.py) and re-running the identical core.

Run:  python -m vdrift.pilot            (mock smoke pilot)
      python -m vdrift.pilot --steps 8  (more checkpoints)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import (
    Sample,
    advantage_sign_flip_rate,
    fn_by_form,
    fn_rate,
    form_distribution,
    fp_rate,
    sign_flip_by_difficulty,
)
from .data import gen_problems
from .forms import classify_form
from .metrics import difficulty_controlled_drift, drift
from .policy import MockPolicy
from .verifier import Oracle, RuleVerifier

# Pre-registered thresholds (see PREREGISTRATION.md).
THRESH = {
    "sign_flip_drift_min": 0.05,          # sign-flip rate must rise >= 5 pts start->end
    "sign_flip_final_min": 0.10,          # and reach >= 10% at the end
    "difficulty_controlled": True,        # drift must hold within >=1 difficulty bin
}


def _mean_bool(xs) -> float:
    xs = list(xs)
    return sum(1.0 for x in xs if x) / len(xs) if xs else 0.0


def run_pilot(steps=6, groups_per_step=64, group_size=8, n_problems=128, seed=0,
              out="results/pilot_summary.json") -> dict:
    rng_problems = gen_problems(n_problems, seed=seed)
    policy = MockPolicy(seed)
    verifier, oracle = RuleVerifier(), Oracle()

    trajectory = []
    for step in range(steps):
        t = step / (steps - 1) if steps > 1 else 0.0
        groups = []
        for gi in range(groups_per_step):
            prob = rng_problems[(step * groups_per_step + gi) % len(rng_problems)]
            g = []
            for _ in range(group_size):
                pred = policy.rollout(prob, t)
                g.append(Sample(pred, prob.gold, classify_form(pred), prob.difficulty,
                                verifier.check(pred, prob.gold), oracle.check(pred, prob.gold)))
            groups.append(g)
        flat = [s for g in groups for s in g]
        trajectory.append({
            "step": step,
            "t": round(t, 3),
            "verifier_acc": round(_mean_bool(s.v_correct for s in flat), 4),
            "oracle_acc": round(_mean_bool(s.o_correct for s in flat), 4),
            "fn_rate": round(fn_rate(flat), 4),
            "fp_rate": round(fp_rate(flat), 4),
            "advantage_sign_flip_rate": round(advantage_sign_flip_rate(groups), 4),
            "sign_flip_by_difficulty": {int(k): round(v, 4) for k, v in sign_flip_by_difficulty(groups).items()},
            "form_distribution": {k: round(v, 3) for k, v in form_distribution(flat).items()},
            "fn_by_form": {k: round(v, 3) for k, v in fn_by_form(flat).items()},
        })

    sf_drift = drift(trajectory, "advantage_sign_flip_rate")
    sf_final = trajectory[-1]["advantage_sign_flip_rate"]
    diff_ctrl = difficulty_controlled_drift(trajectory)
    diff_ctrl_ok = any(v >= THRESH["sign_flip_drift_min"] for v in diff_ctrl.values())
    go = bool(sf_drift >= THRESH["sign_flip_drift_min"]
              and sf_final >= THRESH["sign_flip_final_min"]
              and (diff_ctrl_ok or not THRESH["difficulty_controlled"]))

    summary = {
        "note": "MOCK-POLICY SMOKE RUN — demonstrates the measurement pipeline; "
                "NOT evidence about real models. Replace MockPolicy with a GRPO'd "
                "HF model (grpo.py) on the Mac Studio for the real result.",
        "config": {"steps": steps, "groups_per_step": groups_per_step,
                   "group_size": group_size, "n_problems": n_problems, "seed": seed},
        "thresholds": THRESH,
        "trajectory": trajectory,
        "sign_flip_drift": round(sf_drift, 4),
        "sign_flip_final": round(sf_final, 4),
        "difficulty_controlled_drift": diff_ctrl,
        "verdict": {"drift_detected": go, "decision": "DRIFT" if go else "NO-DRIFT"},
    }
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--groups-per-step", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--n-problems", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/pilot_summary.json")
    a = ap.parse_args()
    s = run_pilot(a.steps, a.groups_per_step, a.group_size, a.n_problems, a.seed, a.out)
    print("verifier_acc start->end:", s["trajectory"][0]["verifier_acc"], "->", s["trajectory"][-1]["verifier_acc"])
    print("oracle_acc   start->end:", s["trajectory"][0]["oracle_acc"], "->", s["trajectory"][-1]["oracle_acc"])
    print("sign_flip    start->end:", s["trajectory"][0]["advantage_sign_flip_rate"], "->", s["sign_flip_final"],
          f"(drift {s['sign_flip_drift']:+.3f})")
    print("difficulty-controlled drift:", s["difficulty_controlled_drift"])
    print("VERDICT:", s["verdict"]["decision"], "(written to", a.out + ")")


if __name__ == "__main__":
    main()
