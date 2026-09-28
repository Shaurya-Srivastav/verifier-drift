## Summary

What changed, and why. Link the issue it addresses, if any.

## Checks

- [ ] `make lint`, `make typecheck` and `make test` pass
- [ ] New behaviour has a unit test, or a pilot check with a threshold
- [ ] A claim that belongs in the paper points at that check from `docs/theory.md`
- [ ] Held-out prompts were not used to fit a verifier, a calibration or a forecast, or to choose a setting
- [ ] No gate was weakened to make a run pass

## Reported numbers

- [ ] No reported number changed, or the pilot that produces it was rerun and its report is committed here
- [ ] Documents that quote a changed number are updated
- [ ] A design change is logged in `pilots/README.md` (or under Deviations in `PREREGISTRATION.md` once frozen)
- [ ] `CHANGELOG.md` has an entry under "Unreleased"

## Artefacts

- [ ] No weights, checkpoints, `runs/`, `wandb/`, credentials or `cursor_*.md`
- [ ] Pilot output is limited to reports and audit `calibration.json`, `forecasts.json` and `lock.json`
