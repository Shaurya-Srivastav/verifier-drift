from vdrift.pilot import run_pilot


def test_pilot_smoke(tmp_path=None):
    out = "results/_test_summary.json"
    s = run_pilot(steps=3, groups_per_step=16, group_size=8, n_problems=32, seed=0, out=out)
    assert len(s["trajectory"]) == 3
    for rec in s["trajectory"]:
        assert 0.0 <= rec["advantage_sign_flip_rate"] <= 1.0
        assert 0.0 <= rec["oracle_acc"] <= 1.0
    assert "decision" in s["verdict"]
