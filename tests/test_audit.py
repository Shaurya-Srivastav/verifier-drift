from vdrift.audit import (
    Sample,
    advantage_sign_flip_rate,
    fn_rate,
    fp_rate,
    group_advantages,
)


def _s(v, o, form="fraction", diff=1):
    return Sample("p", "g", form, diff, v, o)


def test_group_advantages():
    assert group_advantages([1.0, 0.0]) == [0.5, -0.5]


def test_fn_fp_rates():
    # oracle-correct rejected by verifier -> FN; oracle-wrong accepted -> FP
    samples = [_s(False, True), _s(True, True), _s(True, False), _s(False, False)]
    assert fn_rate(samples) == 0.5   # 1 of 2 oracle-correct was rejected
    assert fp_rate(samples) == 0.5   # 1 of 2 oracle-wrong was accepted


def test_sign_flip_rate():
    # group where verifier and oracle disagree on one member's relative standing
    # verifier rewards [1,0] -> adv [+,-]; oracle rewards [0,1] -> adv [-,+]
    # both members flip sign => rate 1.0
    g = [_s(True, False), _s(False, True)]
    assert advantage_sign_flip_rate([g]) == 1.0
    # perfect verifier: no flips
    g2 = [_s(True, True), _s(False, False)]
    assert advantage_sign_flip_rate([g2]) == 0.0
