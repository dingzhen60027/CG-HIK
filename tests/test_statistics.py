import numpy as np

from confik.experiments.statistics import holm_adjust, paired_bootstrap_difference, paired_sign_flip_pvalue


def test_paired_statistics_and_holm_adjustment() -> None:
    baseline = np.arange(20, dtype=float)
    proposed = baseline - 2.0
    interval = paired_bootstrap_difference(baseline, proposed, samples=500, seed=2)
    assert interval["ci_upper"] < 0.0
    assert paired_sign_flip_pvalue(baseline, proposed, samples=1000, seed=2) < 0.01
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
