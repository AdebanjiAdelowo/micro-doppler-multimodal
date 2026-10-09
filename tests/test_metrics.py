import numpy as np
import pytest

from mdradar import metrics as M


def test_classification_metrics_hand_computed():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 0, 1, 1, 1, 2, 0])
    m = M.classification_metrics(y, p, 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 1 + 1 / 2) / 3)
    f1 = [2 * 0.75 * 0.75 / 1.5, 2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * 0.5 / 1.5]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    np.testing.assert_array_equal(M.confusion(y, p, 3), [[3, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert M.per_class_report(y, p, ["a", "b", "c"])["c"]["recall"] == pytest.approx(0.5)


def test_ece_perfectly_calibrated_and_overconfident():
    rng = np.random.default_rng(0)
    n = 200000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf
    y = np.zeros(n, dtype=int)
    prob = np.stack([np.where(correct, conf, 1 - conf), np.where(correct, 1 - conf, conf)], axis=1)
    assert M.expected_calibration_error(y, prob) < 0.01
    over = np.tile([0.99, 0.01], (1000, 1))
    y_over = np.r_[np.zeros(700, int), np.ones(300, int)]
    assert M.expected_calibration_error(y_over, over) == pytest.approx(0.29, abs=1e-9)


def test_brier_and_nll_limits():
    y = np.array([0, 1])
    assert M.brier_score(y, np.eye(2)) == 0.0
    assert M.brier_score(y, np.eye(2)[::-1]) == 2.0
    assert M.brier_score(y, np.full((2, 2), 0.5)) == pytest.approx(0.5)
    assert M.nll(y, np.full((2, 2), 0.5)) == pytest.approx(np.log(2))


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    n, k, t_true = 20000, 4, 2.5
    z = rng.standard_normal((n, k)) * 2
    p = M.apply_temperature(z, 1.0)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    t = M.fit_temperature(z * t_true, y)  # logits inflated by t_true are over-confident
    assert t == pytest.approx(t_true, rel=0.05)
    before = M.expected_calibration_error(y, M.apply_temperature(z * t_true, 1.0))
    after = M.expected_calibration_error(y, M.apply_temperature(z * t_true, t))
    assert after < before and after < 0.02
    # temperature scaling never changes the predicted class
    np.testing.assert_array_equal((z * t_true).argmax(1), M.apply_temperature(z * t_true, t).argmax(1))


def test_risk_coverage_and_selective_metrics():
    y = np.array([0, 0, 0, 0])
    prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
    cov, risk = M.risk_coverage(y, prob)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 0.5])
    s = M.selective_metrics(y, prob, coverages=(0.5,))
    assert s["acc_at_cov_50"] == 1.0 and s["aurc"] == pytest.approx(np.mean([0, 0, 1 / 3, 0.5]))


def test_cluster_bootstrap_is_wider_than_iid_for_correlated_data():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 30)
    good = rng.random(20) < 0.7  # each subject is entirely right or entirely wrong
    y = np.zeros(600, dtype=int)
    pred = np.where(good[groups], 0, 1)
    acc = lambda a, b: float(np.mean(a == b))  # noqa: E731
    lo_i, hi_i = M.bootstrap_ci(acc, y, pred, n_boot=400)
    lo_c, hi_c = M.bootstrap_ci(acc, y, pred, groups=groups, n_boot=400)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)
    assert lo_c <= acc(y, pred) <= hi_c
