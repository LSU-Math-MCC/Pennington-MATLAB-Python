import numpy as np
from pipeline.geometry.depth_fusion import (robust_affine, information_fuse,
                                            fuse_estimators)


def test_inverse_variance_mean():
    # two constant estimators, known sigmas -> fused = inverse-variance weighted mean
    d1 = np.full((4, 4), 2.0)
    d2 = np.full((4, 4), 3.0)
    s1 = np.full((4, 4), 1.0)
    s2 = np.full((4, 4), 0.5)            # estimator 2 is 4x more precise
    d_hat, var, info = information_fuse([d1, d2], [s1, s2])
    w1, w2 = 1 / 1.0**2, 1 / 0.5**2
    expected = (2.0 * w1 + 3.0 * w2) / (w1 + w2)
    assert np.allclose(d_hat, expected, atol=1e-9)
    assert np.allclose(var, 1.0 / (w1 + w2), atol=1e-9)


def test_information_monotonic():
    # adding a measurement can only shrink posterior variance (info adds)
    d = np.full((3, 3), 1.0)
    s = np.full((3, 3), 1.0)
    _, var1, _ = information_fuse([d], [s])
    _, var2, _ = information_fuse([d, d], [s, s])
    _, var3, _ = information_fuse([d, d, d], [s, s, s])
    assert np.all(var2 <= var1 + 1e-12)
    assert np.all(var3 <= var2 + 1e-12)
    assert np.allclose(var3, 1.0 / 3.0)


def test_missing_measurement_ignored():
    d1 = np.array([[2.0, np.nan]])
    s1 = np.array([[1.0, 1.0]])
    d2 = np.array([[4.0, 4.0]])
    s2 = np.array([[1.0, 1.0]])
    d_hat, var, info = information_fuse([d1, d2], [s1, s2])
    # pixel 0: both -> mean 3; pixel 1: only d2 -> 4 with var 1
    assert np.isclose(d_hat[0, 0], 3.0)
    assert np.isclose(d_hat[0, 1], 4.0)
    assert np.isclose(var[0, 1], 1.0)


def test_prior_as_measurement():
    d = np.full((2, 2), 2.0)
    s = np.full((2, 2), 1.0)
    d_hat, var, info = information_fuse([d], [s], prior=np.full((2, 2), 4.0), prior_sigma=1.0)
    assert np.allclose(d_hat, 3.0)        # mean of measurement and prior
    assert np.allclose(var, 0.5)


def test_robust_affine_recovers_transform():
    rng = np.random.default_rng(0)
    ref = rng.uniform(1, 3, (20, 20))
    d = (ref - 0.5) / 2.0                  # ref = 2*d + 0.5  -> a=2, b=0.5
    mask = np.ones((20, 20), bool)
    a, b, sig = robust_affine(d, ref, mask)
    assert np.isclose(a, 2.0, atol=1e-6)
    assert np.isclose(b, 0.5, atol=1e-6)
    assert sig < 1e-3


def test_fuse_estimators_shrinks_with_more():
    # Calibrated against an EXTERNAL metric reference (the realistic SMPL-X-prior case),
    # adding independent estimators must lower the posterior sigma.
    rng = np.random.default_rng(1)
    truth = np.add.outer(np.linspace(2, 3, 16), np.zeros(16))
    mask = np.ones((16, 16), bool)
    def noisy(scale, sigma):
        return truth / scale + rng.normal(0, sigma, truth.shape)
    one = fuse_estimators([noisy(2.0, 0.02)], mask, reference=truth)
    three = fuse_estimators([noisy(2.0, 0.02), noisy(0.5, 0.02), noisy(1.3, 0.02)],
                            mask, reference=truth)
    assert np.nanmean(three["sigma"]) < np.nanmean(one["sigma"])
