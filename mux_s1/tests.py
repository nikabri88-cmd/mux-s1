"""Unit and invariant tests for MUX-S1. Run: python -m mux_s1.tests"""
import itertools

import numpy as np

from . import core as C
from . import experiments as X


def test_hmm_recovers_T():
    rng = np.random.default_rng(1)
    P, M, Tn, p = 3, 60, 70, 0.12
    z = np.zeros((P, M, Tn), dtype=int)
    z[:, :, 0] = rng.integers(0, 2, (P, M))
    for t in range(1, Tn):
        flip = rng.random((P, M)) < p
        z[:, :, t] = np.where(flip, 1 - z[:, :, t - 1], z[:, :, t - 1])
    E = np.zeros((P, M, Tn, 2))
    noise = rng.standard_normal((P, M, Tn))
    E[..., 0] = np.where(z == 0, 2.0, -2.0) + noise
    E[..., 1] = -E[..., 0]
    Tm = C.hmm_fit_batch(E)
    assert np.allclose(Tm[:, 0, 1], p, atol=0.03), Tm
    zz = C.viterbi_batch(E, Tm)
    assert (zz == z).mean() > 0.95


def test_viterbi_batch_matches_single():
    rng = np.random.default_rng(2)
    E = rng.standard_normal((2, 5, 9, 2))
    Tm = np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.1, 0.9]]])
    zb = C.viterbi_batch(E, Tm)
    for p in range(2):
        assert np.array_equal(zb[p], C.viterbi(E[p], Tm[p]))


def test_viterbi_bruteforce():
    rng = np.random.default_rng(3)
    E = rng.standard_normal((1, 1, 7, 2))
    Tm = np.array([[[0.7, 0.3], [0.4, 0.6]]])
    best, arg = -np.inf, None
    for path in itertools.product([0, 1], repeat=7):
        v = np.log(0.5) + E[0, 0, 0, path[0]]
        for t in range(1, 7):
            v += np.log(Tm[0, path[t - 1], path[t]]) + E[0, 0, t, path[t]]
        if v > best:
            best, arg = v, path
    assert tuple(C.viterbi_batch(E, Tm)[0, 0]) == arg


def test_forward_scaled_bruteforce():
    rng = np.random.default_rng(4)
    K, Tn = 3, 5
    E = rng.standard_normal((2, Tn, K)) * 3
    Km = rng.random((K, K)); Km /= Km.sum(1, keepdims=True)
    pi = rng.random(K); pi /= pi.sum()
    for n in range(2):
        tot = 0.0
        for path in itertools.product(range(K), repeat=Tn):
            p = pi[path[0]] * np.exp(E[n, 0, path[0]])
            for t in range(1, Tn):
                p *= Km[path[t - 1], path[t]] * np.exp(E[n, t, path[t]])
            tot += p
        assert np.isclose(C.forward_loglik_scaled(E, Km, pi)[n], np.log(tot))
        assert np.isclose(C.forward_loglik(E[n:n + 1], np.log(Km), np.log(pi))[0], np.log(tot))


def test_fit_D_exponential():
    D = 67.0
    acf = 0.5 * np.exp(-2 * 10 * np.arange(21) / D)
    acf[0] = 1.0
    Dh, rho, zc = C.fit_D(acf)
    assert abs(Dh - D) < 3, Dh


def test_decoder_lofo():
    rng = np.random.default_rng(5)
    P = rng.standard_normal((80, 30)); Y = rng.standard_normal((80, 4))
    dec = C.Decoder(P, Y, K=10)
    B = dec.weights_lofo(7)
    X_ = np.delete(dec.X, 7, axis=0); Y_ = np.delete(Y, 7, axis=0)
    B2 = np.linalg.lstsq(X_, Y_, rcond=None)[0]
    assert np.allclose(B, B2, atol=1e-8)


def test_lofo_linear():
    rng = np.random.default_rng(6)
    F = rng.standard_normal((40, 5)); R = rng.standard_normal((40, 3))
    loo, _ = C.lofo_linear_predictions(F, R)
    Xf = np.hstack([F, np.ones((40, 1))])
    j = 11
    w = np.linalg.lstsq(np.delete(Xf, j, 0), np.delete(R, j, 0), rcond=None)[0]
    assert np.allclose(loo[j], Xf[j] @ w)


def test_masked_rate():
    rng = np.random.default_rng(7)
    cf = rng.poisson(1.0, (3, C.NB, 4)).astype(np.float32)
    mask = rng.random((3, C.NB)) < 0.4
    ref = cf[mask].sum(0) / (mask.sum() * C.DT)
    assert np.allclose(X.masked_rate(cf, mask), ref, rtol=1e-5)


def test_quarter_kurtosis_telegraph():
    """Signal +-1 shared by four noisy channels: excess kurtosis of the shared signal -> -2."""
    rng = np.random.default_rng(8)
    n = 400000
    s = rng.choice([-1.0, 1.0], n)
    L = [0.5 * s + rng.standard_normal(n) for _ in range(4)]
    m4 = np.mean(L[0] * L[1] * L[2] * L[3])
    m2 = np.mean([np.mean(L[0] * L[1]) * np.mean(L[2] * L[3]), np.mean(L[0] * L[2]) * np.mean(L[1] * L[3]),
                  np.mean(L[0] * L[3]) * np.mean(L[1] * L[2])])
    assert abs(m4 / m2 - 3 - (-2)) < 0.25


def test_telegraph_dwell():
    rng = np.random.default_rng(9)
    D = 60.0
    flips = rng.random((4000, C.SIM_MS)) < 1 / D
    assert abs(flips.mean() * 1000 - 1000 / D) < 0.3


def test_gamma_mean_and_balance():
    rng = np.random.default_rng(10)
    m = C.dyn_gamma(rng, 3000, 60.0, 6)
    assert abs(m.mean() - 0.5) < 0.02


def test_tls():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(5000); y = 0.5 * x + 0.01 * rng.standard_normal(5000)
    assert abs(C.tls_slope(x, y) - 0.5) < 0.01




def test_rate_equations():
    rng = np.random.default_rng(12)
    pop = C.Population(rng, 50, 0.4)
    F = C.make_faces(rng, 2)
    d = pop.delta(F)
    # pure conscious state, no adaptation: rate = b + drive of the conscious face
    spec = dict(kind="average")
    lam, _ = C.binocular_rates(rng, spec, pop, d[0], d[1], 2, 60.0, 1000.0)
    assert np.allclose(lam, np.maximum(pop.b + 0.5 * d[0] + 0.5 * d[1], 0.5))
    # tuning-only adaptation keeps the mean face drive s
    specA = dict(kind="average", adapt_a=0.4, adapt_tau=1000.0, adapt_target="tuning")
    Fm = C.make_faces(rng, 4000)
    dm = pop.delta(Fm)
    g = C.adapt_gain(C.BIN_CENTERS, 0.4, 1000.0)
    lamP = np.stack([C.priming_rates(specA, pop, dm[i], 1)[0] for i in range(0, 4000, 40)])
    lam0 = np.stack([C.priming_rates(dict(kind="average"), pop, dm[i], 1)[0] for i in range(0, 4000, 40)])
    assert abs(lamP.mean() - lam0.mean()) / lam0.mean() < 0.05
    # whole-drive adaptation lowers the mean
    specD = dict(kind="average", adapt_a=0.4, adapt_tau=1000.0, adapt_target="drive")
    lamD = np.stack([C.priming_rates(specD, pop, dm[i], 1)[0] for i in range(0, 4000, 40)])
    assert lamD.mean() < lam0.mean() * 0.95
    # conscious gain on tuning only: c multiplies the tuning part of the conscious face
    specG = dict(kind="average", cgain=1.21, cgain_target="tuning")
    lamG, _ = C.binocular_rates(rng, specG, pop, d[0], d[1], 1, 60.0, 1000.0)
    expect = pop.b + 0.5 * (pop.s + 1.21 * (d[0] - pop.s)) + 0.5 * d[1]
    assert np.allclose(lamG[0, 0], np.maximum(expect, 0.5))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print("all", len(tests), "tests passed")
