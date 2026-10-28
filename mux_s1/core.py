"""MUX-S1 core: synthetic face-patch populations, latent dynamics of the represented face,
spike generation, and a reimplementation of the analysis pipeline of Hesse et al. (2026).

Conventions
-----------
* Analysis bins are 10 ms; the analysis window is 300-1000 ms after stimulus onset (70 bins).
* m(t) in [0, 1] is the weight of the conscious (flashed) face C in the population rate;
  1 - m(t) is the weight of the suppressed (priming) face S.
* Rates are in Hz; counts per bin are Poisson(rate * DT).
"""
import numpy as np

DT = 0.010                      # s per analysis bin
NB = 70                         # bins in the 300-1000 ms window
T0 = 300                        # ms, start of the analysis window
SIM_MS = 1000                   # ms simulated after the flash
N_FEAT = 50
FEAT_VAR = np.concatenate([1.0 / np.arange(1, 21), 1.0 / np.arange(1, 31)])
SHAPE = np.arange(0, 20)
APP = np.arange(20, 50)


# ----------------------------------------------------------------------------------------
# faces and neurons
# ----------------------------------------------------------------------------------------
def make_faces(rng, n):
    return rng.standard_normal((n, N_FEAT)) * np.sqrt(FEAT_VAR)


class Population:
    """Linearly (axis-)tuned face cells: r = b + s + A * (w . f), clipped at 0.5 Hz."""

    def __init__(self, rng, n, kappa):
        self.n = n
        self.b = rng.lognormal(np.log(8.0), 0.4, n)           # spontaneous rate
        self.s = rng.lognormal(np.log(6.0), 0.5, n)           # mean face drive
        eta = np.concatenate([np.full(20, 0.5), np.full(30, 1.0)])
        W = rng.standard_normal((n, N_FEAT)) * np.sqrt(FEAT_VAR * eta)
        W /= np.sqrt((W ** 2 * FEAT_VAR).sum(axis=1, keepdims=True))   # SD over faces of W.f = 1
        self.W = W
        self.A = kappa * (self.b + self.s)

    def rates(self, F):
        return np.maximum(self.b + self.s + (F @ self.W.T) * self.A, 0.5)

    def delta(self, F):
        return self.rates(F) - self.b


# ----------------------------------------------------------------------------------------
# latent dynamics, simulated at 1 ms and averaged into 10 ms bins of the analysis window
# ----------------------------------------------------------------------------------------
def _bin_window(x):
    """x: (..., SIM_MS) at 1 ms -> (..., NB) mean over 10-ms bins of 300-1000 ms."""
    w = x[..., T0:T0 + NB * 10]
    return w.reshape(w.shape[:-1] + (NB, 10)).mean(axis=-1)


def dyn_telegraph(rng, n_tr, D):
    """Symmetric two-state Markov switching, mean dwell D ms (exponential)."""
    flips = rng.random((n_tr, SIM_MS)) < (1.0 / D)
    s0 = rng.random(n_tr) < 0.5
    state = (s0[:, None].astype(np.int8) + np.cumsum(flips, axis=1)) % 2
    return _bin_window(state.astype(np.float32))


def dyn_gamma(rng, n_tr, D, k):
    """Alternating renewal process, gamma(k, D/k) dwell times (k=1 -> Markov). Stationary start
    via a 3000 ms burn-in."""
    burn = 3000
    total = burn + SIM_MS
    n_int = int(total / D * 2.5) + 20
    dur = rng.gamma(k, D / k, size=(n_tr, n_int))
    edges = np.cumsum(dur, axis=1)
    if np.any(edges[:, -1] < total):
        raise RuntimeError("increase n_int")
    t = np.arange(burn, total) + 0.5
    idx = np.empty((n_tr, SIM_MS), dtype=np.int32)
    for i in range(n_tr):
        idx[i] = np.searchsorted(edges[i], t)
    s0 = rng.random(n_tr) < 0.5
    state = (idx + s0[:, None]) % 2
    return _bin_window(state.astype(np.float32))


def dyn_logit_ou(rng, n_tr, tau, sigma):
    """m = sigmoid(x), x an Ornstein-Uhlenbeck process (stationary SD sigma, time constant tau ms)."""
    a = np.exp(-1.0 / tau)
    q = sigma * np.sqrt(1 - a * a)
    x = np.empty((n_tr, SIM_MS), dtype=np.float32)
    x[:, 0] = rng.standard_normal(n_tr) * sigma
    noise = rng.standard_normal((n_tr, SIM_MS)).astype(np.float32) * q
    for t in range(1, SIM_MS):
        x[:, t] = a * x[:, t - 1] + noise[:, t]
    return _bin_window(1.0 / (1.0 + np.exp(-x)))


def dyn_ring(rng, n_tr, D_main, D_tr):
    """Cycle C -> X -> S -> Y -> C (exponential dwell; D_main in C,S and D_tr in X,Y).
    Returns bin fractions (n_tr, NB, 4) for states (C, X, S, Y)."""
    means = np.array([D_main, D_tr, D_main, D_tr])
    state = rng.integers(0, 4, n_tr)
    frac = np.zeros((n_tr, SIM_MS, 4), dtype=np.float32)
    for t in range(SIM_MS + 2000):
        if t >= 2000:
            frac[np.arange(n_tr), t - 2000, state] = 1.0
        move = rng.random(n_tr) < 1.0 / means[state]
        state = np.where(move, (state + 1) % 4, state)
    w = frac[:, T0:T0 + NB * 10, :]
    return w.reshape(n_tr, NB, 10, 4).mean(axis=2)


def adapt_gain(t_on_ms, a, tau_a):
    """Multiplicative adaptation of the stimulus drive, as a function of time since onset."""
    if a == 0:
        return np.ones_like(np.asarray(t_on_ms, dtype=float))
    return 1.0 - a * (1.0 - np.exp(-np.asarray(t_on_ms, dtype=float) / tau_a))


BIN_CENTERS = T0 + 10 * np.arange(NB) + 5.0     # ms after onset


# ----------------------------------------------------------------------------------------
# model specification
# ----------------------------------------------------------------------------------------
MODELS = {
    # name: dict(kind, params)
    "M0_markov": dict(kind="telegraph"),
    "M1_rhythm": dict(kind="gamma", k=6),
    "M1b_rhythm_k3": dict(kind="gamma", k=3),
    "M2_cont_unimodal": dict(kind="logit_ou", sigma=1.0),
    "M3_cont_bimodal": dict(kind="logit_ou", sigma=2.5),
    "M4_adaptation": dict(kind="telegraph", adapt_a=0.4, adapt_tau=1000.0, adapt_target="drive"),
    "M4c_adaptation_tuning": dict(kind="telegraph", adapt_a=0.4, adapt_tau=1000.0, adapt_target="tuning"),
    "M5_conscious_gain": dict(kind="telegraph", cgain=1.21, cgain_target="tuning"),
    "M6_heterogeneous": dict(kind="telegraph", fail_frac=0.2),
    "M7_ring": dict(kind="ring", D_tr=15.0),
    "C1_averaging": dict(kind="average"),
    "C2_uncoordinated": dict(kind="uncoordinated"),
}


def latent(rng, spec, n_tr, timescale):
    """Return m (n_tr, NB) or, for the ring, fractions (n_tr, NB, 4); and per-trial fail flags."""
    kind = spec["kind"]
    if kind == "telegraph":
        m = dyn_telegraph(rng, n_tr, timescale)
    elif kind == "gamma":
        m = dyn_gamma(rng, n_tr, timescale, spec["k"])
    elif kind == "logit_ou":
        m = dyn_logit_ou(rng, n_tr, timescale, spec["sigma"])
    elif kind == "ring":
        return dyn_ring(rng, n_tr, timescale, spec["D_tr"]), np.zeros(n_tr, bool)
    elif kind == "average":
        m = np.full((n_tr, NB), 0.5, dtype=np.float32)
    elif kind == "uncoordinated":
        m = None  # generated per neuron in binocular_rates
    else:
        raise ValueError(kind)
    fail = np.zeros(n_tr, bool)
    if spec.get("fail_frac", 0) > 0:
        fail = rng.random(n_tr) < spec["fail_frac"]
        m[fail] = 0.0
    return m, fail


def _terms(spec, pop, d, g):
    """Drive above baseline of one face with adaptation gain g (array broadcastable to (.., NB, 1)).
    adapt_target 'drive' scales the whole drive; 'tuning' scales only the face-specific tuning part
    (drive minus the cell's mean face response s)."""
    if spec.get("adapt_target", "drive") == "tuning":
        return pop.s[None, None, :] + g * (d - pop.s)[None, None, :]
    return g * d[None, None, :]


def binocular_rates(rng, spec, pop, dC, dS, n_tr, timescale, t_prime_ms, ring_u=None):
    """Rates (n_tr, NB, N) for binocular trials with conscious face drive dC and suppressed dS
    (each (N,), drive above baseline). t_prime_ms: priming duration before the flash."""
    a = spec.get("adapt_a", 0.0)
    ta = spec.get("adapt_tau", 1000.0)
    gC = adapt_gain(BIN_CENTERS, a, ta)[None, :, None]
    gS = adapt_gain(BIN_CENTERS + t_prime_ms, a, ta)[None, :, None]
    c = spec.get("cgain", 1.0)
    b = pop.b[None, None, :]
    termC = _terms(spec, pop, dC, gC)
    termS = _terms(spec, pop, dS, gS)
    if c != 1.0:
        if spec.get("cgain_target", "drive") == "tuning":
            termC = pop.s[None, None, :] + c * (termC - pop.s[None, None, :])
        else:
            termC = c * termC
    if spec["kind"] == "uncoordinated":
        p = 1.0 - np.exp(-10.0 / timescale)
        st = np.empty((n_tr, NB, pop.n), dtype=np.float32)
        cur = rng.random((n_tr, pop.n)) < 0.5
        for t in range(NB):
            st[:, t, :] = cur
            cur = cur ^ (rng.random((n_tr, pop.n)) < p)
        return np.maximum(b + st * termC + (1 - st) * termS, 0.5), None
    m, fail = latent(rng, spec, n_tr, timescale)
    if spec["kind"] == "ring":
        f = m
        mid = 0.5 * (dC + dS)
        lam = (b + f[..., 0:1] * termC + f[..., 2:3] * termS
               + (f[..., 1:2] + f[..., 3:4]) * mid + (f[..., 1:2] - f[..., 3:4]) * ring_u)
        return np.maximum(lam, 0.5), f
    mm = m[..., None]
    return np.maximum(b + mm * termC + (1 - mm) * termS, 0.5), m


def priming_rates(spec, pop, dP, n_tr):
    a = spec.get("adapt_a", 0.0)
    ta = spec.get("adapt_tau", 1000.0)
    g = adapt_gain(BIN_CENTERS, a, ta)[None, :, None]
    lam = np.maximum(pop.b[None, None, :] + _terms(spec, pop, dP, g), 0.5)
    return np.broadcast_to(lam, (n_tr, NB, pop.n))


def unambiguous_rates(spec, pop, dU, n_tr):
    return priming_rates(spec, pop, dU, n_tr)


def poisson(rng, lam):
    return rng.poisson(lam * DT).astype(np.int16)


# ----------------------------------------------------------------------------------------
# analysis pipeline
# ----------------------------------------------------------------------------------------
def llr(counts, lamC, lamS):
    """Population Poisson log-likelihood ratio per bin (Hesse et al. eq.). counts (..., N)."""
    w = np.log(lamC / lamS)
    return counts @ w - (lamC - lamS).sum() * DT


def emis_loglik(counts, lam_list):
    """log P(counts | state k) up to a constant shared by all states. Returns (..., K)."""
    L = np.stack([np.log(l * DT) for l in lam_list], axis=1)      # (N, K)
    tot = np.array([l.sum() * DT for l in lam_list])
    return counts @ L - tot


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def forward_loglik(E, logT, logpi):
    """E: (n_tr, T, K) emission log-likelihoods. Returns per-trial log-likelihood."""
    a = logpi[None, :] + E[:, 0, :]
    for t in range(1, E.shape[1]):
        a = _logsumexp(a[:, :, None] + logT[None, :, :], axis=1) + E[:, t, :]
    return _logsumexp(a, axis=1)


def hmm_fit_T(E, n_iter=60, tol=1e-6, T_init=None):
    """Baum-Welch for the 2x2 transition matrix with fixed emissions and pi=(.5,.5).
    E: (n_tr, T, 2). Returns T (2x2)."""
    n_tr, Tn, K = E.shape
    T = np.array([[0.85, 0.15], [0.15, 0.85]]) if T_init is None else T_init.copy()
    logpi = np.log(np.full(K, 1.0 / K))
    prev = -np.inf
    for _ in range(n_iter):
        logT = np.log(T)
        la = np.empty((n_tr, Tn, K))
        la[:, 0] = logpi + E[:, 0]
        for t in range(1, Tn):
            la[:, t] = _logsumexp(la[:, t - 1][:, :, None] + logT[None], axis=1) + E[:, t]
        lb = np.zeros((n_tr, Tn, K))
        for t in range(Tn - 2, -1, -1):
            lb[:, t] = _logsumexp(logT[None] + (E[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        ll = _logsumexp(la[:, -1], axis=1)
        # expected transitions
        xi = (la[:, :-1, :, None] + logT[None, None] + (E[:, 1:] + lb[:, 1:])[:, :, None, :]
              - ll[:, None, None, None])
        cnt = np.exp(xi).sum(axis=(0, 1))
        T = cnt / cnt.sum(axis=1, keepdims=True)
        T = np.clip(T, 1e-6, 1 - 1e-6)
        T /= T.sum(axis=1, keepdims=True)
        tot = ll.sum()
        if tot - prev < tol * abs(tot):
            break
        prev = tot
    return T


def _scaled_e(E):
    """exp of emission log-likelihoods, rescaled per (.., t) so the max over states is 1."""
    return np.exp(E - E.max(axis=-1, keepdims=True))


def hmm_fit_batch(E, n_iter=40, tol=1e-5):
    """Batched Baum-Welch for 2-state HMMs with fixed emissions and pi=(.5,.5).
    E: (P, M, T, 2) emission log-likelihoods for P independent HMMs (one per stimulus pair),
    each with M sequences of length T. Returns T matrices (P, 2, 2)."""
    P, M, Tn, K = E.shape
    e = _scaled_e(E)
    Tm = np.tile(np.array([[0.85, 0.15], [0.15, 0.85]]), (P, 1, 1))
    for _ in range(n_iter):
        al = np.empty_like(e)
        cs = np.empty((P, M, Tn))
        a = 0.5 * e[:, :, 0, :]
        c = a.sum(-1); a = a / c[..., None]
        al[:, :, 0] = a; cs[:, :, 0] = c
        for t in range(1, Tn):
            a = np.matmul(a[:, :, None, :], Tm[:, None])[:, :, 0, :] * e[:, :, t, :]
            c = a.sum(-1); a = a / c[..., None]
            al[:, :, t] = a; cs[:, :, t] = c
        be = np.empty_like(e)
        b = np.ones((P, M, K))
        be[:, :, -1] = b
        for t in range(Tn - 2, -1, -1):
            b = np.matmul(Tm[:, None], (e[:, :, t + 1, :] * b)[..., None])[..., 0] / cs[:, :, t + 1, None]
            be[:, :, t] = b
        # xi summed over sequences and time
        w = e[:, :, 1:, :] * be[:, :, 1:, :] / cs[:, :, 1:, None]          # (P,M,T-1,K)
        cnt = np.einsum("pmti,pmtj->pij", al[:, :, :-1, :], w) * Tm
        newT = cnt / cnt.sum(axis=2, keepdims=True)
        newT = np.clip(newT, 1e-6, 1 - 1e-6)
        newT /= newT.sum(axis=2, keepdims=True)
        delta = np.abs(newT - Tm).max()
        Tm = newT
        if delta < tol:
            break
    return Tm


def viterbi_batch(E, Tm):
    """E (P, M, T, 2), Tm (P, 2, 2) -> most likely paths (P, M, T) int8."""
    P, M, Tn, K = E.shape
    lT = np.log(Tm)
    d = np.log(0.5) + E[:, :, 0, :]
    bp = np.zeros((P, M, Tn, K), dtype=np.int8)
    for t in range(1, Tn):
        c0 = np.stack([d[..., 0] + lT[:, None, 0, 0], d[..., 1] + lT[:, None, 1, 0]], -1)
        c1 = np.stack([d[..., 0] + lT[:, None, 0, 1], d[..., 1] + lT[:, None, 1, 1]], -1)
        bp[:, :, t, 0] = np.argmax(c0, -1)
        bp[:, :, t, 1] = np.argmax(c1, -1)
        d = np.stack([c0.max(-1), c1.max(-1)], -1) + E[:, :, t, :]
    z = np.empty((P, M, Tn), dtype=np.int8)
    z[:, :, -1] = np.argmax(d, -1)
    for t in range(Tn - 2, -1, -1):
        z[:, :, t] = np.take_along_axis(bp[:, :, t + 1, :], z[:, :, t + 1][..., None].astype(np.int64), -1)[..., 0]
    return z


def forward_loglik_scaled(E, Kmat, pi):
    """Forward algorithm in scaled probability space. E (n, T, K) emission log-likelihoods,
    Kmat (K, K) transition probabilities, pi (K,). Returns per-sequence log-likelihoods."""
    off = E.max(axis=-1, keepdims=True)
    e = np.exp(E - off)
    a = pi[None, :] * e[:, 0, :]
    c = a.sum(-1)
    ll = np.log(c) + off[:, 0, 0]
    a = a / c[:, None]
    for t in range(1, E.shape[1]):
        a = (a @ Kmat) * e[:, t, :]
        c = a.sum(-1)
        ll += np.log(c) + off[:, t, 0]
        a = a / c[:, None]
    return ll


def viterbi(E, T):
    n_tr, Tn, K = E.shape
    logT = np.log(T)
    d = np.log(0.5) + E[:, 0]
    bp = np.empty((n_tr, Tn, K), dtype=np.int8)
    for t in range(1, Tn):
        cand = d[:, :, None] + logT[None]
        bp[:, t] = np.argmax(cand, axis=1)
        d = np.max(cand, axis=1) + E[:, t]
    z = np.empty((n_tr, Tn), dtype=np.int8)
    z[:, -1] = np.argmax(d, axis=1)
    for t in range(Tn - 2, -1, -1):
        z[:, t] = bp[np.arange(n_tr), t + 1, z[:, t + 1]]
    return z


def run_lengths(z, exclude_edges=False):
    """Run lengths (in bins) of each state in each row of z. Returns dict state->list."""
    out = {0: [], 1: []}
    for row in z:
        ch = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate([[0], ch])
        ends = np.concatenate([ch, [len(row)]])
        for k, (s, e) in enumerate(zip(starts, ends)):
            if exclude_edges and (k == 0 or k == len(starts) - 1):
                continue
            out[int(row[s])].append(e - s)
    return out


def acf_per_trial_mean(x, max_lag):
    """x: (n_tr, T). Per-trial mean subtraction, products pooled over trials, normalized by lag 0."""
    xc = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    out = np.empty(max_lag + 1)
    for L in range(max_lag + 1):
        out[L] = (xc[:, :T - L] * xc[:, L:]).sum() / (x.shape[0] * (T - L))
    return out / out[0]


def acf_global_mean(x, max_lag):
    xc = x - x.mean()
    T = x.shape[1]
    out = np.empty(max_lag + 1)
    for L in range(max_lag + 1):
        out[L] = (xc[:, :T - L] * xc[:, L:]).sum() / (x.shape[0] * (T - L))
    return out / out[0]


def fit_D(acf_bins):
    """Hesse et al.: fit rho*exp(-2*tau/D) to the 1-ms-resolution ACF over 10 ms .. first zero
    crossing (the ACF of a signal constant within 10-ms bins is linear between bin lags).
    acf_bins: ACF at lags 0,10,20,... ms. Returns (D, rho, zero_crossing_ms)."""
    lags_b = 10 * np.arange(len(acf_bins))
    lags = np.arange(0, lags_b[-1] + 1)
    acf1 = np.interp(lags, lags_b, acf_bins)
    neg = np.flatnonzero(acf1[10:] <= 0)
    zc = int(lags[10 + neg[0]]) if len(neg) else int(lags[-1])
    sel = (lags >= 10) & (lags <= zc)
    tt, yy = lags[sel].astype(float), acf1[sel]
    best = (np.inf, None, None)
    for D in np.arange(10.0, 400.0, 0.5):
        g = np.exp(-2 * tt / D)
        rho = (g @ yy) / (g @ g)
        sse = ((yy - rho * g) ** 2).sum()
        if sse < best[0]:
            best = (sse, D, rho)
    return best[1], best[2], zc


def tls_slope(x, y):
    """Total least squares slope through the means."""
    xc, yc = x - x.mean(), y - y.mean()
    C = np.cov(np.vstack([xc, yc]))
    w, v = np.linalg.eigh(C)
    vec = v[:, np.argmax(w)]
    return vec[1] / vec[0]


def explained_variance(Y, Yhat):
    """Per-feature EV across faces (rows)."""
    return 1 - ((Y - Yhat) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def aggregate_ev(ev):
    """Mean of the variance-weighted shape EV and appearance EV."""
    sh = (ev[SHAPE] * FEAT_VAR[SHAPE]).sum() / FEAT_VAR[SHAPE].sum()
    ap = (ev[APP] * FEAT_VAR[APP]).sum() / FEAT_VAR[APP].sum()
    return 0.5 * (sh + ap)


class Decoder:
    """PCA (K comps, fit on priming responses) + OLS to features, with leave-one-face-out
    weights for selected faces via Sherman-Morrison."""

    def __init__(self, P, Y, K=500):
        self.mu = P.mean(axis=0)
        Pc = P - self.mu
        U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
        K = min(K, Vt.shape[0])
        self.V = Vt[:K].T
        X = np.hstack([Pc @ self.V, np.ones((P.shape[0], 1))])
        self.X = X
        self.Y = Y
        A = X.T @ X
        self.Ainv = np.linalg.pinv(A)
        self.XtY = X.T @ Y

    def weights_lofo(self, j):
        x = self.X[j][:, None]
        Ai = self.Ainv
        Aix = Ai @ x
        denom = 1.0 - (x.T @ Aix)[0, 0]
        Ai_j = Ai + (Aix @ Aix.T) / denom
        return Ai_j @ (self.XtY - x @ self.Y[j][None, :])

    def predict(self, r, B):
        z = np.hstack([(np.atleast_2d(r) - self.mu) @ self.V, np.ones((np.atleast_2d(r).shape[0], 1))])
        return z @ B


def lofo_linear_predictions(F, R):
    """Leave-one-face-out OLS predictions of each neuron's response from face features.
    F (n_faces, 50), R (n_faces, N). Returns (n_faces, N) and cross-validated R^2 per neuron."""
    X = np.hstack([F, np.ones((F.shape[0], 1))])
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    h = np.diag(H)[:, None]
    yhat = H @ R
    loo = (yhat - h * R) / (1 - h)
    r2 = 1 - ((R - loo) ** 2).sum(0) / ((R - R.mean(0)) ** 2).sum(0)
    return loo, r2


def ols_slope(x, y):
    xc = x - x.mean()
    return (xc @ (y - y.mean())) / (xc @ xc)
