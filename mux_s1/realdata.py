"""MUX-S1 version 1.1: the checks run on the released data of Hesse et al. (2026).

The code below takes a canonical form of the data and nothing else, so that the analysis can be
frozen and hashed before the release while the reader of the DANDI files (`mux_s1/adapters.py`)
stays outside the frozen set: the file format becomes visible only on release day, the statistics
do not.

Canonical form
--------------
`Pair` holds the spike counts of one stimulus pair, binned at `bin_ms`:

    bino_a   (n_trials, T, N) int counts, binocular trials where face a was flashed (conscious = a)
    bino_b   (n_trials, T, N) the same with face b flashed
    prime_a  (n_trials, T, N) priming trials of face a alone
    prime_b  (n_trials, T, N) priming trials of face b alone
    unamb_a, unamb_b  optional unambiguous-context trials
    face_a, face_b    indices into `Session.faces` when a feature space is available

`Session` holds the pairs recorded together (the same neurons), the bin width and, for the
1500-face experiment, the face feature matrix. The analysis window is whatever the adapter put in
the counts; the pre-registered window is 300-1000 ms after the flash.

Every statistic is one of the checks A1-A4 (calibration: does the released data match what we read
off the figures) or one of the eleven predictions of MUX-S1 1.0 about the released data, numbered
B1-B12 with B8 unused (its statistic separated nothing in the simulation). The thresholds
are module constants, fixed before the release together with the reference values measured on the
simulated models of version 1.0.
"""
from dataclasses import dataclass, field

import numpy as np

from . import core as C

PSEUDO = 0.5                     # pseudo-count (spikes) in a template, as in version 1.0
BIN_WIDTHS_MS = (5, 10, 20, 50)  # widths for the dwell-vs-bin-width curve (Fig. 5c)
LABEL_SIZES = (10, 25, 50, 100, None)   # neurons used for the state labels in B4 (None = half)
M_GRID = np.linspace(0.0, 1.0, 41)      # static mixtures in the likelihood tests

# ---- digitized targets of the preprint (digitize/targets.json, version 1.0) -------------------
PUB = dict(priming_state1=0.229, acf10=(0.35, 0.39), bino_state1=0.47, acf_D_ms=67.0,
           dwell_median_5ms_ms=59.0, dwell_by_binwidth={5: 59.6, 10: 62.1, 20: 77.1, 50: 110.8},
           gain_ratio=1.21, gain_r=0.90, decoder_acc=0.66, acf_tail=-0.058)

# ---- reference values from the simulated models of version 1.0 (results/summary.json) ---------
REF = dict(M0_acf10=0.224, M0_D=66.5, M0_D_global=112.6, M0_dwell_median_ms=91.0,
           M0_dwell_cv=0.811, M0_overdispersion=1.089, M6_overdispersion=1.267,
           M1_dwell_cv=0.623, M1b_dwell_cv=0.696, M2_dwell_cv=0.930, M3_dwell_cv=0.965,
           M0_ratio_dwell_over_D=1.37, M0_ratio_dwell_over_D_face1500=1.80,
           M4_decoder=0.764, M4c_decoder=0.484, M4_ratio_early_minus_late=0.071,
           M5_ratio_early_minus_late=0.007, M4_frac_s2_higher=0.272, M5_frac_s2_higher=0.495,
           # the decoder without the per-state mean rate, measured on the models of version 1.0
           # before the release (reduced design: 2 sessions, 40 pairs, 10 trials, 300 neurons)
           decoder_no_mean=dict(M0=0.516, M4=0.518, M5=0.508, M4c=0.527),
           decoder_raw_reduced=dict(M0=0.531, M4=0.773, M5=0.532, M4c=0.502),
           # the Fig. 3a test in the equal-noise version separates switching from mixing, which
           # the authors' version does not (measured before the release, 6-8 sessions each)
           fig3a_equal=dict(M0=0.0043, C1=-0.0132, M3=-0.0049),
           fig3a_unequal=dict(M0=0.0553, C1=0.0361, M3=0.0457))

# ---- pre-registered thresholds ---------------------------------------------------------------
THR = dict(
    A1_tol=0.030,              # |priming state-1 fraction - 0.229|
    A2_range=(0.32, 0.42),     # ACF at the first lag
    A3_range=(0.42, 0.55),     # binocular state-1 fraction
    A4_max_rel_dev=0.25,       # dwell-vs-bin-width curve against Fig. 5c
    B1_max=1.15,               # median dwell (10 ms bins) / D of the ACF fit
    B1_refuted_min=1.35,
    B2_min=1.40,               # D(global mean) / D(per-trial mean)
    B3_min=1.25,               # ACF at 10 ms, relative to the calibrated M0 value
    B4_small_max=0.0,          # two-rate advantage with 10 label neurons
    B5_max_ratio=0.50,         # equal-noise advantage / unequal-noise advantage (Fig. 3a)
    B6_max=0.56,               # state decoder after subtracting the per-state mean rate
    B6_refuted_min=0.62,       # above it the decoder reads more than the mean-rate shift
    B7_min=0.05,               # gain ratio, early window minus late window
    B9_frac_min=0.55,          # cells with the higher mean rate in state 2
    B9_ratio_min=1.10,         # gain ratio alongside it
    B10_rho_min=0.30,          # Spearman rho between the gain and the coding R^2
    B10_drop_min=0.15,         # drop of r(gain S1C, gain S2S) inside a narrow R^2 band
    B11_markov=(0.75, 0.90),   # dwell CV: Markov band
    B11_rhythm_max=0.72,       # below it: rhythmic
    B11_continuous_min=0.92,   # above it: continuous mixing
    B12_max=1.15,              # trial overdispersion without flash failures
    B12_hetero_min=1.20,
    SIGNAL_MIN=0.20,           # ACF at the first lag below which B1, B5 and B11 are descriptive
)

# B1, B5 and B11 read the state labels, so a weak state signal moves them on its own: measured
# before the release on M4 at a reduced scale (ACF first lag 0.12), where B5 turns negative and
# the dwell CV rises to 0.99 although the latent dynamics are Markov switching. They are
# interpreted only when the signal passes THR["SIGNAL_MIN"], and reported descriptively otherwise.
SIGNAL_GATED = ("B1_dwell_over_D", "B5_fig3a_equal_noise", "B11_dwell_cv")


# ----------------------------------------------------------------------------------------------
# canonical containers
# ----------------------------------------------------------------------------------------------
@dataclass
class Pair:
    bino_a: np.ndarray
    bino_b: np.ndarray
    prime_a: np.ndarray
    prime_b: np.ndarray
    unamb_a: np.ndarray = None
    unamb_b: np.ndarray = None
    face_a: int = -1
    face_b: int = -1


@dataclass
class Session:
    pairs: list
    bin_ms: float = 10.0
    name: str = ""
    faces: np.ndarray = None                 # (n_faces, n_feat) or None
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self):
        return self.pairs[0].bino_a.shape[2]

    @property
    def n_bins(self):
        return self.pairs[0].bino_a.shape[1]


def check_sessions(sessions):
    """Structural validation of the canonical form; raises on anything the analysis relies on."""
    for s in sessions:
        if not s.pairs:
            raise ValueError(f"session {s.name}: no pairs")
        N, T = s.n_neurons, s.n_bins
        for i, p in enumerate(s.pairs):
            for nm in ("bino_a", "bino_b", "prime_a", "prime_b"):
                x = getattr(p, nm)
                if x is None or x.ndim != 3:
                    raise ValueError(f"session {s.name} pair {i}: {nm} must be (n_tr, T, N)")
                if x.shape[1] != T or x.shape[2] != N:
                    raise ValueError(f"session {s.name} pair {i}: {nm} has shape {x.shape}, expected (*, {T}, {N})")
                if x.min() < 0:
                    raise ValueError(f"session {s.name} pair {i}: {nm} has negative counts")
        if s.faces is not None:
            n_f = s.faces.shape[0]
            for p in s.pairs:
                if not (0 <= p.face_a < n_f and 0 <= p.face_b < n_f):
                    raise ValueError(f"session {s.name}: face index outside the feature matrix")
    return True


# ----------------------------------------------------------------------------------------------
# bin-width aware primitives (version 1.0 fixes DT = 10 ms inside core)
# ----------------------------------------------------------------------------------------------
def rebin(x, k):
    """Sum neighbouring bins in groups of k; a tail shorter than k is dropped."""
    if k == 1:
        return x
    T = (x.shape[1] // k) * k
    return x[:, :T].reshape(x.shape[0], T // k, k, x.shape[2]).sum(axis=2)


def template(counts, dt):
    n_tr, T = counts.shape[0], counts.shape[1]
    return (counts.sum(axis=(0, 1)) + PSEUDO) / (n_tr * T * dt)


def llr(counts, lamC, lamS, dt):
    return counts @ np.log(lamC / lamS) - (lamC - lamS).sum() * dt


def emis(counts, lams, dt):
    L = np.stack([np.log(l * dt) for l in lams], axis=1)
    tot = np.array([l.sum() * dt for l in lams])
    return counts @ L - tot


def fit_D_ms(acf_bins, bin_ms):
    """The authors' fit of rho*exp(-2*tau/D) from the first lag to the first zero crossing,
    on a 1-ms interpolation of the binned ACF (version 1.0, generalized to any bin width)."""
    lags_b = bin_ms * np.arange(len(acf_bins))
    lags = np.arange(0, int(lags_b[-1]) + 1)
    acf1 = np.interp(lags, lags_b, acf_bins)
    start = int(bin_ms)
    neg = np.flatnonzero(acf1[start:] <= 0)
    zc = int(lags[start + neg[0]]) if len(neg) else int(lags[-1])
    sel = (lags >= start) & (lags <= zc)
    tt, yy = lags[sel].astype(float), acf1[sel]
    best = (np.inf, None, None)
    for D in np.arange(10.0, 400.0, 0.5):
        g = np.exp(-2 * tt / D)
        rho = (g @ g) and (g @ yy) / (g @ g)
        sse = ((yy - rho * g) ** 2).sum()
        if sse < best[0]:
            best = (sse, D, rho)
    return best[1], best[2], zc


# ----------------------------------------------------------------------------------------------
# per-session pipeline: templates, LLR, HMM states
# ----------------------------------------------------------------------------------------------
def pair_templates(p, dt):
    """Templates of the two faces from their priming trials."""
    return template(p.prime_a, dt), template(p.prime_b, dt)


def session_llr(sess, bin_ms=None, rng=None):
    """LLR of binocular and priming trials, in the convention of version 1.0: the LLR is signed
    towards the conscious (flashed) face, so a positive bin is 'state 1'. Priming trials of the
    suppressed face are scored against the conscious template (the false state-1 rate of Fig. 2c).
    """
    k = 1 if bin_ms is None else int(round(bin_ms / sess.bin_ms))
    dt = sess.bin_ms * k / 1000.0
    out = dict(bino=[], prime=[], occ_bino=[], occ_prime=[])
    for p in sess.pairs:
        ba, bb = rebin(p.bino_a, k), rebin(p.bino_b, k)
        pa, pb = rebin(p.prime_a, k), rebin(p.prime_b, k)
        lamA, lamB = template(pa, dt), template(pb, dt)
        lb = np.concatenate([llr(ba, lamA, lamB, dt), llr(bb, lamB, lamA, dt)])
        lp = np.concatenate([llr(pb, lamA, lamB, dt), llr(pa, lamB, lamA, dt)])
        out["bino"].append(lb)
        out["prime"].append(lp)
        out["occ_bino"].append(float((lb > 0).mean()))
        out["occ_prime"].append(float((lp > 0).mean()))
    return out


def session_states(sess, bin_ms=None, halves=None, label_idx=None):
    """Two-state HMM per pair with fixed emissions from the priming templates, Baum-Welch
    transitions and a Viterbi path (version 1.0). Returns per pair the conscious-state indicator
    of every binocular trial, as (n_tr_total, T) with trials of face a first.

    label_idx: neurons the states are read from (None = all); the labels are returned together
    with the held-out neurons so the likelihood tests can use disjoint sets.
    """
    k = 1 if bin_ms is None else int(round(bin_ms / sess.bin_ms))
    dt = sess.bin_ms * k / 1000.0
    groups, data = [], []
    for p in sess.pairs:
        ba, bb = rebin(p.bino_a, k), rebin(p.bino_b, k)
        pa, pb = rebin(p.prime_a, k), rebin(p.prime_b, k)
        lamA, lamB = template(pa, dt), template(pb, dt)
        bino = np.concatenate([ba, bb])
        idx = np.arange(bino.shape[2]) if label_idx is None else np.asarray(label_idx)
        groups.append(emis(bino[..., idx], [lamA[idx], lamB[idx]], dt))
        data.append((bino, lamA, lamB, ba.shape[0], bb.shape[0]))
    states = []
    by_shape = {}
    for i, g in enumerate(groups):
        by_shape.setdefault(g.shape[:2], []).append(i)
    for shape, idxs in by_shape.items():
        E = np.stack([groups[i] for i in idxs])
        Tm = C.hmm_fit_batch(E)
        z = C.viterbi_batch(E, Tm)
        for n_, i in enumerate(idxs):
            states.append((i, z[n_]))
    states.sort()
    out = []
    for (i, z), (bino, lamA, lamB, na, nb) in zip(states, data):
        cond = np.concatenate([np.zeros(na, int), np.ones(nb, int)])   # 0: face a flashed
        c = np.where(cond[:, None] == 0, z == 0, z == 1)               # conscious-state indicator
        out.append(dict(c=c, bino=bino, lamA=lamA, lamB=lamB, na=na, nb=nb, dt=dt))
    return out


# ----------------------------------------------------------------------------------------------
# A. calibration checks
# ----------------------------------------------------------------------------------------------
def dwell_stats(states_per_session):
    """Median dwell per state and the CV with the trial edges excluded, pooled over sessions."""
    runs = {0: [], 1: []}
    ex = {0: [], 1: []}
    for sess in states_per_session:
        for pr in sess:
            z = pr["c"].astype(np.int8)
            dt_ms = pr["dt"] * 1000.0
            r = C.run_lengths(z)
            rx = C.run_lengths(z, exclude_edges=True)
            for s in (0, 1):
                runs[s] += [v * dt_ms for v in r[s]]
                ex[s] += [v * dt_ms for v in rx[s]]
    allx = np.array(ex[0] + ex[1], float)
    return dict(median_state1_ms=float(np.median(runs[1])) if runs[1] else None,
                median_state2_ms=float(np.median(runs[0])) if runs[0] else None,
                median_ms=float(np.median(np.array(runs[0] + runs[1], float))),
                cv_edges_excluded=float(allx.std() / allx.mean()) if allx.size else None,
                n_runs_edges_excluded=int(allx.size))


def acf_stats(llr_lists, bin_ms):
    """ACF with both estimators (per-trial mean E-a, global mean E-b), averaged over sessions."""
    max_lag = int(round(200.0 / bin_ms))
    ea = np.mean([C.acf_per_trial_mean(np.concatenate(x), max_lag) for x in llr_lists], axis=0)
    eb = np.mean([C.acf_global_mean(np.concatenate(x), max_lag) for x in llr_lists], axis=0)
    Da, rho_a, zc = fit_D_ms(ea, bin_ms)
    Db, _, _ = fit_D_ms(eb, bin_ms)
    lags = bin_ms * np.arange(max_lag + 1)
    tail = (lags >= 110) & (lags <= 200)
    return dict(acf_a=ea.tolist(), acf_b=eb.tolist(), lags_ms=lags.tolist(), D_a_ms=float(Da),
                D_b_ms=float(Db), rho_a=float(rho_a), zero_cross_ms=int(zc),
                acf_first_lag=float(ea[1]), mean_110_200=float(ea[tail].mean()))


def check_A(sessions, rng=None):
    """A1 priming false-state fraction, A2 ACF at the first lag, A3 binocular state-1 fraction,
    A4 the dwell-vs-bin-width curve against Fig. 5c."""
    per_sess = [session_llr(s, bin_ms=10.0) for s in sessions]
    occ_p = float(np.mean([np.mean(x["occ_prime"]) for x in per_sess]))
    occ_b = float(np.mean([np.mean(x["occ_bino"]) for x in per_sess]))
    acf = acf_stats([x["bino"] for x in per_sess], 10.0)
    curve, dev = {}, []
    for w in BIN_WIDTHS_MS:
        if (w / sessions[0].bin_ms) % 1 != 0:
            continue
        st = [session_states(s, bin_ms=float(w)) for s in sessions]
        m = dwell_stats(st)["median_ms"]
        curve[w] = m
        if w in PUB["dwell_by_binwidth"]:
            dev.append(abs(m - PUB["dwell_by_binwidth"][w]) / PUB["dwell_by_binwidth"][w])
    out = {
        "A1_priming_state1": dict(value=occ_p, target=PUB["priming_state1"],
                                  passed=abs(occ_p - PUB["priming_state1"]) <= THR["A1_tol"]),
        "A2_acf_first_lag": dict(value=acf["acf_first_lag"], target=PUB["acf10"],
                                 passed=THR["A2_range"][0] <= acf["acf_first_lag"] <= THR["A2_range"][1]),
        "A3_bino_state1": dict(value=occ_b, target=PUB["bino_state1"],
                               passed=THR["A3_range"][0] <= occ_b <= THR["A3_range"][1]),
        "A4_dwell_by_binwidth": dict(value=curve, target=PUB["dwell_by_binwidth"],
                                     max_rel_dev=(float(max(dev)) if dev else None),
                                     passed=(bool(max(dev) <= THR["A4_max_rel_dev"]) if dev else None)),
    }
    return out, acf, per_sess


# ----------------------------------------------------------------------------------------------
# B. the predictions of version 1.0
# ----------------------------------------------------------------------------------------------
def check_B1_B3(acf, dwell10):
    """B1 the two time numbers, B2 the estimator, B3 the strength of the state signal."""
    R1 = dwell10["median_ms"] / acf["D_a_ms"]
    R2 = acf["D_b_ms"] / acf["D_a_ms"]
    R3 = acf["acf_first_lag"] / REF["M0_acf10"]
    return {
        "B1_dwell_over_D": dict(value=float(R1), threshold=THR["B1_max"], reference_M0=REF["M0_ratio_dwell_over_D"],
                                passed=bool(R1 <= THR["B1_max"]),
                                refuted=bool(R1 >= THR["B1_refuted_min"])),
        "B2_D_global_over_D_trial": dict(value=float(R2), threshold=THR["B2_min"],
                                         reference_M0=REF["M0_D_global"] / REF["M0_D"],
                                         passed=bool(R2 >= THR["B2_min"])),
        "B3_acf_over_calibration": dict(value=float(R3), threshold=THR["B3_min"],
                                        acf_first_lag=acf["acf_first_lag"], reference_M0=REF["M0_acf10"],
                                        passed=bool(R3 >= THR["B3_min"])),
    }


def two_rate_advantage(sessions, label_sizes=LABEL_SIZES, seed=0):
    """B4. The advantage of two state-conditioned rates over the best static mixture, on neurons
    held out from the ones that produced the state labels, as a function of how many neurons the
    labels came from. Units: nats per bin per neuron (version 1.0 convention)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n_lab in label_sizes:
        vals = []
        for s in sessions:
            N = s.n_neurons
            perm = rng.permutation(N)
            half = N // 2
            pool, held = np.sort(perm[:half]), np.sort(perm[half:])
            idx = pool if n_lab is None else np.sort(rng.choice(pool, min(n_lab, half), replace=False))
            st = session_states(s, bin_ms=10.0, label_idx=idx)
            for pr in st:
                n = pr["bino"][..., held].astype(float)
                lc, ls = pr["lamA"][held] * pr["dt"], pr["lamB"][held] * pr["dt"]
                cond_a = np.arange(n.shape[0]) < pr["na"]
                lamC = np.where(cond_a[:, None, None], lc, ls)
                lamS = np.where(cond_a[:, None, None], ls, lc)
                lam2 = np.where(pr["c"][..., None], lamC, lamS)
                ll2 = (n * np.log(lam2) - lam2).sum(axis=(0, 1))
                nb = n.shape[0] * n.shape[1]
                nsum = n.sum(axis=(0, 1))
                best = None
                for m in M_GRID:
                    lamp = (m * lamC + (1 - m) * lamS).mean(axis=(0, 1))
                    llp = nsum * np.log(lamp) - lamp * nb
                    if best is None or llp.sum() > best.sum():
                        best = llp
                vals.append(float(((ll2 - best) / nb).mean()))
        out["half" if n_lab is None else str(n_lab)] = float(np.mean(vals))
    ks = [k for k in out if k != "half"]
    ks.sort(key=int)
    seq = [out[k] for k in ks] + [out["half"]]
    return dict(by_n_labels=out, monotone=bool(all(b >= a - 1e-6 for a, b in zip(seq, seq[1:]))),
                small_n=out[ks[0]], full=out["half"],
                passed=bool(out["half"] > 0 and out[ks[0]] <= THR["B4_small_max"]))


def fig3a_equal_noise(sessions, seed=0):
    """B5. The Fig. 3a test in two versions: the authors' (a static rate estimated from the other
    half of the binocular trials, which carries more estimation noise than the two templates) and
    an equal-noise version (the static rate is the half-and-half mixture of the same two templates).
    """
    rng = np.random.default_rng(seed)
    uneq, eq = [], []
    for s in sessions:
        k = int(round(10.0 / s.bin_ms))
        dt = s.bin_ms * k / 1000.0
        for p in s.pairs:
            pa, pb = rebin(p.prime_a, k), rebin(p.prime_b, k)
            lamA, lamB = template(pa, dt), template(pb, dt)
            for bino, lC, lS in ((rebin(p.bino_a, k), lamA, lamB), (rebin(p.bino_b, k), lamB, lamA)):
                n_tr = bino.shape[0]
                if n_tr < 4:
                    continue
                perm = rng.permutation(n_tr)
                h1, h2 = perm[: n_tr // 2], perm[n_tr // 2:]
                eC = emis(bino, [lC], dt)[..., 0]
                eS = emis(bino, [lS], dt)[..., 0]
                mix = np.logaddexp(eC, eS) + np.log(0.5)
                for a, b in ((h1, h2), (h2, h1)):
                    static_tr = np.maximum(template(bino[b], dt), 1e-3)
                    st = emis(bino[a], [static_tr], dt)[..., 0]
                    uneq.append(float((mix[a] - st).sum() / (bino.shape[2] * len(a))))
                static_t = np.maximum(0.5 * (lC + lS), 1e-3)
                st2 = emis(bino, [static_t], dt)[..., 0]
                eq.append(float((mix - st2).sum() / (bino.shape[2] * n_tr)))
    u, e = float(np.mean(uneq)), float(np.mean(eq))
    ratio = e / u if u != 0 else float("nan")
    return dict(value=e, unequal_noise=u, equal_noise=e, ratio=float(ratio),
                threshold=THR["B5_max_ratio"], reference=REF["fig3a_equal"],
                switching=bool(e > 0),
                passed=bool(e > 0 and ratio <= THR["B5_max_ratio"]))


def check_B11_B12(sessions, dwell10, seed=0):
    """B11 the CV of the dwell times, B12 the overdispersion of the state-1 fraction over trials."""
    cv = dwell10["cv_edges_excluded"]
    call = ("markov" if THR["B11_markov"][0] <= cv <= THR["B11_markov"][1] else
            "rhythmic" if cv < THR["B11_rhythm_max"] else
            "continuous" if cv > THR["B11_continuous_min"] else "between")
    rng = np.random.default_rng(seed)
    obs, null = 0.0, 0.0
    for s in sessions:
        for pr in session_states(s, bin_ms=10.0):
            for sl in (slice(0, pr["na"]), slice(pr["na"], pr["na"] + pr["nb"])):
                z = pr["c"][sl].astype(np.int8)
                if z.shape[0] < 2:
                    continue
                obs += float(z.mean(axis=1).var())
                flat = z.ravel()
                null += float(np.mean([np.roll(flat, rng.integers(0, flat.size)).reshape(z.shape)
                                       .mean(axis=1).var() for _ in range(50)]))
    R = obs / null if null else float("nan")
    return {
        "B11_dwell_cv": dict(value=float(cv), markov_band=THR["B11_markov"], call=call,
                             reference=dict(M0=REF["M0_dwell_cv"], M1=REF["M1_dwell_cv"],
                                            M1b=REF["M1b_dwell_cv"], M2=REF["M2_dwell_cv"],
                                            M3=REF["M3_dwell_cv"]),
                             passed=bool(call == "markov")),
        "B12_overdispersion": dict(value=float(R), threshold=THR["B12_max"],
                                   reference=dict(M0=REF["M0_overdispersion"], M6=REF["M6_overdispersion"]),
                                   heterogeneous=bool(R >= THR["B12_hetero_min"]),
                                   passed=bool(R <= THR["B12_max"])),
    }


# ----------------------------------------------------------------------------------------------
# the gain block: needs a face feature space (the second experiment)
# ----------------------------------------------------------------------------------------------
def state_resolved_responses(sess, seed=0):
    """Responses of every cell to every face, split by role (conscious / suppressed) and by state,
    with the states of one half of the neurons applied to the responses of the other half
    (version 1.0). Returns prim (n_faces_s, N), R (2 roles, 2 states, 3 windows, n_faces_s, N) and
    the face indices of the session."""
    k = int(round(10.0 / sess.bin_ms))
    dt = sess.bin_ms * k / 1000.0
    N, T = sess.n_neurons, sess.n_bins // k
    rng = np.random.default_rng([int(seed), 7])
    perm = rng.permutation(N)
    hA, hB = np.sort(perm[: N // 2]), np.sort(perm[N // 2:])
    faces = sorted({f for p in sess.pairs for f in (p.face_a, p.face_b)})
    pos = {f: i for i, f in enumerate(faces)}
    prim = np.full((len(faces), N), np.nan, np.float32)
    R = np.full((2, 2, 3, len(faces), N), np.nan, np.float32)
    wins = [slice(0, T), slice(0, T // 2), slice(T // 2, T)]
    for p in sess.pairs:
        ba, bb_ = rebin(p.bino_a, k), rebin(p.bino_b, k)
        pa, pb = rebin(p.prime_a, k), rebin(p.prime_b, k)
        lamA, lamB = template(pa, dt), template(pb, dt)
        prim[pos[p.face_a]] = pa.sum(axis=1).mean(0) / (T * dt)
        prim[pos[p.face_b]] = pb.sum(axis=1).mean(0) / (T * dt)
        bino = np.concatenate([ba, bb_])
        na = ba.shape[0]
        cond = np.concatenate([np.zeros(na, int), np.ones(bb_.shape[0], int)])
        cs = {}
        for nm, idx in (("A", hA), ("B", hB)):
            E = emis(bino[..., idx], [lamA[idx], lamB[idx]], dt)[None]
            z = C.viterbi_batch(E, C.hmm_fit_batch(E))[0]
            cs[nm] = np.where(cond[:, None] == 0, z == 0, z == 1)
        bf = bino.astype(np.float32)
        for face, conds in ((p.face_a, (0, 1)), (p.face_b, (1, 0))):
            for role, cd in enumerate(conds):
                sl = slice(0, na) if cd == 0 else slice(na, bino.shape[0])
                x = bf[sl]
                for st in (0, 1):
                    mA = cs["A"][sl] if st == 0 else ~cs["A"][sl]
                    mB = cs["B"][sl] if st == 0 else ~cs["B"][sl]
                    for wi, w in enumerate(wins):
                        keep = np.zeros(T, bool); keep[w] = True
                        for m, idx in ((mA, hB), (mB, hA)):
                            mm = (m & keep)
                            n_ok = mm.sum()
                            if n_ok:
                                R[role, st, wi, pos[face], idx] = (
                                    mm.reshape(-1).astype(np.float32)
                                    @ x[..., idx].reshape(-1, len(idx))) / (n_ok * dt)
    return prim, R, np.array(faces)


def gain_block(sessions, seed=0, r2_min=0.05):
    """B6 the state decoder with and without the per-state mean rate, B7 the early-late change of
    the gain ratio, B9 the sign of the mean rate, B10 the gain against the coding R^2."""
    from scipy import stats as _st
    from sklearn.svm import LinearSVC
    if any(s.faces is None for s in sessions):
        return dict(available=False, reason="no face feature space in the release; B6-B10 skipped")
    gS1C, gS2S, gE, gL, gPrime, r2s, mS1, mS2 = [], [], [], [], [], [], [], []
    V1s, V2s, face_sets = [], [], []
    skipped = []
    for s in sessions:
        prim, R, faces = state_resolved_responses(s, seed=seed)
        F = s.faces[faces]
        if F.shape[0] < F.shape[1] + 10:
            # the leave-one-face-out coding model needs more faces than features
            skipped.append(dict(session=s.name, n_faces=int(F.shape[0]), n_features=int(F.shape[1])))
            continue
        loo, r2 = C.lofo_linear_predictions(F, np.nan_to_num(prim.astype(float)))
        keep = r2 > r2_min
        if keep.sum() < 10:
            continue
        proj = loo[:, keep]

        def slope(role, st, wi):
            y = R[role, st, wi][:, keep].astype(float)
            out = np.full(proj.shape[1], np.nan)
            for i in range(proj.shape[1]):
                ok = np.isfinite(y[:, i])
                if ok.sum() > 10:
                    out[i] = C.ols_slope(proj[ok, i], y[ok, i])
            return out
        gS1C.append(slope(0, 0, 0)); gS2S.append(slope(1, 1, 0))
        gE.append((slope(0, 0, 1), slope(1, 1, 1)))
        gL.append((slope(0, 0, 2), slope(1, 1, 2)))
        gPrime.append(np.array([C.ols_slope(proj[:, i], prim[:, keep][:, i].astype(float))
                                for i in range(proj.shape[1])]))
        r2s.append(r2[keep])
        mS1.append(np.nanmean(R[0, 0, 0][:, keep], axis=0))
        mS2.append(np.nanmean(R[1, 1, 0][:, keep], axis=0))
        V1s.append(np.nan_to_num(R[0, 0, 0][:, keep].astype(float)))
        V2s.append(np.nan_to_num(R[1, 1, 0][:, keep].astype(float)))
        face_sets.append(faces)
    if not gS1C:
        return dict(available=False, skipped=skipped,
                    reason="no session has enough faces for the coding model; B6-B10 skipped")
    g1 = np.concatenate(gS1C); g2 = np.concatenate(gS2S)
    e1 = np.concatenate([a for a, _ in gE]); e2 = np.concatenate([b for _, b in gE])
    l1 = np.concatenate([a for a, _ in gL]); l2 = np.concatenate([b for _, b in gL])
    gp = np.concatenate(gPrime); r2a = np.concatenate(r2s)
    m1 = np.concatenate(mS1); m2 = np.concatenate(mS2)
    ok = np.isfinite(g1) & np.isfinite(g2)
    ratio = float(np.nanmean(g1[ok]) / np.nanmean(g2[ok]))
    ratio_e = float(np.nanmean(e1) / np.nanmean(e2))
    ratio_l = float(np.nanmean(l1) / np.nanmean(l2))
    frac_s2 = float(np.mean(m2 > m1))
    r_all = float(np.corrcoef(g1[ok], g2[ok])[0, 1])
    q = np.quantile(r2a[ok], [1 / 3, 2 / 3])
    band = ok & (r2a >= q[0]) & (r2a <= q[1])
    r_band = float(np.corrcoef(g1[band], g2[band])[0, 1]) if band.sum() > 20 else float("nan")
    rho = float(_st.spearmanr(r2a[ok], g1[ok]).statistic)
    # decoder of the concatenation order, raw and after removing the per-state mean rate
    rng = np.random.default_rng([int(seed), 13])
    accs = {}
    for mode in ("raw", "no_mean", "no_mean_no_scale"):
        vals = []
        for V1, V2, faces in zip(V1s, V2s, face_sets):
            n_f = V1.shape[0] - V1.shape[0] % 2
            pairs = np.arange(n_f // 2)
            if len(pairs) < 8:
                continue
            for _ in range(20):
                p = rng.permutation(pairs)
                tr, te = p[: len(p) // 2], p[len(p) // 2:]
                f_tr = np.concatenate([2 * tr, 2 * tr + 1])
                # every normalization is estimated on the training faces alone and applied to both
                A1, A2 = V1.copy(), V2.copy()
                if mode != "raw":
                    A1 = A1 - V1[f_tr].mean(axis=0, keepdims=True)
                    A2 = A2 - V2[f_tr].mean(axis=0, keepdims=True)
                if mode == "no_mean_no_scale":
                    A1 = A1 / (V1[f_tr].std(axis=0, keepdims=True) + 1e-6)
                    A2 = A2 / (V2[f_tr].std(axis=0, keepdims=True) + 1e-6)
                both = np.concatenate([A1[f_tr], A2[f_tr]])
                mu, sd = both.mean(0), both.std(0) + 1e-6
                A1 = (A1 - mu) / sd; A2 = (A2 - mu) / sd

                def build(pp):
                    f = np.concatenate([2 * pp, 2 * pp + 1])
                    Xa = np.hstack([A1[f], A2[f]]); Xb = np.hstack([A2[f], A1[f]])
                    return np.vstack([Xa, Xb]), np.concatenate([np.ones(len(f)), np.zeros(len(f))])
                Xtr, ytr = build(tr); Xte, yte = build(te)
                clf = LinearSVC(C=1.0, dual=False, max_iter=20000).fit(Xtr, ytr)
                vals.append(float((clf.predict(Xte) == yte).mean()))
        # the order decoder can be systematically inverted; what matters is how much the two
        # states differ at all, so the oriented accuracy max(a, 1 - a) is reported
        accs[mode] = float(max(np.mean(vals), 1 - np.mean(vals))) if vals else None
        accs[mode + "_signed"] = float(np.mean(vals)) if vals else None
    out = dict(available=True, n_cells=int(ok.sum()), sessions_skipped=skipped, gain_ratio=ratio,
               gain_ratio_early=ratio_e, gain_ratio_late=ratio_l,
               ratio_S1C_over_prime=float(np.nanmean(g1[ok]) / np.nanmean(gp[ok])),
               ratio_S2S_over_prime=float(np.nanmean(g2[ok]) / np.nanmean(gp[ok])),
               r_gain_all=r_all, r_gain_r2_band=r_band, decoder=accs)
    out["B6_decoder_without_mean_rate"] = dict(
        value=accs.get("no_mean"), raw=accs.get("raw"), pattern_only=accs.get("no_mean_no_scale"),
        threshold=THR["B6_max"], reference=REF["decoder_no_mean"],
        passed=(None if accs.get("no_mean") is None else bool(accs["no_mean"] <= THR["B6_max"])),
        refuted=(None if accs.get("no_mean") is None
                 else bool(accs["no_mean"] >= THR["B6_refuted_min"])))
    out["B7_gain_early_minus_late"] = dict(value=float(ratio_e - ratio_l), threshold=THR["B7_min"],
                                           early=ratio_e, late=ratio_l, gain_ratio=ratio,
                                           applicable=bool(ratio >= THR["B9_ratio_min"]),
                                           reference=dict(M4=REF["M4_ratio_early_minus_late"],
                                                          M5=REF["M5_ratio_early_minus_late"]),
                                           passed=bool(ratio >= THR["B9_ratio_min"]
                                                       and ratio_e - ratio_l >= THR["B7_min"]))
    out["B9_mean_rate_sign"] = dict(frac_state2_higher=frac_s2, gain_ratio=ratio,
                                    thresholds=(THR["B9_frac_min"], THR["B9_ratio_min"]),
                                    reference=dict(M4=REF["M4_frac_s2_higher"], M5=REF["M5_frac_s2_higher"]),
                                    unexplained_by_tested_models=bool(frac_s2 >= THR["B9_frac_min"]
                                                                      and ratio >= THR["B9_ratio_min"]))
    out["B10_gain_vs_r2"] = dict(spearman_rho=rho, r_all=r_all, r_band=r_band,
                                 drop=float(r_all - r_band) if np.isfinite(r_band) else None,
                                 thresholds=(THR["B10_rho_min"], THR["B10_drop_min"]),
                                 passed=bool(rho >= THR["B10_rho_min"] and np.isfinite(r_band)
                                             and (r_all - r_band) >= THR["B10_drop_min"]))
    return out


# ----------------------------------------------------------------------------------------------
# driver for the count-only block
# ----------------------------------------------------------------------------------------------
def run_counts_block(sessions, seed=0):
    check_sessions(sessions)
    A, acf, _ = check_A(sessions, rng=None)
    st10 = [session_states(s, bin_ms=10.0) for s in sessions]
    dw = dwell_stats(st10)
    out = dict(A=A, acf=acf, dwell_10ms=dw)
    out["B"] = {}
    out["B"].update(check_B1_B3(acf, dw))
    out["B"]["B4_label_threshold"] = two_rate_advantage(sessions, seed=seed)
    out["B"]["B5_fig3a_equal_noise"] = fig3a_equal_noise(sessions, seed=seed)
    out["B"].update(check_B11_B12(sessions, dw, seed=seed))
    signal_ok = bool(acf["acf_first_lag"] >= THR["SIGNAL_MIN"])
    out["signal_ok"] = signal_ok
    out["signal_value"] = acf["acf_first_lag"]
    for k in SIGNAL_GATED:
        out["B"][k]["conditional_on_signal"] = True
        out["B"][k]["signal_ok"] = signal_ok
        if not signal_ok:
            out["B"][k]["passed"] = None
            out["B"][k]["note"] = "state signal below the pre-registered floor; descriptive only"
    out["n_sessions"] = len(sessions)
    out["n_neurons_total"] = int(sum(s.n_neurons for s in sessions))
    out["n_pairs_total"] = int(sum(len(s.pairs) for s in sessions))
    return out
