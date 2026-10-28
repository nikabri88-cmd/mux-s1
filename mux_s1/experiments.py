"""MUX-S1 experiments: simulated sessions in three designs and per-session analyses.

FACE1500  - Hesse et al. second experiment: 1500 faces in 750 pairs, 1 s priming, 1 s test.
TEN       - Hesse et al. first experiment: 10 stimuli, 25 pairs, 2 s priming, binocular and
            unambiguous contexts (modulation indices).
DYN       - dynamics-only design for the discriminating statistics (part II).

States: conscious-state indicator c = 1 when the population is assigned to the template of the
conscious (flashed) face ("state 1"), 0 for the suppressed face ("state 2").
"""
import numpy as np
from scipy import stats

from . import core as C

PSEUDO = 0.5     # pseudo-count (spikes) added to templates


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def template(counts):
    n_tr = counts.shape[0]
    return (counts.sum(axis=(0, 1)) + PSEUDO) / (n_tr * C.NB * C.DT)


def per_trial_rates(counts):
    return counts.sum(axis=1) / (C.NB * C.DT)


def masked_rate(cf, mask):
    """cf: float32 counts (n_tr, NB, n), mask (n_tr, NB) -> mean rate over masked bins (n,)."""
    k = mask.sum()
    if k == 0:
        return np.full(cf.shape[2], np.nan)
    return (mask.reshape(-1).astype(np.float32) @ cf.reshape(-1, cf.shape[2])) / (k * C.DT)


def mixture_vs_static(bino, lamC, lamS, static):
    eC = C.emis_loglik(bino, [lamC])[..., 0]
    eS = C.emis_loglik(bino, [lamS])[..., 0]
    mix = np.logaddexp(eC, eS) + np.log(0.5)
    st = C.emis_loglik(bino, [np.maximum(static, 1e-3)])[..., 0]
    return (mix - st).sum(axis=1) / bino.shape[2]


def heldout_two_vs_static(counts, cstate, lamC, lamS, idx):
    n = counts[..., idx].astype(float)
    lc, ls = lamC[idx] * C.DT, lamS[idx] * C.DT
    lam2 = np.where(cstate[..., None] == 1, lc, ls)
    ll2 = (n * np.log(lam2) - lam2).sum(axis=(0, 1))
    nb = n.shape[0] * n.shape[1]
    nsum = n.sum(axis=(0, 1))
    best = None
    for p in np.linspace(0, 1, 41):
        lamp = p * lc + (1 - p) * ls
        llp = nsum * np.log(lamp) - lamp * nb
        if best is None or llp.sum() > best.sum():
            best = llp
    return (ll2 - best) / nb


def pair_hmms(groups):
    """groups: list over pairs of (E_all, E_hA, E_hB), each E (M, NB, 2) with state 0 = face a.
    Pairs are batched by M. Returns list over pairs of (z_all, z_hA, z_hB)."""
    out = [None] * len(groups)
    by_m = {}
    for i, g in enumerate(groups):
        by_m.setdefault(g[0].shape[0], []).append(i)
    for M, idxs in by_m.items():
        res = []
        for k in range(3):
            E = np.stack([groups[i][k] for i in idxs])            # (P, M, NB, 2)
            Tm = C.hmm_fit_batch(E)
            res.append(C.viterbi_batch(E, Tm))
        for n_, i in enumerate(idxs):
            out[i] = (res[0][n_], res[1][n_], res[2][n_])
    return out


def emis3(counts, lamA, lamB, hA, hB):
    return (C.emis_loglik(counts, [lamA, lamB]),
            C.emis_loglik(counts[..., hA], [lamA[hA], lamB[hA]]),
            C.emis_loglik(counts[..., hB], [lamA[hB], lamB[hB]]))


def runs_add(runs, cz):
    rr = C.run_lengths(cz.astype(np.int8))
    runs[1] += rr[1]; runs[0] += rr[0]
    rx = C.run_lengths(cz.astype(np.int8), exclude_edges=True)
    runs["ex"][1] += rx[1]; runs["ex"][0] += rx[0]


def new_runs():
    return {0: [], 1: [], "ex": {0: [], 1: []}}


# ----------------------------------------------------------------------------------------
# FACE1500
# ----------------------------------------------------------------------------------------
def face1500_session(rng, spec, timescale, faces, kappa, N=300, n_hr_pairs=30, n_lr_pairs=200,
                     n_tr_hr=20, n_tr_lr=2, t_prime=1000.0):
    pop = C.Population(rng, N, kappa)
    n_faces = faces.shape[0]
    D = pop.delta(faces)
    perm = rng.permutation(N)
    hA, hB = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    n_res = n_hr_pairs + n_lr_pairs
    n_an = 2 * n_res
    prim = np.zeros((n_faces, N))
    # ---- simulate all resolved pairs
    data, groups = [], []
    for j in range(n_res):
        a, b = 2 * j, 2 * j + 1
        n_tr = n_tr_hr if j < n_hr_pairs else n_tr_lr
        ring_u = None
        if spec["kind"] == "ring":
            u = rng.standard_normal(N)
            ring_u = u / np.linalg.norm(u) * np.linalg.norm(D[a] - D[b]) / 2
        prime_b = C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr))   # cond 0 priming (b suppressed)
        prime_a = C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr))   # cond 1 priming (a suppressed)
        lam0, _ = C.binocular_rates(rng, spec, pop, D[a], D[b], n_tr, timescale, t_prime, ring_u)
        lam1, _ = C.binocular_rates(rng, spec, pop, D[b], D[a], n_tr, timescale, t_prime, ring_u)
        bino = np.concatenate([C.poisson(rng, lam0), C.poisson(rng, lam1)])   # cond0 trials then cond1
        lamA, lamB = template(prime_a), template(prime_b)
        prim[a] = per_trial_rates(prime_a).mean(0)
        prim[b] = per_trial_rates(prime_b).mean(0)
        data.append((bino, prime_a, prime_b, lamA, lamB, n_tr))
        groups.append(emis3(bino, lamA, lamB, hA, hB))
    zs = pair_hmms(groups)
    # ---- per-pair analyses
    wins = [slice(0, C.NB), slice(0, C.NB // 2), slice(C.NB // 2, C.NB)]
    R_all = np.full((2, 2, n_an, N), np.nan, dtype=np.float32)        # role, state
    R_xv = np.full((2, 2, 3, n_an, N), np.nan, dtype=np.float32)      # role, state, window
    llr_b, occ_p, occ_b, lik_mix, lik_held = [], [], [], [], []
    runs = new_runs()
    for j in range(n_res):
        bino, prime_a, prime_b, lamA, lamB, n_tr = data[j]
        z, zA, zB = zs[j]
        cond = np.repeat([0, 1], n_tr)
        # conscious indicator: cond0 conscious = a (state 0); cond1 conscious = b (state 1)
        conv = lambda zz: np.where(cond[:, None] == 0, zz == 0, zz == 1)
        c, cA, cB = conv(z), conv(zA), conv(zB)
        l_b = np.concatenate([C.llr(bino[:n_tr], lamA, lamB), C.llr(bino[n_tr:], lamB, lamA)])
        l_p = np.concatenate([C.llr(prime_b, lamA, lamB), C.llr(prime_a, lamB, lamA)])
        occ_p.append((l_p > 0).mean()); occ_b.append((l_b > 0).mean())
        if j < n_hr_pairs:
            llr_b.append(l_b)
            runs_add(runs, c)
            for cc, lC, lS in ((slice(0, n_tr), lamA, lamB), (slice(n_tr, 2 * n_tr), lamB, lamA)):
                bb = bino[cc]
                h = n_tr // 2
                m1 = per_trial_rates(bb[h:]).mean(0); m2 = per_trial_rates(bb[:h]).mean(0)
                lik_mix.append(np.concatenate([mixture_vs_static(bb[:h], lC, lS, m1),
                                               mixture_vs_static(bb[h:], lC, lS, m2)]).mean())
                lik_held.append(np.concatenate([
                    heldout_two_vs_static(bb, cA[cc].astype(np.int8), lC, lS, hB),
                    heldout_two_vs_static(bb, cB[cc].astype(np.int8), lC, lS, hA)]))
        bf = bino.astype(np.float32)
        # roles: face a conscious in cond0, suppressed in cond1; face b the reverse
        for face_i, conds in ((2 * j, (0, 1)), (2 * j + 1, (1, 0))):
            for role, cd in enumerate(conds):
                sl = slice(0, n_tr) if cd == 0 else slice(n_tr, 2 * n_tr)
                bb = bf[sl]
                for st in (0, 1):
                    m_all = c[sl] if st == 0 else ~c[sl]
                    R_all[role, st, face_i] = masked_rate(bb, m_all)
                    mA = cA[sl] if st == 0 else ~cA[sl]
                    mB = cB[sl] if st == 0 else ~cB[sl]
                    for wi, w in enumerate(wins):
                        keep = np.zeros(C.NB, bool); keep[w] = True
                        R_xv[role, st, wi, face_i, hB] = masked_rate(bb[..., hB], mA & keep)
                        R_xv[role, st, wi, face_i, hA] = masked_rate(bb[..., hA], mB & keep)
    # priming averages for the remaining faces (n_tr_lr trials each)
    rest = np.arange(n_an, n_faces)
    lam_win = np.stack([C.priming_rates(spec, pop, D[f], 1)[0].sum(axis=0) for f in rest]) * C.DT * n_tr_lr
    prim[rest] = rng.poisson(lam_win) / (n_tr_lr * C.NB * C.DT)
    return dict(prim=prim.astype(np.float32), R_all=R_all, R_xv=R_xv, hA=hA, hB=hB,
                llr_bino=np.concatenate(llr_b), occ_prime=float(np.mean(occ_p)),
                occ_bino=float(np.mean(occ_b)), runs=runs, lik_mix=np.array(lik_mix),
                lik_held=np.concatenate(lik_held), n_an=n_an)


# ----------------------------------------------------------------------------------------
# TEN (modulation indices; 2 s priming; binocular + unambiguous contexts)
# ----------------------------------------------------------------------------------------
def ten_session(rng, spec, timescale, kappa, N=300, n_hr=4, tr_hr=30, tr_lr=5, t_prime=2000.0):
    pop = C.Population(rng, N, kappa)
    F = C.make_faces(rng, 10)
    D = pop.delta(F)
    perm = rng.permutation(N)
    hA, hB = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    pairs = [(i, 5 + k) for i in range(5) for k in range(5)]
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    prime_store = {f: [] for f in range(10)}
    recs = []
    for pi_, (a, b) in enumerate(pairs):
        n_tr = tr_hr if pi_ < n_hr else tr_lr
        ring_u = None
        if spec["kind"] == "ring":
            u = rng.standard_normal(N)
            ring_u = u / np.linalg.norm(u) * np.linalg.norm(D[a] - D[b]) / 2
        rec = {"pair": (a, b), "n_tr": n_tr}
        binos = []
        for dom, (fc, fs) in enumerate(((a, b), (b, a))):          # dom 0: a flashed/conscious
            pr = C.poisson(rng, C.priming_rates(spec, pop, D[fs], 2 * n_tr))
            prime_store[fs].append(pr)
            lam, _ = C.binocular_rates(rng, spec, pop, D[fc], D[fs], n_tr, timescale, t_prime, ring_u)
            binos.append(C.poisson(rng, lam))
            rec[("unamb", dom)] = C.poisson(rng, C.unambiguous_rates(spec, pop, D[fc], n_tr))
            rec[("prime", dom)] = pr
        rec["bino"] = np.concatenate(binos)
        recs.append(rec)
    tmpl = {f: template(np.concatenate(prime_store[f], axis=0)) for f in range(10)}
    groups = [emis3(r["bino"], tmpl[r["pair"][0]], tmpl[r["pair"][1]], hA, hB) for r in recs]
    zs = pair_hmms(groups)
    rows, llr_b, occ_p, occ_b, lik_mix, lik_held = [], [], [], [], [], []
    runs = new_runs()
    for rec, (z, zA, zB) in zip(recs, zs):
        a, b = rec["pair"]; n_tr = rec["n_tr"]
        lamA, lamB = tmpl[a], tmpl[b]
        bino = rec["bino"]
        cond = np.repeat([0, 1], n_tr)
        conv = lambda zz: np.where(cond[:, None] == 0, zz == 0, zz == 1)
        c, cA, cB = conv(z), conv(zA), conv(zB)
        l_b = np.concatenate([C.llr(bino[:n_tr], lamA, lamB), C.llr(bino[n_tr:], lamB, lamA)])
        l_p = np.concatenate([C.llr(rec[("prime", 0)], lamA, lamB), C.llr(rec[("prime", 1)], lamB, lamA)])
        llr_b.append(l_b); occ_p.append((l_p > 0).mean()); occ_b.append((l_b > 0).mean())
        runs_add(runs, c)
        resp = {}
        bf = bino.astype(np.float32)
        for dom, (sl, lC, lS) in enumerate(((slice(0, n_tr), lamA, lamB), (slice(n_tr, 2 * n_tr), lamB, lamA))):
            bb = bino[sl]
            h = n_tr // 2
            m1 = per_trial_rates(bb[h:]).mean(0); m2 = per_trial_rates(bb[:h]).mean(0)
            lik_mix.append(np.concatenate([mixture_vs_static(bb[:h], lC, lS, m1),
                                           mixture_vs_static(bb[h:], lC, lS, m2)]).mean())
            lik_held.append(np.concatenate([heldout_two_vs_static(bb, cA[sl].astype(np.int8), lC, lS, hB),
                                            heldout_two_vs_static(bb, cB[sl].astype(np.int8), lC, lS, hA)]))
            resp[("unamb", dom)] = per_trial_rates(rec[("unamb", dom)])
            resp[("bavg", dom)] = per_trial_rates(bb).mean(0)
            for st in (0, 1):
                xv = np.full(N, np.nan)
                xv[hB] = masked_rate(bf[sl][..., hB], cA[sl] if st == 0 else ~cA[sl])
                xv[hA] = masked_rate(bf[sl][..., hA], cB[sl] if st == 0 else ~cB[sl])
                resp[("xv", dom, st)] = xv
        pa = per_trial_rates(rec[("prime", 1)]); pb = per_trial_rates(rec[("prime", 0)])
        with np.errstate(all="ignore"):
            p_prime = stats.ttest_ind(pa, pb, axis=0).pvalue
            ua, ub = resp[("unamb", 0)], resp[("unamb", 1)]
            p_unamb = stats.ttest_ind(ua, ub, axis=0).pvalue
            mi = lambda x, y: (x - y) / (x + y)
            rows.append(dict(
                mi_unamb=mi(ua.mean(0), ub.mean(0)),
                mi_bavg=mi(resp[("bavg", 0)], resp[("bavg", 1)]),
                mi_s1=mi(resp[("xv", 0, 0)], resp[("xv", 1, 0)]),
                mi_s2=mi(resp[("xv", 0, 1)], resp[("xv", 1, 1)]),
                r_s1c=0.5 * (resp[("xv", 0, 0)] + resp[("xv", 1, 0)]),
                r_s2=0.5 * (resp[("xv", 0, 1)] + resp[("xv", 1, 1)]),
                p_prime=p_prime, p_unamb=p_unamb,
                pos=np.all(np.vstack([resp[("xv", d, s)] for d in (0, 1) for s in (0, 1)]) > 0, axis=0)))
    return dict(rows=rows, llr_bino=np.concatenate(llr_b), occ_prime=float(np.mean(occ_p)),
                occ_bino=float(np.mean(occ_b)), runs=runs, lik_mix=np.array(lik_mix),
                lik_held=np.concatenate(lik_held))


# ----------------------------------------------------------------------------------------
# DYN (part II discriminators)
# ----------------------------------------------------------------------------------------
X_GRID = np.linspace(-5, 5, 21)
M_GRID = 1 / (1 + np.exp(-X_GRID))
TAU_GRID = [15.0, 30.0, 60.0, 120.0]
SIG_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
PSW_GRID = [0.05, 0.08, 0.12, 0.17, 0.25, 0.35]


def ou_kernel(tau, sigma):
    a = np.exp(-10.0 / tau)
    q = sigma * np.sqrt(1 - a * a)
    K = np.exp(-0.5 * ((X_GRID[None, :] - a * X_GRID[:, None]) / q) ** 2)
    K /= K.sum(axis=1, keepdims=True)
    pi = np.exp(-0.5 * (X_GRID / sigma) ** 2)
    pi /= pi.sum()
    return K, pi


def dyn_session(rng, spec, timescale, kappa, N=300, n_pairs=25, n_tr=15, t_prime=1000.0, want_hyst=False):
    pop = C.Population(rng, N, kappa)
    F = C.make_faces(rng, 2 * n_pairs)
    D = pop.delta(F)
    perm = rng.permutation(N)
    hA, hB = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    quarters = [np.sort(perm[k::4]) for k in range(4)]
    data, groups = [], []
    for j in range(n_pairs):
        a, b = 2 * j, 2 * j + 1
        ring_u = None
        if spec["kind"] == "ring":
            u = rng.standard_normal(N)
            ring_u = u / np.linalg.norm(u) * np.linalg.norm(D[a] - D[b]) / 2
        prime_b = C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr))
        prime_a = C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr))
        lam0, _ = C.binocular_rates(rng, spec, pop, D[a], D[b], n_tr, timescale, t_prime, ring_u)
        lam1, _ = C.binocular_rates(rng, spec, pop, D[b], D[a], n_tr, timescale, t_prime, ring_u)
        bino = np.concatenate([C.poisson(rng, lam0), C.poisson(rng, lam1)])
        lamA, lamB = template(prime_a), template(prime_b)
        data.append((bino, prime_a, prime_b, lamA, lamB))
        groups.append(emis3(bino, lamA, lamB, hA, hB))
    zs = pair_hmms(groups)
    llr_b, occ_p, occ_b = [], [], []
    runs = new_runs()
    k4 = dict(s1234=0.0, s12=0.0, s34=0.0, s13=0.0, s24=0.0, s14=0.0, s23=0.0, n=0)
    od_obs, od_null = 0.0, 0.0
    Ed = {"tr": [], "te": []}
    Ec = {"tr": [], "te": []}
    hyst = []
    cond = np.repeat([0, 1], n_tr)
    for (bino, prime_a, prime_b, lamA, lamB), (z, zA, zB) in zip(data, zs):
        conv = lambda zz: np.where(cond[:, None] == 0, zz == 0, zz == 1)
        c, cA = conv(z), conv(zA)
        l_b = np.concatenate([C.llr(bino[:n_tr], lamA, lamB), C.llr(bino[n_tr:], lamB, lamA)])
        l_p = np.concatenate([C.llr(prime_b, lamA, lamB), C.llr(prime_a, lamB, lamA)])
        llr_b.append(l_b); occ_p.append((l_p > 0).mean()); occ_b.append((l_b > 0).mean())
        runs_add(runs, c)
        for sl in (slice(0, n_tr), slice(n_tr, 2 * n_tr)):
            cz = c[sl].astype(np.int8)
            od_obs += cz.mean(axis=1).var()
            flat = cz.ravel()
            nv = [np.roll(flat, rng.integers(0, flat.size)).reshape(cz.shape).mean(axis=1).var() for _ in range(50)]
            od_null += float(np.mean(nv))
        for sl, lC, lS in ((slice(0, n_tr), lamA, lamB), (slice(n_tr, 2 * n_tr), lamB, lamA)):
            bb = bino[sl]
            w = np.log(lC / lS)
            L = [bb[..., q] @ w[q] for q in quarters]
            L = [x - x.mean(axis=0, keepdims=True) for x in L]
            k4["s1234"] += float((L[0] * L[1] * L[2] * L[3]).sum())
            for (p_, q_), key in (((0, 1), "s12"), ((2, 3), "s34"), ((0, 2), "s13"), ((1, 3), "s24"),
                                  ((0, 3), "s14"), ((1, 2), "s23")):
                k4[key] += float((L[p_] * L[q_]).sum())
            k4["n"] += L[0].size
            ed = C.emis_loglik(bb, [lS, lC])                 # level index 1 = conscious face
            ec = C.emis_loglik(bb, [m * lC + (1 - m) * lS for m in M_GRID])
            Ed["tr"].append(ed[0::2]); Ed["te"].append(ed[1::2])
            Ec["tr"].append(ec[0::2]); Ec["te"].append(ec[1::2])
        if want_hyst:
            hyst.append(hysteresis_segments(bino, cA, lamA, lamB, hB, n_tr))
    Edtr, Edte = np.concatenate(Ed["tr"]), np.concatenate(Ed["te"])
    Ectr, Ecte = np.concatenate(Ec["tr"]), np.concatenate(Ec["te"])
    pi2 = np.array([0.5, 0.5])
    Tsw = lambda p: np.array([[1 - p, p], [p, 1 - p]])
    best_p = max(PSW_GRID, key=lambda p: C.forward_loglik_scaled(Edtr, Tsw(p), pi2).sum())
    lld = C.forward_loglik_scaled(Edte, Tsw(best_p), pi2).sum()
    best, best_c = -np.inf, None
    for tau in TAU_GRID:
        for sg in SIG_GRID:
            K, pi = ou_kernel(tau, sg)
            v = C.forward_loglik_scaled(Ectr, K, pi).sum()
            if v > best:
                best, best_c = v, (tau, sg)
    K, pi = ou_kernel(*best_c)
    llc = C.forward_loglik_scaled(Ecte, K, pi).sum()
    disc = dict(ll_cont_minus_disc=float(llc - lld), n_test_bins=int(Edte.shape[0] * Edte.shape[1]),
                best_psw=best_p, best_ou=best_c)
    return dict(llr_bino=np.concatenate(llr_b), occ_prime=float(np.mean(occ_p)),
                occ_bino=float(np.mean(occ_b)), runs=runs, k4=k4, od=(od_obs, od_null), disc=disc,
                hyst=hyst)


def hysteresis_segments(bino, cA, lamA, lamB, hB, n_tr):
    """Exploratory loop detector: held-out (half B) activity around transitions detected from
    half A. Returns list of (direction, segment (6, nB) rates, C-S axis)."""
    out = []
    for sl, lC, lS in ((slice(0, n_tr), lamA, lamB), (slice(n_tr, 2 * n_tr), lamB, lamA)):
        x = bino[sl][..., hB].astype(np.float32)
        zz = cA[sl]
        axis = (lC - lS)[hB]
        axis = axis / np.linalg.norm(axis)
        for tr in range(x.shape[0]):
            ch = np.flatnonzero(np.diff(zz[tr].astype(np.int8))) + 1
            for t in ch:
                if t - 3 < 0 or t + 3 > x.shape[1]:
                    continue
                out.append((int(zz[tr, t]), x[tr, t - 3:t + 3], axis, (lC + lS)[hB] / 2))
    return out
