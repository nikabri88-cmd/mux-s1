"""E4 (post hoc, exploratory, NOT pre-registered): a scale-corrected version of statistic D2 (kappa4).

Problem found in part II: the pre-registered D2 pools raw four-quarter products over stimulus pairs whose
signal scale differs several-fold; a scale mixture of a binary signal has a large positive fourth
cumulant, so the pooled value no longer sits at -2 for jumps. Correction: before pooling, each quarter
projection of each pair-condition is divided by its template-based expected separation
a_k = sum_{i in quarter} w_i (lamC_i - lamS_i) dt (independent of the binocular data), then centred
across trials per bin as before. Ground truth: the same cumulant of the latent weight of the conscious
face (m-bar; for the ring, f_C + (f_X + f_Y)/2). DYN design, calibrated kappa and timescales.
Usage: python explore_k4.py
"""
import json
import time
from multiprocessing import Pool

import numpy as np

from mux_s1 import core as C
from mux_s1 import experiments as X
from mux_s1.runner import MODEL_INDEX

MODELS = ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal", "M6_heterogeneous", "M7_ring"]
SEEDS = [0, 1, 2]
N_SESS = 19
KEYS = ("s1234", "s12", "s34", "s13", "s24", "s14", "s23")


def k4_value(s):
    n = s["n"]
    den = np.mean([(s["s12"] / n) * (s["s34"] / n), (s["s13"] / n) * (s["s24"] / n), (s["s14"] / n) * (s["s23"] / n)])
    return float(s["s1234"] / n / den - 3.0)


def accumulate(s, L):
    L = [x - x.mean(axis=0, keepdims=True) for x in L]
    s["s1234"] += float((L[0] * L[1] * L[2] * L[3]).sum())
    for (p, q), key in (((0, 1), "s12"), ((2, 3), "s34"), ((0, 2), "s13"), ((1, 3), "s24"), ((0, 3), "s14"), ((1, 2), "s23")):
        s[key] += float((L[p] * L[q]).sum())
    s["n"] += L[0].size


def session(rng, spec, ts, kappa, sums, N=300, n_pairs=25, n_tr=15):
    pop = C.Population(rng, N, kappa)
    F = C.make_faces(rng, 2 * n_pairs)
    D = pop.delta(F)
    perm = rng.permutation(N)
    quarters = [np.sort(perm[k::4]) for k in range(4)]
    for j in range(n_pairs):
        a, b = 2 * j, 2 * j + 1
        ring_u = None
        if spec["kind"] == "ring":
            u = rng.standard_normal(N)
            ring_u = u / np.linalg.norm(u) * np.linalg.norm(D[a] - D[b]) / 2
        lamA = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr)))
        lamB = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr)))
        for fc, fs, lC, lS in ((a, b, lamA, lamB), (b, a, lamB, lamA)):
            lam, m = C.binocular_rates(rng, spec, pop, D[fc], D[fs], n_tr, ts, 1000.0, ring_u)
            bb = C.poisson(rng, lam)
            w = np.log(lC / lS)
            L = [bb[..., q] @ w[q] for q in quarters]
            accumulate(sums["raw"], L)
            ahat = [float(w[q] @ (lC[q] - lS[q]) * C.DT) for q in quarters]
            accumulate(sums["scaled"], [x / s_ for x, s_ in zip(L, ahat)])
            if m is not None:
                mm = m if m.ndim == 2 else m[..., 0] + 0.5 * (m[..., 1] + m[..., 3])
                accumulate(sums["truth"], [mm] * 4)


def job(args):
    model, seed, ts, kappa = args
    t0 = time.time()
    rng = np.random.default_rng([seed, MODEL_INDEX[model], 6])
    sums = {k: dict({kk: 0.0 for kk in KEYS}, n=0) for k in ("raw", "scaled", "truth")}
    for _ in range(N_SESS):
        session(rng, C.MODELS[model], ts, kappa, sums)
    out = {k: (k4_value(v) if v["n"] else None) for k, v in sums.items()}
    print(time.strftime("%H:%M:%S"), model, seed, {k: (round(v, 2) if v is not None else None) for k, v in out.items()},
          round(time.time() - t0, 1), "s", flush=True)
    return model, seed, out


if __name__ == "__main__":
    cal = json.load(open("results/calibration.json"))
    jobs = [(m, s, cal["model_timescale"][m], cal["kappa"]) for s in SEEDS for m in MODELS]
    res = {}
    with Pool(2) as pool:
        for model, seed, out in pool.imap_unordered(job, jobs):
            res.setdefault(model, {})[seed] = out
    json.dump(res, open("results/explore_k4.json", "w"), indent=1)
    print("done explore_k4")
