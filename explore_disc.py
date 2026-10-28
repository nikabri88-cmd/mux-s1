"""E5 (post hoc, exploratory, NOT pre-registered): why does D1 favour the continuous model for pure jumps?

Two candidate mechanisms:
 (i) switches inside 10-ms bins create mixed bins (about 8% of bins for M0);
 (ii) priming templates are noisy (15 trials), and intermediate levels of the continuous model shrink the
     two noisy templates towards each other, which beats the raw templates of the two-level model.
Test, at the calibrated kappa and timescales, DYN design, D1 computed exactly as in the pre-registered
code (even trials fit, odd trials test):
 - M0 as generated, and M0 with the bin weight m-bar rounded to 0 or 1 (switches aligned to bin edges):
   mechanism (i) predicts that alignment removes the bias;
 - a fairer discrete model with two symmetric levels sigmoid(+-x0) fitted on the training trials
   (x0 from the same 21-level grid), compared with the same continuous model (D1-fit): mechanism (ii)
   predicts that D1-fit is near zero or negative for jumps and stays positive for continuous mixing.
Usage: python explore_disc.py
"""
import json
import time
from multiprocessing import Pool

import numpy as np

from mux_s1 import core as C
from mux_s1 import experiments as X
from mux_s1.runner import MODEL_INDEX

SEEDS = [0, 1]
N_SESS = 19
POS = np.where(X.X_GRID > 0)[0]
NEG = np.array([len(X.X_GRID) - 1 - i for i in POS])          # mirror indices (-x0)


def session(rng, spec, ts, kappa, aligned, N=300, n_pairs=25, n_tr=15):
    pop = C.Population(rng, N, kappa)
    F = C.make_faces(rng, 2 * n_pairs)
    D = pop.delta(F)
    Ed = {"tr": [], "te": []}
    Ec = {"tr": [], "te": []}
    for j in range(n_pairs):
        a, b = 2 * j, 2 * j + 1
        lamA = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr)))
        lamB = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr)))
        for fc, fs, lC, lS in ((a, b, lamA, lamB), (b, a, lamB, lamA)):
            m, _ = C.latent(rng, spec, n_tr, ts)
            if aligned:
                m = np.round(m)
            mm = m[..., None]
            lam = np.maximum(pop.b[None, None, :] + mm * D[fc][None, None, :] + (1 - mm) * D[fs][None, None, :], 0.5)
            bb = C.poisson(rng, lam)
            ed = C.emis_loglik(bb, [lS, lC])
            ec = C.emis_loglik(bb, [mg * lC + (1 - mg) * lS for mg in X.M_GRID])
            Ed["tr"].append(ed[0::2]); Ed["te"].append(ed[1::2])
            Ec["tr"].append(ec[0::2]); Ec["te"].append(ec[1::2])
    Edtr, Edte = np.concatenate(Ed["tr"]), np.concatenate(Ed["te"])
    Ectr, Ecte = np.concatenate(Ec["tr"]), np.concatenate(Ec["te"])
    pi2 = np.array([0.5, 0.5])
    Tsw = lambda p: np.array([[1 - p, p], [p, 1 - p]])
    F_ = C.forward_loglik_scaled
    best_p = max(X.PSW_GRID, key=lambda p: F_(Edtr, Tsw(p), pi2).sum())
    lld = F_(Edte, Tsw(best_p), pi2).sum()
    best, best_c = -np.inf, None
    for tau in X.TAU_GRID:
        for sg in X.SIG_GRID:
            K, pi = X.ou_kernel(tau, sg)
            v = F_(Ectr, K, pi).sum()
            if v > best:
                best, best_c = v, (tau, sg)
    K, pi = X.ou_kernel(*best_c)
    llc = F_(Ecte, K, pi).sum()
    bestf, bestf_c = -np.inf, None                              # discrete model with fitted symmetric levels
    for ip, ineg in zip(POS, NEG):
        Etr = Ectr[..., [ineg, ip]]
        for p in X.PSW_GRID:
            v = F_(Etr, Tsw(p), pi2).sum()
            if v > bestf:
                bestf, bestf_c = v, (ineg, ip, p)
    ineg, ip, p = bestf_c
    lldf = F_(Ecte[..., [ineg, ip]], Tsw(p), pi2).sum()
    return dict(d1=float(llc - lld), d1fit=float(llc - lldf), n=int(Edte.shape[0] * Edte.shape[1]),
                x0=float(X.X_GRID[ip]), psw=float(p), best_ou=best_c)


def job(args):
    model, aligned, seed, ts, kappa = args
    t0 = time.time()
    rng = np.random.default_rng([seed, MODEL_INDEX[model], 7, int(aligned)])
    res = [session(rng, C.MODELS[model], ts, kappa, aligned) for _ in range(N_SESS)]
    n = sum(r["n"] for r in res)
    out = dict(model=model, aligned=aligned, seed=seed,
               D1=1000.0 * sum(r["d1"] for r in res) / n, D1_fit=1000.0 * sum(r["d1fit"] for r in res) / n,
               x0_median=float(np.median([r["x0"] for r in res])))
    print(time.strftime("%H:%M:%S"), out, round(time.time() - t0, 1), "s", flush=True)
    return out


if __name__ == "__main__":
    cal = json.load(open("results/calibration.json"))
    jobs = [(m, al, s, cal["model_timescale"][m], cal["kappa"]) for s in SEEDS
            for m, al in (("M0_markov", False), ("M0_markov", True), ("M2_cont_unimodal", False), ("M3_cont_bimodal", False))]
    with Pool(2) as pool:
        res = list(pool.imap_unordered(job, jobs))
    json.dump(res, open("results/explore_disc.json", "w"), indent=1)
    print("done explore_disc")
