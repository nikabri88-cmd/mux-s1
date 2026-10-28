"""E3b (post hoc, exploratory, NOT pre-registered): how often are the inferred state labels right?
Accuracy of LLR-sign labels (all neurons) and of HMM Viterbi labels (all neurons; one half of the
neurons) against the true latent state (m-bar > 0.5), in the DYN design (25 pairs x 15 trials per
dominance condition, templates from 15 priming trials of each face, 300 neurons), at the calibrated
kappa and at 1.5 x kappa. Bins with 0.45 < m-bar < 0.55 are excluded from the accuracy.
Usage: python explore_accuracy.py
"""
import json

import numpy as np

from mux_s1 import core as C
from mux_s1 import experiments as X
from mux_s1.runner import MODEL_INDEX

MODELS = ["M0_markov", "M1_rhythm", "M3_cont_bimodal"]
MULTS = [1.0, 1.5]
N_SESS = 4


def session(rng, spec, ts, kappa, N=300, n_pairs=25, n_tr=15):
    pop = C.Population(rng, N, kappa)
    F = C.make_faces(rng, 2 * n_pairs)
    D = pop.delta(F)
    half = np.sort(rng.permutation(N)[:N // 2])
    acc = {"llr_all": [], "hmm_all": [], "hmm_half": []}
    E_all, E_half, truth = [], [], []
    for p in range(n_pairs):
        a, b = 2 * p, 2 * p + 1
        lamA = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[a], n_tr)))
        lamB = X.template(C.poisson(rng, C.priming_rates(spec, pop, D[b], n_tr)))
        lam, m = C.binocular_rates(rng, spec, pop, D[a], D[b], n_tr, ts, 1000.0)   # a conscious
        cnt = C.poisson(rng, lam)
        l = C.llr(cnt, lamA, lamB)
        true = m > 0.5
        clear = np.abs(m - 0.5) > 0.05
        acc["llr_all"].append(((l > 0) == true)[clear].mean())
        E_all.append(C.emis_loglik(cnt, [lamA, lamB]))
        E_half.append(C.emis_loglik(cnt[..., half], [lamA[half], lamB[half]]))
        truth.append((true, clear))
    for key, E in (("hmm_all", E_all), ("hmm_half", E_half)):
        Es = np.stack(E)                                   # (P, M, T, 2), state 0 = face a
        z = C.viterbi_batch(Es, C.hmm_fit_batch(Es))
        for zz, (true, clear) in zip(z, truth):
            acc[key].append(((zz == 0) == true)[clear].mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    cal = json.load(open("results/calibration.json"))
    out = {}
    for m in MODELS:
        for k in MULTS:
            rng = np.random.default_rng([0, MODEL_INDEX[m], 5, int(round(k * 100))])
            res = [session(rng, C.MODELS[m], cal["model_timescale"][m], cal["kappa"] * k) for _ in range(N_SESS)]
            out[f"{m}|{k}"] = {key: float(np.mean([r[key] for r in res])) for key in res[0]}
            print(m, k, {key: round(v, 3) for key, v in out[f"{m}|{k}"].items()}, flush=True)
    json.dump(out, open("results/explore_accuracy.json", "w"), indent=1)
