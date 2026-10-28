"""E3 (post hoc, exploratory, NOT pre-registered; added after part I showed that the authors' own model
barely passes the held-out likelihood test at the calibrated tuning strength).

Question: how do the held-out likelihood test and the published consistency numbers (priming false-state
fraction, ACF at 10 ms, HMM median dwell) depend on the per-neuron tuning strength kappa?
Design: TEN sessions only (19 per dataset, as the 19-session t-test of the battery), timescales frozen
at their calibrated values, kappa multiplied by 1.0, 1.2, 1.5, 2.0 (1.2 ~ the kappa that would give 22.9% with TEN-style templates pooled over pairs). Separate RNG stream (tag 4).
Usage: python explore_snr.py
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from mux_s1 import analysis as A
from mux_s1 import core as C
from mux_s1 import experiments as X
from mux_s1.runner import MODEL_INDEX, to_json

MODELS = ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal", "M4_adaptation"]
MULTS = [1.0, 1.2, 1.5, 2.0]
SEEDS = [0, 1]
N_SESS = 19


def job(args):
    model, mult, seed, ts, kappa = args
    out = f"results/explore_snr/{model}_k{int(round(mult * 100))}_{seed}.json"
    if os.path.exists(out):
        return out
    t0 = time.time()
    rng = np.random.default_rng([seed, MODEL_INDEX[model], 4, int(round(mult * 100))])
    spec = C.MODELS[model]
    ss = [X.ten_session(rng, spec, ts, kappa * mult) for _ in range(N_SESS)]
    held = [float(np.mean(s["lik_held"])) for s in ss]
    mix = [float(np.mean(s["lik_mix"])) for s in ss]
    o = dict(model=model, mult=mult, seed=seed, kappa=kappa * mult, timescale=ts)
    o["occ_prime"] = float(np.mean([s["occ_prime"] for s in ss]))
    o["occ_bino"] = float(np.mean([s["occ_bino"] for s in ss]))
    o["acf"] = A.acf_summary([s["llr_bino"] for s in ss])
    o["dwell"] = A.dwell_summary([s["runs"] for s in ss])
    o["lik_held"] = dict(zip(("mean", "t", "p_one"), A.session_t(held)), frac_pos=float(np.mean(np.array(held) > 0)),
                         sessions=held)
    o["lik_held"]["passed"] = bool(o["lik_held"]["mean"] > 0 and o["lik_held"]["p_one"] < 0.05)
    o["lik_mix"] = dict(zip(("mean", "t", "p_one"), A.session_t(mix)), frac_pos=float(np.mean(np.array(mix) > 0)))
    o["seconds"] = round(time.time() - t0, 1)
    with open(out + ".tmp", "w") as f:
        f.write(to_json(o))
    os.replace(out + ".tmp", out)
    print(time.strftime("%H:%M:%S"), model, mult, seed, o["seconds"], "s", flush=True)
    return out


if __name__ == "__main__":
    cal = json.load(open("results/calibration.json"))
    os.makedirs("results/explore_snr", exist_ok=True)
    jobs = [(m, k, s, cal["model_timescale"][m], cal["kappa"]) for s in SEEDS for k in MULTS for m in MODELS]
    with Pool(2) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("done explore_snr")
