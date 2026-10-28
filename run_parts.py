"""Main runs of MUX-S1 (prereg section 10). Usage: python run_parts.py I|II"""
import json
import os
import sys
import time
from multiprocessing import Pool

from mux_s1 import runner as RN

PART1 = [(m, s) for m in ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal", "M4_adaptation",
                          "M5_conscious_gain", "C1_averaging", "C2_uncoordinated"] for s in range(5)] + \
        [("M4c_adaptation_tuning", s) for s in range(3)]
PART2 = [(m, s) for m in ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal",
                          "M6_heterogeneous", "M7_ring"] for s in range(10)]


def job(args):
    part, model, seed, ts, kappa = args
    out = f"results/part{part}/{model}_{seed}.json"
    if os.path.exists(out):
        return out
    o = RN.part1_dataset(model, seed, ts, kappa) if part == "I" else RN.part2_dataset(model, seed, ts, kappa)
    with open(out + ".tmp", "w") as f:
        f.write(RN.to_json(o))
    os.replace(out + ".tmp", out)
    print(time.strftime("%H:%M:%S"), part, model, seed, o["seconds"], "s", flush=True)
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    cal = json.load(open("results/calibration.json"))
    os.makedirs(f"results/part{part}", exist_ok=True)
    lst = PART1 if part == "I" else PART2
    jobs = [(part, m, s, cal["model_timescale"][m], cal["kappa"]) for m, s in lst]
    # interleave models so partial results cover all models
    jobs.sort(key=lambda j: (j[2], j[1]))
    with Pool(2) as pool:
        for _ in pool.imap_unordered(job, jobs):
            pass
    print("done", part)
