"""Pre-registered calibration (MUX-S1 prereg section 6): kappa from the priming floor, then the
timescale of each dynamics family from the authors' ACF time constant D = 67 ms. DYN design,
separate seed stream ("CAL")."""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from mux_s1 import runner as RN

TARGET_FLOOR = 0.229
TARGET_D = 67.0
KAPPA_GRID = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]   # extended downward: deviation 1
FAMILIES = {
    "telegraph": ("M0_markov", [40.0, 50.0, 60.0, 70.0, 80.0]),
    "gamma6": ("M1_rhythm", [40.0, 50.0, 60.0, 70.0, 80.0]),
    "gamma3": ("M1b_rhythm_k3", [40.0, 50.0, 60.0, 70.0, 80.0]),
    "ou1": ("M2_cont_unimodal", [20.0, 27.0, 35.0, 45.0, 60.0]),
    "ou25": ("M3_cont_bimodal", [20.0, 30.0, 45.0, 65.0, 90.0]),
    "ring": ("M7_ring", [40.0, 50.0, 60.0, 70.0, 80.0]),
}


def job(args):
    model, ts, kappa, seed, n_sess = args
    o = RN.part2_dataset(model, seed, ts, kappa, n_sess=n_sess, part="CAL")
    return dict(model=model, ts=ts, kappa=kappa, seed=seed, n_sess=n_sess, D=o["acf"]["D"],
                occ_prime=o["occ_prime"], occ_bino=o["occ_bino"], acf10=o["acf"]["acf_10ms"],
                dwell1=o["dwell"]["median_dwell_state1_ms"], dwell2=o["dwell"]["median_dwell_state2_ms"])


def interp_root(xs, ys, target):
    xs, ys = np.asarray(xs), np.asarray(ys)
    for i in range(len(xs) - 1):
        if (ys[i] - target) * (ys[i + 1] - target) <= 0:
            return float(xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
    # outside the grid: linear extrapolation from the nearest end
    i = 0 if abs(ys[0] - target) < abs(ys[-1] - target) else len(xs) - 2
    return float(xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))


if __name__ == "__main__":
    t0 = time.time()
    log = {"kappa_grid": [], "timescale_grid": {}, "verify": {}}
    with Pool(2) as pool:
        # 1. kappa (priming floor does not depend on dynamics; M0 with D = 60 ms)
        res = pool.map(job, [("M0_markov", 60.0, k, 1000 + i, 6) for i, k in enumerate(KAPPA_GRID)])
        log["kappa_grid"] = res
        floors = [r["occ_prime"] for r in res]
        # the floor decreases with kappa
        kappa = interp_root(KAPPA_GRID, floors, TARGET_FLOOR)
        chk = pool.map(job, [("M0_markov", 60.0, kappa, 2000 + i, 6) for i in range(2)])
        log["kappa"] = kappa
        log["kappa_check_floor"] = float(np.mean([c["occ_prime"] for c in chk]))
        print("kappa grid", [(k, round(f, 4)) for k, f in zip(KAPPA_GRID, floors)], flush=True)
        print("kappa", round(kappa, 4), "floor check", round(log["kappa_check_floor"], 4), flush=True)
        if abs(log["kappa_check_floor"] - TARGET_FLOOR) > 0.02:
            json.dump(log, open("results/calibration_log_failed.json", "w"), indent=1, default=float)
            sys.exit("kappa check outside +-2 pp")
        # 2. timescales
        jobs = []
        for fam, (model, grid) in FAMILIES.items():
            for i, ts in enumerate(grid):
                jobs.append((model, ts, kappa, 3000 + 10 * list(FAMILIES).index(fam) + i, 8))
        res = pool.map(job, jobs)
        ts_cal = {}
        for fam, (model, grid) in FAMILIES.items():
            rr = [r for r in res if r["model"] == model]
            rr.sort(key=lambda r: r["ts"])
            log["timescale_grid"][fam] = rr
            ts_cal[fam] = interp_root([r["ts"] for r in rr], [r["D"] for r in rr], TARGET_D)
            print(fam, [(r["ts"], round(r["D"], 1)) for r in rr], "->", round(ts_cal[fam], 2), flush=True)
        # 3. verification with 19 sessions, one secant refinement if off by more than 3 ms
        ver = pool.map(job, [(FAMILIES[f][0], ts_cal[f], kappa, 4000 + i, 19) for i, f in enumerate(FAMILIES)])
        for f, v in zip(FAMILIES, ver):
            log["verify"][f] = [v]
        redo = [f for f, v in zip(FAMILIES, ver) if abs(v["D"] - TARGET_D) > 3.0]
        if redo:
            new = {}
            for f in redo:
                rr = log["timescale_grid"][f]
                xs = np.array([r["ts"] for r in rr]); ys = np.array([r["D"] for r in rr])
                slope = np.polyfit(xs, ys, 1)[0]
                v = log["verify"][f][0]
                new[f] = float(ts_cal[f] + (TARGET_D - v["D"]) / slope)
            ver2 = pool.map(job, [(FAMILIES[f][0], new[f], kappa, 5000 + i, 19) for i, f in enumerate(redo)])
            for f, v in zip(redo, ver2):
                log["verify"][f].append(v)
                ts_cal[f] = new[f]
    log["timescale"] = ts_cal
    log["seconds"] = round(time.time() - t0, 1)
    final = {"kappa": kappa, "timescale": ts_cal,
             "model_timescale": {
                 "M0_markov": ts_cal["telegraph"], "M4_adaptation": ts_cal["telegraph"],
                 "M4c_adaptation_tuning": ts_cal["telegraph"], "M5_conscious_gain": ts_cal["telegraph"],
                 "M6_heterogeneous": ts_cal["telegraph"], "C2_uncoordinated": ts_cal["telegraph"],
                 "C1_averaging": ts_cal["telegraph"], "M1_rhythm": ts_cal["gamma6"],
                 "M1b_rhythm_k3": ts_cal["gamma3"], "M2_cont_unimodal": ts_cal["ou1"],
                 "M3_cont_bimodal": ts_cal["ou25"], "M7_ring": ts_cal["ring"]}}
    json.dump(log, open("results/calibration_log.json", "w"), indent=1, default=float)
    json.dump(final, open("results/calibration.json", "w"), indent=1)
    print(json.dumps(final, indent=1))
    print("verify:", {f: [(round(v["D"], 1), round(v["occ_prime"], 3), round(v["acf10"], 3)) for v in vv] for f, vv in log["verify"].items()})
