"""Summary of the post hoc exploratory analysis E3 (explore_snr.py). Descriptive only."""
import glob
import json
from collections import defaultdict

import numpy as np

rows = defaultdict(list)
for f in sorted(glob.glob("results/explore_snr/*.json")):
    o = json.load(open(f))
    rows[(o["model"], o["mult"])].append(o)

out = {}
print(f"{'model':18s} {'k x':>4s} {'n':>2s} {'kappa':>6s} {'prime':>6s} {'acf10':>6s} {'D':>6s} {'dwell':>6s} "
      f"{'lik_held 1e-4':>14s} {'pass':>5s} {'frac+':>6s}")
for (m, k), runs in sorted(rows.items()):
    g = lambda f: [f(o) for o in runs]
    r = dict(n=len(runs), kappa=runs[0]["kappa"], occ_prime=g(lambda o: o["occ_prime"]),
             acf10=g(lambda o: o["acf"]["acf_10ms"]), D=g(lambda o: o["acf"]["D"]),
             tail=g(lambda o: o["acf"]["mean_110_200"]),
             dwell1=g(lambda o: o["dwell"]["median_dwell_state1_ms"]),
             dwell2=g(lambda o: o["dwell"]["median_dwell_state2_ms"]),
             lik_held=g(lambda o: o["lik_held"]["mean"]), lik_held_p=g(lambda o: o["lik_held"]["p_one"]),
             lik_held_pass=g(lambda o: o["lik_held"]["passed"]), lik_held_fracpos=g(lambda o: o["lik_held"]["frac_pos"]),
             lik_mix=g(lambda o: o["lik_mix"]["mean"]))
    out[f"{m}|{k}"] = r
    print(f"{m:18s} {k:4.1f} {r['n']:2d} {r['kappa']:6.3f} {np.mean(r['occ_prime']):6.3f} {np.mean(r['acf10']):6.3f} "
          f"{np.mean(r['D']):6.1f} {np.mean(r['dwell1']):6.1f} {np.mean(r['lik_held']) * 1e4:14.3f} "
          f"{sum(r['lik_held_pass'])}/{r['n']:<3d} {np.mean(r['lik_held_fracpos']):6.2f}")
json.dump(out, open("results/explore_snr_summary.json", "w"), indent=1)
