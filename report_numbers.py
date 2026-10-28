"""Print every number quoted in MUX-S1_report.md, from results/summary.json and the exploratory files."""
import json
import os

import numpy as np

S = json.load(open("results/summary.json"))
P1, P2, V = S["part1"], S["part2"], S["verdicts"]


def r(v, d=3):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def mm(v, d=3):
    v = np.array([x for x in v if x is not None], float)
    return f"{r(v.mean(), d)} [{r(v.min(), d)}; {r(v.max(), d)}]"


print("== Part I battery (passes / n)")
for m, x in P1.items():
    b = x["battery"]
    print(f"{m:24s} rec {sum(b['reconstruction'])}/{x['n']} coord {sum(b['coordinated'])} rev {sum(b['reversal'])} "
          f"lik {sum(b['likelihood'])} all4 {x['battery_all4']} | lik_held x1e4 {mm([1e4 * v for v in x['lik_held_mean']], 2)} "
          f"| mean>0: {sum(1 for v in x['lik_held_mean'] if v > 0)} | lik_mix {mm(x['lik_mix_mean'])} fracpos {mm(x['lik_mix_fracpos'], 2)}")
print("\n== Part I timescale / amplitude")
for m, x in P1.items():
    print(f"{m:24s} D {mm(x['acf_D'], 1)} acf10 {mm(x['acf10'])} zero {mm(x['acf_zero'], 0)} tail {mm(x['acf_tail'])} "
          f"dwell1 {mm(x['dwell1'], 0)} dwell2 {mm(x['dwell2'], 0)} prime_face {mm(x['occ_prime_face'])} prime_ten {mm(x['occ_prime_ten'])} bino {mm(x['occ_bino'])}")
print("\n== Part I reconstruction EV (circular / coordinated)")
for m, x in P1.items():
    ec, eo = x["ev_circ"], x["ev_coord"]
    print(f"{m:24s} circ C_S1 {mm([e['C_S1'] for e in ec], 2)} S_S2 {mm([e['S_S2'] for e in ec], 2)} "
          f"| coord C_S1 {mm([e['C_S1'] for e in eo], 2)} S_S1 {mm([e['S_S1'] for e in eo], 2)} S_S2 {mm([e['S_S2'] for e in eo], 2)} C_S2 {mm([e['C_S2'] for e in eo], 2)} prime {mm([e['prime'] for e in eo], 2)}")
print("\n== Part I modulation")
for m, x in P1.items():
    print(f"{m:24s} slope1h {mm(x['slope_1h'], 2)} s1 {mm(x['slope_s1'], 2)} s2 {mm(x['slope_s2'], 2)} r_avg {mm(x['r_avg'], 2)} r_s1 {mm(x['r_s1'], 2)} r_s2 {mm(x['r_s2'], 2)}")
print("\n== Part I gain")
for m, x in P1.items():
    el = [e - l for e, l in zip(x["ratio_early"], x["ratio_late"])]
    print(f"{m:24s} ratio {mm(x['gain_ratio'])} r_gain {mm(x['r_gain'], 2)} dec_all {mm(x['decoder_all'], 2)} dec500 {mm(x['decoder_500'], 2)} "
          f"S1C/prime {mm(x['s1c_over_prime'], 2)} S2S/prime {mm(x['s2s_over_prime'], 2)} early-late {mm(el, 3)} fracS2hi {mm(x['frac_s2_higher'], 2)} ncell {mm(x['n_cells'], 0)}")
print("\n== Part II")
for m, x in P2.items():
    print(f"{m:24s} D {mm(x['D'], 1)} acf10 {mm(x['acf10'])} tail {mm(x['tail'])} rebound {mm(x['rebound'])} (>0: {sum(1 for v in x['rebound'] if v > 0)}/{x['n']}) "
          f"zero {mm(x['zero'], 0)} dwell {mm(x['dwell1'], 0)} cv {mm(x['cv'], 2)} prime {mm(x['occ_prime'])}")
    print(f"{'':24s} disc {mm(x['disc'], 3)} k4 {mm(x['k4'], 2)} od {mm(x['od'], 2)} hyst {mm(x['hyst'], 3)}")
print("\n== ACF curves: published vs part II means (lags 10..200)")
T = json.load(open("digitize/targets.json"))["digitized_fig2d_llr_acf"]
pub = dict(zip(T["lags_ms"], T["acf"]))
lags = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
print("lag   pub   " + " ".join(f"{m[:8]:>8s}" for m in P2))
for L in lags:
    print(f"{L:3d} {pub[L]:6.3f} " + " ".join(f"{P2[m]['acf_mean'][L // 10]:8.3f}" for m in P2))
print("\n== Verdicts")
print(json.dumps(V, indent=1, default=float)[:6000])
if os.path.exists("results/explore_snr_summary.json"):
    print("\n== E3")
    E = json.load(open("results/explore_snr_summary.json"))
    for k, x in E.items():
        print(f"{k:24s} prime {mm(x['occ_prime'])} acf10 {mm(x['acf10'])} D {mm(x['D'], 1)} dwell {mm(x['dwell1'], 0)} "
              f"lik x1e4 {mm([1e4 * v for v in x['lik_held']], 2)} pass {sum(x['lik_held_pass'])}/{x['n']} frac+ {mm(x['lik_held_fracpos'], 2)}")
if os.path.exists("results/explore_accuracy.json"):
    print("\n== E3b")
    for k, x in json.load(open("results/explore_accuracy.json")).items():
        print(k, {kk: r(v) for kk, v in x.items()})
