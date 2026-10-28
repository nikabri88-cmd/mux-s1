"""Aggregate MUX-S1 runs and evaluate the pre-registered hypotheses H1-H7."""
import glob
import json
from collections import defaultdict

import numpy as np

TGT = json.load(open("digitize/targets.json"))
PUB_ACF = dict(zip(TGT["digitized_fig2d_llr_acf"]["lags_ms"], TGT["digitized_fig2d_llr_acf"]["acf"]))


def load(part):
    out = defaultdict(list)
    for f in sorted(glob.glob(f"results/part{part}/*.json")):
        o = json.load(open(f))
        out[o["model"]].append(o)
    for m in out:
        out[m].sort(key=lambda o: o["seed"])
    return out


def rebound(acf_bins):
    a = np.asarray(acf_bins)
    return float(a[11:21].max() - a[5:11].min())       # max over 110-200 ms minus min over 50-100 ms


def pub_rebound():
    mx = max(v for k, v in PUB_ACF.items() if 110 <= k <= 200)
    mn = min(v for k, v in PUB_ACF.items() if 50 <= k <= 100)
    return mx - mn


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def count(xs, cond):
    return int(sum(1 for x in xs if cond(x)))


def describe_part1(P1):
    rows = {}
    for m, runs in P1.items():
        g = lambda f: [f(o) for o in runs]
        rows[m] = dict(
            n=len(runs),
            battery_all4=count(runs, lambda o: o["battery"]["n_passed"] == 4),
            battery=dict(reconstruction=g(lambda o: o["battery"]["reconstruction"]),
                         coordinated=g(lambda o: o["battery"]["coordinated"]),
                         reversal=g(lambda o: o["battery"]["reversal"]),
                         likelihood=g(lambda o: o["battery"]["likelihood"])),
            occ_prime_face=g(lambda o: o["occ"]["prime_face"]), occ_prime_ten=g(lambda o: o["occ"]["prime_ten"]),
            occ_bino=g(lambda o: 0.5 * (o["occ"]["bino_face"] + o["occ"]["bino_ten"])),
            acf_D=g(lambda o: o["acf"]["D"]), acf10=g(lambda o: o["acf"]["acf_10ms"]),
            acf_zero=g(lambda o: o["acf"]["zero_cross_ms"]), acf_tail=g(lambda o: o["acf"]["mean_110_200"]),
            acf_rebound=g(lambda o: rebound(o["acf"]["acf_a"])),
            dwell1=g(lambda o: o["dwell"]["median_dwell_state1_ms"]), dwell2=g(lambda o: o["dwell"]["median_dwell_state2_ms"]),
            lik_mix_mean=g(lambda o: o["lik_mix"]["mean"]), lik_mix_fracpos=g(lambda o: o["lik_mix"]["frac_pos"]),
            lik_held_mean=g(lambda o: o["lik_held"]["mean"]), lik_held_p=g(lambda o: o["lik_held"]["p_one"]),
            ev_circ=g(lambda o: o["recon_circular"]["ev"]), ev_coord=g(lambda o: o["recon_coordinated"]["ev"]),
            ev_twofold=g(lambda o: o["recon_coordinated_twofold"]["ev"]),
            slope_1h=g(lambda o: o["modulation"]["fig1h_tls_slope_avg"]),
            slope_s1=g(lambda o: o["modulation"]["fig4g_tls_slope_s1"]), slope_s2=g(lambda o: o["modulation"]["fig4g_tls_slope_s2"]),
            r_avg=g(lambda o: o["modulation"]["fig4g_r_avg"]), r_s1=g(lambda o: o["modulation"]["fig4g_r_s1"]),
            r_s2=g(lambda o: o["modulation"]["fig4g_r_s2"]),
            gain_ratio=g(lambda o: o["gain"]["gain_ratio_means"]), gain_ratio_median=g(lambda o: o["gain"]["gain_ratio_median"]),
            r_gain=g(lambda o: o["gain"]["r_gain"]), decoder_all=g(lambda o: o["gain"]["state_decoder_acc"]["all"]),
            decoder_500=g(lambda o: o["gain"]["state_decoder_acc"]["500"]),
            s1c_over_prime=g(lambda o: o["gain"]["ratio_S1C_over_prime"]),
            s2s_over_prime=g(lambda o: o["gain"]["ratio_S2S_over_prime"]),
            ratio_early=g(lambda o: o["gain"]["ratio_early"]), ratio_late=g(lambda o: o["gain"]["ratio_late"]),
            frac_s2_higher=g(lambda o: o["gain"]["frac_cells_mean_rate_S2_higher"]),
            n_cells=g(lambda o: o["gain"]["n_cells"]))
    return rows


def describe_part2(P2):
    rows = {}
    for m, runs in P2.items():
        g = lambda f: [f(o) for o in runs]
        rows[m] = dict(n=len(runs), D=g(lambda o: o["acf"]["D"]), acf10=g(lambda o: o["acf"]["acf_10ms"]),
                       tail=g(lambda o: o["acf"]["mean_110_200"]), rebound=g(lambda o: rebound(o["acf"]["acf_a"])),
                       zero=g(lambda o: o["acf"]["zero_cross_ms"]),
                       dwell1=g(lambda o: o["dwell"]["median_dwell_state1_ms"]),
                       dwell2=g(lambda o: o["dwell"]["median_dwell_state2_ms"]),
                       cv=g(lambda o: o["dwell"]["dwell_cv_edges_excluded"]),
                       disc=g(lambda o: o["discreteness_mnats_per_bin"]), k4=g(lambda o: o["k4"]),
                       od=g(lambda o: o["overdispersion"]), hyst=g(lambda o: o["hysteresis_r"]),
                       occ_prime=g(lambda o: o["occ_prime"]), occ_bino=g(lambda o: o["occ_bino"]),
                       acf_mean=np.mean([o["acf"]["acf_a"] for o in runs], axis=0).tolist(),
                       acf_sd=np.std([o["acf"]["acf_a"] for o in runs], axis=0).tolist())
    return rows


def verdicts(d1, d2):
    V = {}
    if d1:
        V["H1"] = {m: d1[m]["battery_all4"] for m in ("M0_markov", "M1_rhythm", "M3_cont_bimodal") if m in d1}
        V["H1"]["holds"] = all(v >= 4 for v in V["H1"].values())
        if "M2_cont_unimodal" in d1:
            b = d1["M2_cont_unimodal"]["battery"]
            three = sum(1 for i in range(len(b["reversal"])) if b["reconstruction"][i] and b["coordinated"][i] and b["reversal"][i])
            lik_fail = sum(1 for x in b["likelihood"] if not x)
            V["H2"] = dict(three_tests_passed=three, likelihood_failed=lik_fail, holds=three >= 4 and lik_fail >= 4)
        h3 = {}
        for m in ("C1_averaging", "C2_uncoordinated"):
            if m in d1:
                b = d1[m]["battery"]
                fails = [4 - sum(b[k][i] for k in ("reconstruction", "coordinated", "reversal", "likelihood")) for i in range(d1[m]["n"])]
                h3[m] = dict(fails=fails, n_datasets_ge3=sum(1 for f in fails if f >= 3))
        V["H3"] = dict(h3, holds=all(v["n_datasets_ge3"] >= 4 for v in h3.values()))
        if all(m in d1 for m in ("M0_markov", "M4_adaptation", "M5_conscious_gain")):
            a4, a0, a5 = d1["M4_adaptation"], d1["M0_markov"], d1["M5_conscious_gain"]
            c1 = count(a4["gain_ratio"], lambda x: 1.10 <= x <= 1.40)
            c2 = count(a4["decoder_all"], lambda x: x >= 0.60)
            c3 = count(a4["r_gain"], lambda x: x >= 0.8)
            c4 = count(a0["gain_ratio"], lambda x: 0.95 <= x <= 1.05)
            c5 = count(a0["decoder_all"], lambda x: x <= 0.56)
            diff = [x5 - x4 for x5, x4 in zip(a5["s1c_over_prime"], a4["s1c_over_prime"])]
            c6 = count(diff, lambda x: x >= 0.10)
            el4 = [e - l for e, l in zip(a4["ratio_early"], a4["ratio_late"])]
            el5 = [e - l for e, l in zip(a5["ratio_early"], a5["ratio_late"])]
            c7 = count(el4, lambda x: x >= 0.05)
            c8 = count(el5, lambda x: abs(x) <= 0.03)
            parts = dict(M4_ratio_in_range=c1, M4_decoder_ge_060=c2, M4_rgain_ge_08=c3, M0_ratio_in_range=c4,
                         M0_decoder_le_056=c5, M5_minus_M4_s1c_prime_ge_010=c6, M4_early_minus_late_ge_005=c7,
                         M5_early_minus_late_abs_le_003=c8, diff_s1c_prime=diff, early_late_M4=el4, early_late_M5=el5)
            V["H6"] = dict(parts, holds_adaptation_reproduces=(c1 >= 4 and c2 >= 4 and c3 >= 4),
                           holds_controls=(c4 >= 4 and c5 >= 4),
                           holds_discriminators=(c6 >= 4 and c7 >= 4 and c8 >= 4))
    if d2:
        V["H4"] = {m: d2[m]["rebound"] for m in ("M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal") if m in d2}
        V["H4_published_rebound"] = pub_rebound()
        ok = True
        if "M1_rhythm" in d2:
            ok &= count(d2["M1_rhythm"]["rebound"], lambda x: x > 0) >= 8
        for m in ("M0_markov", "M2_cont_unimodal", "M3_cont_bimodal"):
            if m in d2:
                ok &= count(d2[m]["rebound"], lambda x: x < 0) >= 8
        V["H4_holds"] = bool(ok)
        if "M0_markov" in d2:
            c = count(d2["M0_markov"]["tail"], lambda x: -0.116 <= x <= -0.029)
            V["H5"] = dict(in_range=c, values=d2["M0_markov"]["tail"], holds=c >= 8)
        h7 = {}
        if "M2_cont_unimodal" in d2 and "M0_markov" in d2:
            h7["disc_M2_vs_M0"] = auc(d2["M2_cont_unimodal"]["disc"], d2["M0_markov"]["disc"])
        if "M1_rhythm" in d2 and "M0_markov" in d2:
            h7["cv_M1_lower_than_M0"] = auc([-x for x in d2["M1_rhythm"]["cv"]], [-x for x in d2["M0_markov"]["cv"]])
        if "M6_heterogeneous" in d2 and "M0_markov" in d2:
            h7["od_M6_vs_M0"] = auc(d2["M6_heterogeneous"]["od"], d2["M0_markov"]["od"])
        V["H7"] = dict(h7, holds=all(v >= 0.9 for v in h7.values()) and len(h7) == 3)
        ex = {}
        if "M3_cont_bimodal" in d2 and "M0_markov" in d2:
            ex["disc_M3_vs_M0"] = auc(d2["M3_cont_bimodal"]["disc"], d2["M0_markov"]["disc"])
            ex["k4_M3_vs_M0"] = auc(d2["M3_cont_bimodal"]["k4"], d2["M0_markov"]["k4"])
            ex["k4_M2_vs_M0"] = auc(d2["M2_cont_unimodal"]["k4"], d2["M0_markov"]["k4"]) if "M2_cont_unimodal" in d2 else None
        if "M7_ring" in d2 and "M0_markov" in d2:
            ex["hyst_M7_vs_M0"] = auc([x if x is not None else -1 for x in d2["M7_ring"]["hyst"]],
                                      [x if x is not None else -1 for x in d2["M0_markov"]["hyst"]])
        V["H7_exploratory"] = ex
    return V


def fmt(v):
    return f"{np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]"


if __name__ == "__main__":
    P1, P2 = load("I"), load("II")
    d1, d2 = describe_part1(P1), describe_part2(P2)
    V = verdicts(d1, d2)
    json.dump(dict(part1=d1, part2=d2, verdicts=V, published_rebound=pub_rebound()),
              open("results/summary.json", "w"), indent=1, default=float)
    print("PART I")
    for m, r in d1.items():
        print(f"{m:24s} n={r['n']} all4={r['battery_all4']} rec={sum(r['battery']['reconstruction'])} "
              f"coord={sum(r['battery']['coordinated'])} rev={sum(r['battery']['reversal'])} lik={sum(r['battery']['likelihood'])}")
        print(f"   floor face {fmt(r['occ_prime_face'])} ten {fmt(r['occ_prime_ten'])} bino {fmt(r['occ_bino'])} | D {fmt(r['acf_D'])} acf10 {fmt(r['acf10'])} tail {fmt(r['acf_tail'])} reb {fmt(r['acf_rebound'])}")
        print(f"   dwell1 {fmt(r['dwell1'])} dwell2 {fmt(r['dwell2'])} | likmix {fmt(r['lik_mix_mean'])} fracpos {fmt(r['lik_mix_fracpos'])} | likheld {fmt(r['lik_held_mean'])}")
        print(f"   EV circ C_S1 {fmt([e['C_S1'] for e in r['ev_circ']])} S_S1 {fmt([e['S_S1'] for e in r['ev_circ']])} S_S2 {fmt([e['S_S2'] for e in r['ev_circ']])} C_S2 {fmt([e['C_S2'] for e in r['ev_circ']])} prime {fmt([e['prime'] for e in r['ev_circ']])}")
        print(f"   EV coord C_S1 {fmt([e['C_S1'] for e in r['ev_coord']])} S_S1 {fmt([e['S_S1'] for e in r['ev_coord']])} S_S2 {fmt([e['S_S2'] for e in r['ev_coord']])} C_S2 {fmt([e['C_S2'] for e in r['ev_coord']])}")
        print(f"   slopes 1h {fmt(r['slope_1h'])} s1 {fmt(r['slope_s1'])} s2 {fmt(r['slope_s2'])} | r avg {fmt(r['r_avg'])} s1 {fmt(r['r_s1'])} s2 {fmt(r['r_s2'])}")
        print(f"   gain ratio {fmt(r['gain_ratio'])} r {fmt(r['r_gain'])} dec all {fmt(r['decoder_all'])} 500 {fmt(r['decoder_500'])} | S1C/prime {fmt(r['s1c_over_prime'])} early {fmt(r['ratio_early'])} late {fmt(r['ratio_late'])} fracS2hi {fmt(r['frac_s2_higher'])}")
    print("PART II")
    for m, r in d2.items():
        print(f"{m:24s} n={r['n']} D {fmt(r['D'])} acf10 {fmt(r['acf10'])} tail {fmt(r['tail'])} reb {fmt(r['rebound'])} dwell {fmt(r['dwell1'])} cv {fmt(r['cv'])}")
        print(f"   disc {fmt(r['disc'])} k4 {fmt(r['k4'])} od {fmt(r['od'])} hyst {fmt([h if h is not None else np.nan for h in r['hyst']])} floor {fmt(r['occ_prime'])}")
    print(json.dumps(V, indent=1, default=float))
