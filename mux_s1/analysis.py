"""Dataset-level analyses for MUX-S1 (pooled across simulated sessions)."""
import numpy as np
from sklearn.svm import LinearSVC

from . import core as C

N_HR_FACES = 60          # faces of the 30 high-repeat pairs (indices 0..59)


# ----------------------------------------------------------------------------------------
# dynamics fingerprint (all designs)
# ----------------------------------------------------------------------------------------
def acf_summary(llr_list, max_lag=20):
    """llr_list: per-session arrays (n_tr, NB). ACF with per-trial mean subtraction (E-a) and
    global mean (E-b), averaged across sessions."""
    ea = np.mean([C.acf_per_trial_mean(x, max_lag) for x in llr_list], axis=0)
    eb = np.mean([C.acf_global_mean(x, max_lag) for x in llr_list], axis=0)
    D, rho, zc = C.fit_D(ea)
    lags = 10 * np.arange(max_lag + 1)
    sel_reb = (lags >= 100) & (lags <= 200)
    sel_tail = (lags >= 110) & (lags <= 200)
    return dict(acf_a=ea.tolist(), acf_b=eb.tolist(), D=D, rho=rho, zero_cross_ms=zc,
                max_100_200=float(ea[sel_reb].max()), mean_110_200=float(ea[sel_tail].mean()),
                acf_10ms=float(ea[1]), D_b=C.fit_D(eb)[0])


def dwell_summary(runs_list):
    out = {}
    for st, name in ((1, "state1"), (0, "state2")):
        r = np.concatenate([np.asarray(rr[st]) for rr in runs_list]) * 10.0
        out[f"median_dwell_{name}_ms"] = float(np.median(r))
        rx = np.concatenate([np.asarray(rr["ex"][st]) for rr in runs_list]) * 10.0
        out[f"median_dwell_{name}_ms_edges_excluded"] = float(np.median(rx)) if len(rx) else None
    rx = np.concatenate([np.asarray(rr["ex"][s]) for rr in runs_list for s in (0, 1)]) * 10.0
    out["dwell_cv_edges_excluded"] = float(rx.std() / rx.mean()) if len(rx) else None
    out["n_runs_edges_excluded"] = int(len(rx))
    return out


def session_t(x):
    """One-sample t statistic and one-sided p (mean > 0) across sessions."""
    from scipy import stats
    x = np.asarray(x, float)
    t = stats.ttest_1samp(x, 0.0)
    p_one = t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2
    return float(x.mean()), float(t.statistic), float(p_one)


# ----------------------------------------------------------------------------------------
# reconstruction (FACE1500)
# ----------------------------------------------------------------------------------------
def reconstruct(faces, prim, R, n_boot=2000, seed=0, K=500):
    """prim (n_faces, n_neur) priming responses; R (2 roles, 2 states, n_an, n_neur) state-resolved
    rates. Returns aggregated EVs for the 60 high-repeat faces, per combo, and bootstrap p-values
    of the four pass conditions (resampling the 30 pairs)."""
    dec = C.Decoder(prim, faces, K=K)
    idx = np.arange(N_HR_FACES)
    Y = faces[idx]
    preds = {}
    combos = {"C_S1": (0, 0), "C_S2": (0, 1), "S_S1": (1, 0), "S_S2": (1, 1)}
    Bs = [dec.weights_lofo(j) for j in idx]
    for name, (role, st) in combos.items():
        r = np.nan_to_num(R[role, st, :N_HR_FACES].astype(float), nan=0.0)
        preds[name] = np.vstack([dec.predict(r[j], Bs[j])[0] for j in idx])
    preds["prime"] = np.vstack([dec.predict(prim[j], Bs[j])[0] for j in idx])
    ev = {k: float(C.aggregate_ev(C.explained_variance(Y, v))) for k, v in preds.items()}
    ev_feat = {k: C.explained_variance(Y, v).tolist() for k, v in preds.items()}
    rng = np.random.default_rng(seed)
    fails = np.zeros(4)
    for _ in range(n_boot):
        pr = rng.integers(0, N_HR_FACES // 2, N_HR_FACES // 2)
        sel = np.concatenate([2 * pr, 2 * pr + 1])
        e = {k: C.aggregate_ev(C.explained_variance(Y[sel], preds[k][sel])) for k in ("C_S1", "S_S1", "S_S2", "C_S2")}
        fails += [e["C_S1"] <= 0, e["C_S1"] <= e["S_S1"], e["S_S2"] <= 0, e["S_S2"] <= e["C_S2"]]
    p = fails / n_boot
    return dict(ev=ev, ev_feat=ev_feat, p=p.tolist(), passed=bool(np.all(p < 0.05)))


# ----------------------------------------------------------------------------------------
# tuning gain, state decoder (FACE1500)
# ----------------------------------------------------------------------------------------
def gains_and_decoder(faces, prim, R_xv, n_an, seed=0, n_split=20):
    an = np.arange(n_an)
    loo, r2 = C.lofo_linear_predictions(faces, prim.astype(float))
    keep = r2 > 0.05
    proj = loo[an][:, keep]                                  # (n_an, n_keep)
    def g(role, st, wi):
        y = R_xv[role, st, wi][:, keep].astype(float)
        out = np.full(proj.shape[1], np.nan)
        for i in range(proj.shape[1]):
            ok = np.isfinite(y[:, i])
            if ok.sum() > 10:
                out[i] = C.ols_slope(proj[ok, i], y[ok, i])
        return out
    gS1C, gS2S = g(0, 0, 0), g(1, 1, 0)
    gS1S, gS2C = g(1, 0, 0), g(0, 1, 0)
    gprime = np.array([C.ols_slope(proj[:, i], prim[an][:, keep][:, i].astype(float)) for i in range(proj.shape[1])])
    eS1C, eS2S = g(0, 0, 1), g(1, 1, 1)
    lS1C, lS2S = g(0, 0, 2), g(1, 1, 2)
    ok = np.isfinite(gS1C) & np.isfinite(gS2S)
    ratio_means = float(np.nanmean(gS1C[ok]) / np.nanmean(gS2S[ok]))
    pos = ok & (gS2S > 0.05)
    med_ratio = float(np.median(gS1C[pos] / gS2S[pos])) if pos.sum() else None
    r_gain = float(np.corrcoef(gS1C[ok], gS2S[ok])[0, 1])
    # mean rate across faces in state1-conscious vs state2-suppressed
    mS1 = np.nanmean(R_xv[0, 0, 0][:, keep], axis=0); mS2 = np.nanmean(R_xv[1, 1, 0][:, keep], axis=0)
    frac_s2_higher = float(np.mean(mS2 > mS1))
    okp = np.isfinite(gprime) & ok
    res = dict(n_cells=int(keep.sum()), gain_ratio_means=ratio_means, gain_ratio_median=med_ratio,
               r_gain=r_gain, mean_gain_S1C=float(np.nanmean(gS1C)), mean_gain_S2S=float(np.nanmean(gS2S)),
               mean_gain_S1S=float(np.nanmean(gS1S)), mean_gain_S2C=float(np.nanmean(gS2C)),
               mean_gain_prime=float(np.nanmean(gprime)),
               ratio_S1C_over_prime=float(np.nanmean(gS1C[okp]) / np.nanmean(gprime[okp])),
               ratio_S2S_over_prime=float(np.nanmean(gS2S[okp]) / np.nanmean(gprime[okp])),
               ratio_early=float(np.nanmean(eS1C) / np.nanmean(eS2S)),
               ratio_late=float(np.nanmean(lS1C) / np.nanmean(lS2S)),
               frac_cells_mean_rate_S2_higher=frac_s2_higher,
               mean_rate_S1C=float(np.nanmean(mS1)), mean_rate_S2S=float(np.nanmean(mS2)))
    # order-concatenation state decoder (all neurons)
    V1 = np.nan_to_num(R_xv[0, 0, 0].astype(float)); V2 = np.nan_to_num(R_xv[1, 1, 0].astype(float))
    mu = np.concatenate([V1, V2]).mean(0); sd = np.concatenate([V1, V2]).std(0) + 1e-6
    V1 = (V1 - mu) / sd; V2 = (V2 - mu) / sd
    rng = np.random.default_rng(seed)
    pairs = np.arange(n_an // 2)
    accs = {}
    for n_sub in (500, None):
        a_list = []
        for _ in range(n_split):
            cols = np.arange(V1.shape[1]) if n_sub is None else rng.choice(V1.shape[1], min(n_sub, V1.shape[1]), replace=False)
            perm = rng.permutation(pairs)
            tr_p, te_p = perm[:len(pairs) // 2], perm[len(pairs) // 2:]
            def build(pp):
                f = np.concatenate([2 * pp, 2 * pp + 1])
                Xa = np.hstack([V1[f][:, cols], V2[f][:, cols]]); Xb = np.hstack([V2[f][:, cols], V1[f][:, cols]])
                return np.vstack([Xa, Xb]), np.concatenate([np.ones(len(f)), np.zeros(len(f))])
            Xtr, ytr = build(tr_p); Xte, yte = build(te_p)
            clf = LinearSVC(C=1.0, dual="auto", max_iter=5000).fit(Xtr, ytr)
            a_list.append(float((clf.predict(Xte) == yte).mean()))
        accs["all" if n_sub is None else str(n_sub)] = float(np.mean(a_list))
    res["state_decoder_acc"] = accs
    return res


# ----------------------------------------------------------------------------------------
# modulation indices (TEN)
# ----------------------------------------------------------------------------------------
def modulation(ten_sessions, n_boot=2000, seed=0):
    rows = [r for s in ten_sessions for r in s["rows"]]
    cat = lambda k: np.concatenate([r[k] for r in rows])
    mu, mb, m1, m2 = cat("mi_unamb"), cat("mi_bavg"), cat("mi_s1"), cat("mi_s2")
    pp, pu, pos = cat("p_prime"), cat("p_unamb"), cat("pos")
    out = {}
    sel1h = (pu < 0.05) & np.isfinite(mu) & np.isfinite(mb)
    out["fig1h_n"] = int(sel1h.sum())
    out["fig1h_tls_slope_avg"] = float(C.tls_slope(mu[sel1h], mb[sel1h]))
    out["fig1h_r_avg"] = float(np.corrcoef(mu[sel1h], mb[sel1h])[0, 1])
    sel = (pp < 0.05) & pos & np.isfinite(m1) & np.isfinite(m2) & np.isfinite(mu)
    out["fig4g_n"] = int(sel.sum())
    for name, y in (("avg", mb), ("s1", m1), ("s2", m2)):
        out[f"fig4g_tls_slope_{name}"] = float(C.tls_slope(mu[sel], y[sel]))
        out[f"fig4g_r_{name}"] = float(np.corrcoef(mu[sel], y[sel])[0, 1])
    rng = np.random.default_rng(seed)
    x, y = mu[sel], m2[sel]
    n = len(x)
    bs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    out["reversal_p"] = float(np.mean(bs >= 0))
    out["reversal_passed"] = bool(out["fig4g_r_s2"] < 0 and out["reversal_p"] < 0.05)
    out["slope_ratio_abs_s1_over_s2"] = float(abs(out["fig4g_tls_slope_s1"] / out["fig4g_tls_slope_s2"]))
    return out


# ----------------------------------------------------------------------------------------
# part II helpers
# ----------------------------------------------------------------------------------------
def k4_pooled(k4_list):
    s = {k: sum(d[k] for d in k4_list) for k in k4_list[0]}
    n = s["n"]
    m4 = s["s1234"] / n
    den = np.mean([(s["s12"] / n) * (s["s34"] / n), (s["s13"] / n) * (s["s24"] / n), (s["s14"] / n) * (s["s23"] / n)])
    return float(m4 / den - 3.0)


def overdispersion(od_list):
    return float(sum(o[0] for o in od_list) / sum(o[1] for o in od_list))


def discreteness(disc_list):
    num = sum(d["ll_cont_minus_disc"] for d in disc_list)
    den = sum(d["n_test_bins"] for d in disc_list)
    return float(1000.0 * num / den)          # millinats per bin, > 0 favors continuous


def hysteresis_stat(hyst_by_session, seed=0):
    """Exploratory loop detector. hyst_by_session: per session, list over pairs of transition
    segments (direction, segment, C-S axis, mid template). For each pair with >= 20 transitions,
    the orthogonal residual at the transition (bins t-1, t) is averaged separately for S->C and
    C->S transitions in two random halves of the transitions; the statistic is the mean split-half
    correlation of the direction difference (about 0 without a loop)."""
    rng = np.random.default_rng(seed)
    vals = []
    for sess in hyst_by_session:
        for segs in sess:
            if len(segs) < 20:
                continue
            res = []
            for d, seg, axis, mid in segs:
                v = seg[2:4].mean(0) / C.DT - mid
                v = v - (v @ axis) * axis
                res.append((d, v))
            idx = rng.permutation(len(res))
            diffs = []
            for h in (idx[: len(idx) // 2], idx[len(idx) // 2:]):
                up = [res[i][1] for i in h if res[i][0] == 1]
                dn = [res[i][1] for i in h if res[i][0] == 0]
                if len(up) < 3 or len(dn) < 3:
                    break
                diffs.append(np.mean(up, 0) - np.mean(dn, 0))
            if len(diffs) == 2:
                vals.append(float(np.corrcoef(diffs[0], diffs[1])[0, 1]))
    return (float(np.mean(vals)) if vals else None), len(vals)
