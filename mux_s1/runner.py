"""Dataset runners for MUX-S1 parts I and II."""
import json
import time

import numpy as np

from . import analysis as A
from . import core as C
from . import experiments as X

MODEL_INDEX = {name: i for i, name in enumerate(sorted(C.MODELS))}


def _rng(seed, model, part):
    return np.random.default_rng([int(seed), MODEL_INDEX[model], {"I": 1, "II": 2, "CAL": 3}[part]])


def part1_dataset(model, seed, timescale, kappa, n_face_sess=9, n_ten_sess=10):
    t0 = time.time()
    spec = C.MODELS[model]
    rng = _rng(seed, model, "I")
    faces = C.make_faces(rng, 1500)
    fs = [X.face1500_session(rng, spec, timescale, faces, kappa) for _ in range(n_face_sess)]
    ts = [X.ten_session(rng, spec, timescale, kappa) for _ in range(n_ten_sess)]
    n_an = fs[0]["n_an"]
    prim = np.hstack([s["prim"] for s in fs])
    R_all = np.concatenate([s["R_all"] for s in fs], axis=-1)
    R_xv = np.concatenate([s["R_xv"] for s in fs], axis=-1)
    off = np.cumsum([0] + [s["prim"].shape[1] for s in fs])[:-1]
    hB = np.concatenate([o + s["hB"] for o, s in zip(off, fs)])
    out = dict(model=model, seed=seed, timescale=timescale, kappa=kappa)
    out["recon_circular"] = A.reconstruct(faces, prim, R_all, seed=seed)
    out["recon_coordinated"] = A.reconstruct(faces, prim[:, hB], R_xv[:, :, 0][..., hB], seed=seed + 1)
    out["recon_coordinated_twofold"] = A.reconstruct(faces, prim, R_xv[:, :, 0], seed=seed + 2)
    out["gain"] = A.gains_and_decoder(faces, prim, R_xv, n_an, seed=seed)
    out["modulation"] = A.modulation(ts, seed=seed)
    llr = [s["llr_bino"] for s in fs] + [s["llr_bino"] for s in ts]
    out["acf"] = A.acf_summary(llr)
    out["acf_face1500"] = A.acf_summary([s["llr_bino"] for s in fs])
    out["acf_ten"] = A.acf_summary([s["llr_bino"] for s in ts])
    out["dwell"] = A.dwell_summary([s["runs"] for s in fs + ts])
    out["occ"] = dict(prime_face=float(np.mean([s["occ_prime"] for s in fs])),
                      prime_ten=float(np.mean([s["occ_prime"] for s in ts])),
                      bino_face=float(np.mean([s["occ_bino"] for s in fs])),
                      bino_ten=float(np.mean([s["occ_bino"] for s in ts])))
    mix = [float(np.mean(s["lik_mix"])) for s in fs + ts]
    held = [float(np.mean(s["lik_held"])) for s in fs + ts]
    out["lik_mix"] = dict(zip(("mean", "t", "p_one"), A.session_t(mix)), frac_pos=float(np.mean(np.array(mix) > 0)))
    out["lik_held"] = dict(zip(("mean", "t", "p_one"), A.session_t(held)), frac_pos=float(np.mean(np.array(held) > 0)))
    out["battery"] = dict(reconstruction=out["recon_circular"]["passed"],
                          coordinated=out["recon_coordinated"]["passed"],
                          reversal=out["modulation"]["reversal_passed"],
                          likelihood=bool(out["lik_held"]["mean"] > 0 and out["lik_held"]["p_one"] < 0.05))
    out["battery"]["n_passed"] = int(sum(out["battery"].values()))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def part2_dataset(model, seed, timescale, kappa, n_sess=19, part="II"):
    t0 = time.time()
    spec = C.MODELS[model]
    rng = _rng(seed, model, part)
    ss = [X.dyn_session(rng, spec, timescale, kappa, want_hyst=(part == "II")) for _ in range(n_sess)]
    out = dict(model=model, seed=seed, timescale=timescale, kappa=kappa)
    out["acf"] = A.acf_summary([s["llr_bino"] for s in ss])
    out["dwell"] = A.dwell_summary([s["runs"] for s in ss])
    out["occ_prime"] = float(np.mean([s["occ_prime"] for s in ss]))
    out["occ_bino"] = float(np.mean([s["occ_bino"] for s in ss]))
    if part == "II":
        out["k4"] = A.k4_pooled([s["k4"] for s in ss])
        out["overdispersion"] = A.overdispersion([s["od"] for s in ss])
        out["discreteness_mnats_per_bin"] = A.discreteness([s["disc"] for s in ss])
        out["disc_best"] = [s["disc"]["best_ou"] for s in ss]
        h, nh = A.hysteresis_stat([s["hyst"] for s in ss], seed=seed)
        out["hysteresis_r"] = h
        out["hysteresis_n_pairs"] = nh
    out["seconds"] = round(time.time() - t0, 1)
    return out


def to_json(obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    return json.dumps(obj, default=conv)
