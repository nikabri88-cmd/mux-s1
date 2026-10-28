"""MUX-S1 1.1: the secondary analyses declared in MUX-S1_v11_deviations.md before the release was read.

    python verify_secondary.py --cache <cache dir> --out results/real/secondary

Verdicts on A and B come only from the primary run (`verify_real.py --design FACE1500`, frozen).
Everything here explains differences between our recount and the authors' published numbers, and
is reported descriptively. It uses the frozen functions of `mux_s1/realdata.py` unchanged; only the
inputs differ (variants of the cache written in steps 2 and 2b).

Order, cheapest and most useful first; each result is written to disk as soon as it is done:
  1. a4_by_state_FACE1500  A4 medians per state (state 1 conscious, state 2 suppressed) at 10/20/50 ms
  2. prime100_FACE1500     A1-A4 with priming templates on 100-1000 ms (the authors' LLR window)
  3. repeat_FACE1500       A1-A4 on pairs with both indices below 60 (the authors' Fig. 5 selection)
  4. all19                 A1-A4 on the 19 IT recordings (the population of the authors' Fig. 2)
  5. a4_by_state_all19     as 1, on the 19 recordings
  6. noinj_all19           A1-A4 on the 19 recordings without trials after an injection
                           (only when some session records an injection)
  7. gain_subset           the gain block on the FACE1500 sessions with complete face features,
                           only when the primary run could not compute it (pre-registered rule 2:
                           the primary reports B6-B10 as not computed, this is descriptive)
Declared and dropped before computing: A4 at 5 ms needs the raw files at a 5-ms base, which step 2
did not keep (logged in the deviation journal).
"""
import argparse
import json
import os
import time

import numpy as np

from mux_s1 import adapters as AD
from mux_s1 import faces as FC
from mux_s1 import hesse_reader as HR
from mux_s1 import realdata as RD

BIN_WIDTHS = (10.0, 20.0, 50.0)


def to_json(obj):
    def conv(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, indent=1, default=conv, ensure_ascii=False)


def load(cache, names, variant, with_faces=False):
    out = []
    for n in names:
        s = AD.load_session(os.path.join(cache, f"{n}__{variant}.npz"))
        fp = os.path.join(cache, f"{n}__faces.npz")
        if with_faces and os.path.exists(fp):
            try:
                FC.attach_faces(s, fp)
            except ValueError:
                pass
        out.append(s)
    return out


def a4_by_state(sessions):
    rows = {}
    for w in BIN_WIDTHS:
        st = [RD.session_states(s, bin_ms=w) for s in sessions]
        d = RD.dwell_stats(st)
        rows[int(w)] = dict(state1_ms=d["median_state1_ms"], state2_ms=d["median_state2_ms"],
                            pooled_ms=d["median_ms"])
    target = RD.PUB["dwell_by_binwidth"]
    for key in ("state1_ms", "state2_ms", "pooled_ms"):
        dev = [abs(rows[w][key] - target[w]) / target[w] for w in rows if w in target]
        rows[f"max_rel_dev_{key}"] = float(max(dev)) if dev else None
    return dict(curve=rows, target=target, threshold=RD.THR["A4_max_rel_dev"])


def a_only(sessions):
    A, acf, _ = RD.check_A(sessions)
    return dict(A=A, acf_first_lag=acf["acf_first_lag"], D_a_ms=acf["D_a_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results/real/secondary")
    ap.add_argument("--primary", default="results/real/primary/verify.json",
                    help="the primary result, read only to know whether its gain block ran")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F15, ALL = HR.DESIGNS["FACE1500"], HR.DESIGNS["ALL"]

    def save(name, res, t0):
        res["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            f.write(to_json(res))
        print(f"{name}: готово за {res['seconds']:.0f} с", flush=True)

    jobs = [("a4_by_state_FACE1500", lambda: a4_by_state(load(a.cache, F15, "primary"))),
            ("prime100_FACE1500", lambda: a_only(load(a.cache, F15, "prime100"))),
            ("repeat_FACE1500", lambda: a_only(load(a.cache, F15, "repeat"))),
            ("all19", lambda: a_only(load(a.cache, ALL, "primary"))),
            ("a4_by_state_all19", lambda: a4_by_state(load(a.cache, ALL, "primary")))]
    inj = [n for n in ALL if AD.load_session(os.path.join(a.cache, f"{n}__primary.npz")).meta.get("injection_comments")]
    if inj:
        jobs.append(("noinj_all19", lambda: dict(a_only(load(a.cache, ALL, "noinj")), sessions_with_injection=inj)))
    primary_gain = None
    if os.path.exists(a.primary):
        primary_gain = "B6_decoder_without_mean_rate" in json.load(open(a.primary)).get("B", {})
    if primary_gain is False:
        def gain_subset():
            ss = load(a.cache, F15, "primary", with_faces=True)
            ok = [s for s in ss if s.faces is not None]
            res = dict(sessions_used=[s.name for s in ok], sessions_without_faces=[s.name for s in ss if s.faces is None])
            if len(ok) >= 5:
                res["gain"] = RD.gain_block(ok, seed=0)
            else:
                res["gain"] = dict(available=False, reason="fewer than five sessions with complete features")
            return res
        jobs.append(("gain_subset", gain_subset))

    for name, fn in jobs:
        if os.path.exists(os.path.join(a.out, f"{name}.json")):
            print(f"{name}: уже посчитано", flush=True)
            continue
        t0 = time.time()
        save(name, fn(), t0)


if __name__ == "__main__":
    main()
