"""MUX-S1 1.1, exploratory run E-repeat (post hoc: written after the primary result was seen).

    python explore_repeat.py --cache <cache dir> --out results/real/explore

Why. The primary run failed calibration (A1-A4) on the full set of FACE1500 pairs, while the
declared secondary run on the repeated stimuli only (pairs with both indices below 60, the authors'
Fig. 5 selection) passed A1, A3 and A4. Most of the 7000 pairs have one or two trials per cell, so
their templates are noisy and the state labels flicker. The statistics that read the labels (B1, B4,
B5, B11, B12) are the ones most exposed to this.

What. The frozen counts block of `mux_s1/realdata.py` (A1-A4 and B1-B5, B11, B12), unchanged, on the
'repeat' variant of the cache. The gain block is attempted as well; with at most about 60 faces per
session it is expected to be skipped by the frozen code itself (the coding model needs more faces
than its 120 features plus ten).

Status. Exploratory. It gives no verdict and changes none: the verdicts are those of the primary run.
It shows which of the primary refutations hold on the subset where our calibration matches the
authors' figures.

It also prints, for every session whose comments mention an injection, the comments and how many
trials the declared 'noinj' variant removed, to settle whether any analysed trial followed one.
"""
import argparse
import json
import os
import time

from mux_s1 import adapters as AD
from mux_s1 import faces as FC
from mux_s1 import hesse_reader as HR
from mux_s1 import realdata as RD
import verify_real as VR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results/real/explore")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    sessions = []
    for n in HR.DESIGNS["FACE1500"]:
        s = AD.load_session(os.path.join(a.cache, f"{n}__repeat.npz"))
        fp = os.path.join(a.cache, f"{n}__faces.npz")
        if os.path.exists(fp):
            try:
                FC.attach_faces(s, fp)
            except ValueError as e:
                s.meta = dict(s.meta, faces_error=str(e))
        sessions.append(s)
    print(f"E-repeat: {len(sessions)} sessions, {sum(len(s.pairs) for s in sessions)} pairs", flush=True)
    res = RD.run_counts_block(sessions, seed=0)
    print(f"counts block done, {time.time() - t0:.0f} s", flush=True)
    if sessions[0].faces is not None:
        g = RD.gain_block(sessions, seed=0)
        res["gain"] = g
        for k, v in g.items():
            if k.startswith("B"):
                res["B"][k] = v
        print(f"gain block: {'available' if g.get('available', True) else g.get('reason')}", flush=True)
    res["source"] = f"EXPLORATORY E-repeat (post hoc), cache variant 'repeat': {a.cache}"
    res["seconds"] = round(time.time() - t0, 1)
    res["thresholds"] = RD.THR
    res["published_targets"] = RD.PUB
    res["pairs_per_session"] = {s.name: len(s.pairs) for s in sessions}
    with open(os.path.join(a.out, "explore_repeat.json"), "w") as f:
        f.write(VR.to_json(res))
    md = VR.tables(res).replace("# MUX-S1 1.1: сверка с данными Hesse et al.",
                                "# MUX-S1 1.1, поисковый прогон E-repeat (после основного; вердиктов не выносит)")
    with open(os.path.join(a.out, "explore_repeat_tables.md"), "w") as f:
        f.write(md)
    print(md)

    inj = {}
    for n in HR.REC_IT:
        m = AD.load_session(os.path.join(a.cache, f"{n}__primary.npz")).meta
        if m.get("injection_comments"):
            mn = AD.load_session(os.path.join(a.cache, f"{n}__noinj.npz")).meta
            inj[n] = dict(comments=m.get("comments"), injection_comments=m["injection_comments"],
                          n_trials_file=m.get("n_trials_file"),
                          excluded_after_injection=mn.get("excluded_after_injection"),
                          comment_index_note=m.get("comment_index_note"),
                          in_FACE1500=n in HR.DESIGNS["FACE1500"])
    with open(os.path.join(a.out, "injections.json"), "w") as f:
        f.write(VR.to_json(inj))
    print("\nСессии с комментариями об инъекциях:")
    for n, d in inj.items():
        print(f"  {n} (FACE1500: {'да' if d['in_FACE1500'] else 'нет'}): проб в файле {d['n_trials_file']}, "
              f"исключено вариантом noinj {d['excluded_after_injection']}")
        for c in d["comments"] or []:
            print(f"     {c}")


if __name__ == "__main__":
    main()
