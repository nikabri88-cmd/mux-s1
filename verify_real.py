"""MUX-S1 version 1.1: run the pre-registered checks on the released data of Hesse et al. (2026).

    python verify_real.py --source dandi --path <dir> --design FACE1500 --out results/real
    python verify_real.py --source synthetic --model M0_markov          # the pre-release test

The source `synthetic` builds the canonical form from the simulator of version 1.0 and is how the
code was tested before the release; `dandi` uses the reader written on release day. Everything the
checks do is in `mux_s1/realdata.py`, which is hashed before the release; the reader is not.
"""
import argparse
import json
import os
import time

import numpy as np

from mux_s1 import adapters as AD
from mux_s1 import realdata as RD


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


def fmt(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x, n)}" for k, x in v.items())
    return str(v)


ROWS_A = [("A1_priming_state1", "A1 доля ложного состояния в прайминге", "0.229 (рис. 2c)"),
          ("A2_acf_first_lag", "A2 автокорреляция на первом лаге", "0.35-0.39 (рис. 2d)"),
          ("A3_bino_state1", "A3 доля состояния 1 в бинокулярном режиме", "0.47 (рис. 2c)"),
          ("A4_dwell_by_binwidth", "A4 длительности по ширине бина", "рис. 5c")]
ROWS_B = [("B1_dwell_over_D", "B1 медиана HMM / D автокорреляции", "<= 1.15; >= 1.35 опровергает"),
          ("B2_D_global_over_D_trial", "B2 D(общее среднее) / D(среднее в пробе)", ">= 1.40"),
          ("B3_acf_over_calibration", "B3 амплитуда ACF к калибровке", ">= 1.25"),
          ("B4_label_threshold", "B4 выигрыш двух частот по числу нейронов меток", "рост; <= 0 при 10"),
          ("B5_fig3a_equal_noise", "B5 рис. 3a при равном шуме оценки", "> 0 и <= 0.50 от неравного"),
          ("B6_decoder_without_mean_rate", "B6 декодер без средней частоты", "<= 0.56"),
          ("B7_gain_early_minus_late", "B7 отношение усилений, рано минус поздно", ">= +0.05"),
          ("B9_mean_rate_sign", "B9 знак средней частоты и отношение", "совместно"),
          ("B10_gain_vs_r2", "B10 усиление против R2 модели кодирования", "rho >= 0.30, спад >= 0.15"),
          ("B11_dwell_cv", "B11 CV длительностей", "марков 0.75-0.90"),
          ("B12_overdispersion", "B12 сверхдисперсия по пробам", "<= 1.15")]


def tables(res):
    out = ["# MUX-S1 1.1: сверка с данными Hesse et al.", "",
           f"Сессий {res['n_sessions']}, нейронов {res['n_neurons_total']}, пар {res['n_pairs_total']}.",
           f"Источник: {res['source']}. Время {res['seconds']} с.", "",
           "## A. Калибровочные сверки", "",
           "| Проверка | Значение | Ожидалось | Совпало |", "| --- | --- | --- | --- |"]
    for key, name, target in ROWS_A:
        d = res["A"][key]
        val = d.get("max_rel_dev") if key == "A4_dwell_by_binwidth" else d.get("value")
        extra = f" (кривая: {fmt(d['value'], 0)})" if key == "A4_dwell_by_binwidth" else ""
        out.append(f"| {name} | {fmt(val)}{extra} | {target} | {fmt(d.get('passed'))} |")
    out += ["", "## B. Предсказания версии 1.0", "",
            "| Предсказание | Значение | Порог | Подтвердилось |", "| --- | --- | --- | --- |"]
    for key, name, target in ROWS_B:
        d = res["B"].get(key)
        if d is None:
            out.append(f"| {name} | не считалось | {target} | - |")
            continue
        if key == "B4_label_threshold":
            val = f"при 10: {fmt(d['small_n'], 4)}, при половине: {fmt(d['full'], 4)}, монотонно: {fmt(d['monotone'])}"
        elif key == "B5_fig3a_equal_noise":
            val = (f"равный {fmt(d['equal_noise'], 4)}, неравный {fmt(d['unequal_noise'], 4)}, "
                   f"доля {fmt(d['ratio'])}")
        elif key == "B9_mean_rate_sign":
            val = f"доля {fmt(d['frac_state2_higher'])}, отношение {fmt(d['gain_ratio'])}"
        elif key == "B10_gain_vs_r2":
            val = f"rho {fmt(d['spearman_rho'])}, спад {fmt(d['drop'])}"
        elif key == "B11_dwell_cv":
            val = f"{fmt(d['value'])} ({d['call']})"
        elif key == "B6_decoder_without_mean_rate":
            val = f"{fmt(d['value'])} (исходный {fmt(d['raw'])}, только узор {fmt(d['pattern_only'])})"
        else:
            val = fmt(d.get("value"))
        passed = d.get("passed", d.get("unexplained_by_tested_models"))
        out.append(f"| {name} | {val} | {target} | {fmt(passed)} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dandi", "synthetic"], default="synthetic")
    ap.add_argument("--path", default="")
    ap.add_argument("--design", default="FACE1500")
    ap.add_argument("--model", default="M0_markov")
    ap.add_argument("--n-sess", type=int, default=19)
    ap.add_argument("--n-pairs", type=int, default=25)
    ap.add_argument("--n-tr", type=int, default=15)
    ap.add_argument("--neurons", type=int, default=300)
    ap.add_argument("--faces", action="store_true", help="synthetic: attach a face feature space")
    ap.add_argument("--no-gain", action="store_true", help="skip the gain block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.source == "dandi":
        sessions = AD.dandi_sessions(a.path, design=a.design)
        src = f"dandi:{a.design}:{a.path}"
    else:
        sessions = AD.synthetic_sessions(a.model, seed=a.seed, n_sess=a.n_sess, n_pairs=a.n_pairs,
                                         n_tr=a.n_tr, N=a.neurons, with_faces=a.faces)
        src = f"synthetic:{a.model}:seed{a.seed}"
    print(f"{len(sessions)} sessions, {sessions[0].n_neurons} neurons, "
          f"{sum(len(s.pairs) for s in sessions)} pairs, bin {sessions[0].bin_ms} ms", flush=True)
    res = RD.run_counts_block(sessions, seed=a.seed)
    print(f"counts block done, {time.time() - t0:.0f} s", flush=True)
    if not a.no_gain and sessions[0].faces is not None:
        g = RD.gain_block(sessions, seed=a.seed)
        res["gain"] = g
        for k, v in g.items():
            if k.startswith("B"):
                res["B"][k] = v
        print(f"gain block done, {time.time() - t0:.0f} s", flush=True)
    res["source"] = src
    res["seconds"] = round(time.time() - t0, 1)
    res["thresholds"] = RD.THR
    res["published_targets"] = RD.PUB
    with open(os.path.join(a.out, "verify.json"), "w") as f:
        f.write(to_json(res))
    md = tables(res)
    with open(os.path.join(a.out, "verify_tables.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
