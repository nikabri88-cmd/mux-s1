"""Markdown tables of the exploratory analyses E3, E3b, E4, E5 for MUX-S1_report.md."""
import json
import os
import sys

import numpy as np

NAMES = {"M0_markov": "M0 Марков", "M1_rhythm": "M1 ритм k = 6", "M1b_rhythm_k3": "M1b ритм k = 3",
         "M2_cont_unimodal": "M2 плавное унимод.", "M3_cont_bimodal": "M3 плавное бимод.",
         "M4_adaptation": "M4 адаптация драйва", "M6_heterogeneous": "M6 неоднородность", "M7_ring": "M7 кольцо"}


def f(v, d=2):
    """Round half away from zero on the decimal representation (68.05 -> 68,1), decimal comma, true minus."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:f}".replace(".", ",")
    if s.startswith("-"):
        s = "−" + s[1:] if q != 0 else s[1:]
    return s


def e3():
    E = json.load(open("results/explore_snr_summary.json"))
    out = ["| Модель | κ / κ калибровки | Прайминг, % | ACF 10 мс | D, мс | Медиана HMM (состояние 1), мс | Тест правдоподобия, 10⁻⁴ | Прошли тест |",
           "|---|---|---|---|---|---|---|---|"]
    for m in ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal", "M4_adaptation"]:
        for k in sorted({float(key.split("|")[1]) for key in E}):
            x = E.get(f"{m}|{k}")
            if x is None:
                continue
            out.append(f"| {NAMES[m]} | {f(k, 1)} | {f(100 * np.mean(x['occ_prime']), 1)} | {f(np.mean(x['acf10']))} | "
                       f"{f(np.mean(x['D']), 1)} | {f(np.mean(x['dwell1']), 0)} | {f(1e4 * np.mean(x['lik_held']), 2)} | "
                       f"{sum(x['lik_held_pass'])}/{x['n']} |")
    return "\n".join(out)


def e3b():
    E = json.load(open("results/explore_accuracy.json"))
    out = ["| Модель | κ / κ калибровки | Знак LLR, все нейроны | HMM, все нейроны | HMM, половина нейронов |", "|---|---|---|---|---|"]
    for key, x in E.items():
        m, k = key.split("|")
        out.append(f"| {NAMES[m]} | {f(float(k), 1)} | {f(100 * x['llr_all'], 1)}% | {f(100 * x['hmm_all'], 1)}% | {f(100 * x['hmm_half'], 1)}% |")
    return "\n".join(out)


def e4():
    E = json.load(open("results/explore_k4.json"))
    out = ["| Модель | Д2 как в предрегистрации | Д2 с поправкой на масштаб | Истина (латентный вес) |", "|---|---|---|---|"]
    order = ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal", "M6_heterogeneous", "M7_ring"]
    for m in order:
        if m not in E:
            continue
        runs = list(E[m].values())
        cell = lambda key: (f"{f(np.mean([r[key] for r in runs]))} [{f(min(r[key] for r in runs))}; {f(max(r[key] for r in runs))}]"
                            if all(r[key] is not None for r in runs) else "–")
        out.append(f"| {NAMES[m]} | {cell('raw')} | {cell('scaled')} | {cell('truth')} |")
    return "\n".join(out)


def e5():
    E = json.load(open("results/explore_disc.json"))
    rows = {}
    for r in E:
        key = (r["model"], r["aligned"])
        rows.setdefault(key, []).append(r)
    out = ["| Данные | Д1 как в предрегистрации | Д1 с подобранными уровнями | Подобранный уровень m |", "|---|---|---|---|"]
    for (m, al), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lab = NAMES[m] + (", переключения по границам бинов" if al else "")
        x0 = np.median([r["x0_median"] for r in rs])
        out.append(f"| {lab} | {f(np.mean([r['D1'] for r in rs]), 1)} [{f(min(r['D1'] for r in rs), 1)}; {f(max(r['D1'] for r in rs), 1)}] | "
                   f"{f(np.mean([r['D1_fit'] for r in rs]), 1)} [{f(min(r['D1_fit'] for r in rs), 1)}; {f(max(r['D1_fit'] for r in rs), 1)}] | "
                   f"{f(1 / (1 + np.exp(-x0)), 2)} |")
    return "\n".join(out)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["e3", "e3b", "e4", "e5"]:
        print(f"--- {w}")
        try:
            print(globals()[w]())
        except FileNotFoundError as err:
            print("missing:", err)
