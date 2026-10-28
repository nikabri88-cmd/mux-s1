"""Generate the number-heavy tables of MUX-S1_report.md from results/summary.json (prints markdown)."""
import json
import sys

import numpy as np

S = json.load(open("results/summary.json"))
P1, P2, V = S["part1"], S["part2"], S["verdicts"]
NAMES = {"M0_markov": "M0 Марков", "M1_rhythm": "M1 ритм k = 6", "M1b_rhythm_k3": "M1b ритм k = 3",
         "M2_cont_unimodal": "M2 плавное унимод.", "M3_cont_bimodal": "M3 плавное бимод.",
         "M4_adaptation": "M4 адаптация драйва", "M4c_adaptation_tuning": "M4c адаптация настройки",
         "M5_conscious_gain": "M5 усиление сознат.", "M6_heterogeneous": "M6 неоднородность",
         "M7_ring": "M7 кольцо", "C1_averaging": "C1 усреднение", "C2_uncoordinated": "C2 несогласов."}
ORDER1 = ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal", "M4_adaptation",
          "M4c_adaptation_tuning", "M5_conscious_gain", "C1_averaging", "C2_uncoordinated"]
ORDER2 = ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal", "M6_heterogeneous", "M7_ring"]


def f(v, d=2):
    """Round half away from zero on the decimal representation (68.05 -> 68,1), decimal comma, true minus."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:f}".replace(".", ",")
    if s.startswith("-"):
        s = "−" + s[1:] if q != 0 else s[1:]
    return s


def mm(v, d=2, scale=1.0):
    v = np.array([x for x in v if x is not None], float) * scale
    return f"{f(v.mean(), d)} [{f(v.min(), d)}; {f(v.max(), d)}]"


def table3():
    out = ["| Модель | Реконструкция (круговая) | Согласованная реконструкция | Разворот | Правдоподобие | Все четыре |",
           "|---|---|---|---|---|---|"]
    for m in ORDER1:
        x = P1[m]; b = x["battery"]; n = x["n"]
        out.append(f"| {NAMES[m]} | {sum(b['reconstruction'])}/{n} | {sum(b['coordinated'])}/{n} | "
                   f"{sum(b['reversal'])}/{n} | {sum(b['likelihood'])}/{n} | {x['battery_all4']}/{n} |")
    return "\n".join(out)


def likheld():
    out = ["| Модель | Выигрыш, 10⁻⁴ нат/бин/нейрон | Среднее > 0 | p < 0,05 |", "|---|---|---|---|"]
    for m in sorted(ORDER1, key=lambda m: -np.mean(P1[m]["lik_held_mean"])):
        x = P1[m]; n = x["n"]
        out.append(f"| {NAMES[m]} | {mm(x['lik_held_mean'], 2, 1e4)} | {sum(1 for v in x['lik_held_mean'] if v > 0)}/{n} | "
                   f"{sum(x['battery']['likelihood'])}/{n} |")
    return "\n".join(out)


def likmix():
    ctrl = P1["C1_averaging"]["lik_mix_mean"] + P1["C2_uncoordinated"]["lik_mix_mean"]
    allm = [v for m in ORDER1 for v in P1[m]["lik_mix_mean"]]
    fr = [v for m in ORDER1 for v in P1[m]["lik_mix_fracpos"]]
    return (f"выигрыш у контролей C1 и C2 {f(min(ctrl), 3)}–{f(max(ctrl), 3)} нат на пробу на нейрон, у всех моделей "
            f"{f(min(allm), 3)}–{f(max(allm), 3)}; доля сессий с выигрышем {f(min(fr), 2)}–{f(max(fr), 2)}")


def table4():
    out = ["| Модель | S1C/S2S | Декодер, все нейроны | r усилений | S1C/прайминг | Рано − поздно | Доля клеток с частотой выше в состоянии 2 |",
           "|---|---|---|---|---|---|---|"]
    for m in ["M0_markov", "M4_adaptation", "M4c_adaptation_tuning", "M5_conscious_gain", "C1_averaging"]:
        x = P1[m]
        el = [e - l for e, l in zip(x["ratio_early"], x["ratio_late"])]
        out.append(f"| {NAMES[m]} | {mm(x['gain_ratio'])} | {mm(x['decoder_all'])} | {mm(x['r_gain'])} | "
                   f"{mm(x['s1c_over_prime'])} | {mm(el, 3)} | {mm(x['frac_s2_higher'])} |")
    out.append("| Опубликовано | 1,21 | 0,66 | 0,90 | – | – | большинство клеток |")
    return "\n".join(out)


def table5():
    out = ["| Модель | D, мс | ACF 10 мс | Хвост 110–200 мс | Отскок Д5 | Медиана HMM (состояние 1), мс | Д3 CV | Д1, миллинат/бин | Д2 κ̃4 | Д4 R |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER2:
        if m not in P2:
            continue
        x = P2[m]
        out.append(f"| {NAMES[m]} | {mm(x['D'], 1)} | {mm(x['acf10'])} | {mm(x['tail'], 3)} | {mm(x['rebound'], 3)} | "
                   f"{mm(x['dwell1'], 0)} | {mm(x['cv'])} | {mm(x['disc'], 2)} | {mm(x['k4'], 2)} | {mm(x['od'], 2)} |")
    out.append("| Опубликовано | 67 | 0,35–0,39 | −0,058 | −0,013 | ≈ 62 | – | – | – | – |")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["table3", "likheld", "likmix", "table4", "table5"]
    for w in which:
        print(f"--- {w}")
        print(globals()[w]())
