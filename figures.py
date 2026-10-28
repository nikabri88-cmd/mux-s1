"""MUX-S1 figures (PNG, light surface). Palette: validated reference categorical slots 1-4."""
import json

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

S = json.load(open("results/summary.json"))
T = json.load(open("digitize/targets.json"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8})
class Comma(matplotlib.ticker.Formatter):
    """Decimal comma, true minus sign, one number of decimals per axis (from the tick step)."""

    def __call__(self, v, pos=None):
        locs = [x for x in self.locs if np.isfinite(x)]
        step = np.min(np.diff(sorted(locs))) if len(locs) > 1 else 1.0
        d = int(max(0, -np.floor(np.log10(step) + 1e-9))) if step > 0 else 0
        if step > 0 and abs(round(step * 10 ** d) - step * 10 ** d) > 1e-6:
            d += 1
        s = f"{v:.{d}f}".replace(".", ",")
        return s.replace("-", "−") if v < -10 ** (-d - 1) else s.lstrip("-")


comma = Comma()
NAMES = {"M0_markov": "M0 Марков", "M1_rhythm": "M1 ритм k = 6", "M1b_rhythm_k3": "M1b ритм k = 3",
         "M2_cont_unimodal": "M2 плавное унимод.", "M3_cont_bimodal": "M3 плавное бимод.",
         "M4_adaptation": "M4 адаптация драйва", "M4c_adaptation_tuning": "M4c адаптация настройки",
         "M5_conscious_gain": "M5 усиление сознат.", "M6_heterogeneous": "M6 неоднородность", "M7_ring": "M7 кольцо",
         "C1_averaging": "C1 усреднение", "C2_uncoordinated": "C2 несогласов."}


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(comma)


def title(ax, text):
    note = getattr(ax, "_ref_note", None)
    ax.set_title(text + (f"\nпунктир: {note}" if note else ""), loc="left", fontsize=9)


def strip(ax, models, key, part="part2", ref=None, ref_label=None, ylabel=""):
    d = S[part]
    xs = []
    for i, m in enumerate(models):
        if m not in d:
            continue
        v = np.array([x if x is not None else np.nan for x in d[m][key]], float)
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.05
        ax.scatter(i + jit, v, s=22, color=SLOT[0], edgecolor=SURF, linewidth=1.0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.nanmean(v)] * 2, color=INK, linewidth=2, zorder=4)
        xs.append(i)
    ax._ref_note = None
    if ref is not None:
        ax.axhline(ref, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax._ref_note = ref_label
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES[m] for m in models], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    style(ax)


# ---------------------------------------------------------------- figure 1: ACF, rebound and tail
fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.3), gridspec_kw={"width_ratios": [1.65, 1, 1]})
ax = axs[0]
lags = 10 * np.arange(21)
pub = T["digitized_fig2d_llr_acf"]
ax.plot(pub["lags_ms"], pub["acf"], color=INK, linewidth=2.6, zorder=5)
show = ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal"]
ends = [[pub["acf"][-1], pub["acf"][-1], "Hesse et al.", INK, "-"]]
for c, m in zip(SLOT, show):
    if m not in S["part2"]:
        continue
    a = np.array(S["part2"][m]["acf_mean"])
    ax.plot(lags[1:], a[1:], color=c, linewidth=2, zorder=4)
    ends.append([a[-1], a[-1], NAMES[m], c, "-"])
if "M0_markov" in S["part2"]:
    a0 = np.array(S["part2"]["M0_markov"]["acf_mean"])
    pd_ = dict(zip(pub["lags_ms"], pub["acf"]))
    L = np.arange(20, 160, 10)
    x = a0[L // 10]; y = np.array([pd_[l] for l in L])
    c0 = float(x @ y / (x @ x))
    ax.plot(lags[1:], c0 * a0[1:], color=SLOT[0], linewidth=1.6, linestyle=(0, (4, 2.5)), zorder=4)
    ends.append([c0 * a0[-1], c0 * a0[-1], f"M0 × {c0:.2f}".replace(".", ","), SLOT[0], (0, (2, 1.5))])
ends.sort(key=lambda e: e[1])
for k in range(1, len(ends)):
    if ends[k][1] - ends[k - 1][1] < 0.034:
        ends[k][1] = ends[k - 1][1] + 0.034
for y_end, y_lab, lab, c, ls in ends:
    ax.plot([202, 212], [y_end, y_lab], color=INK2, linewidth=0.6, zorder=5)
    ax.plot([213, 222], [y_lab, y_lab], color=c, linewidth=2, linestyle=ls, zorder=5, solid_capstyle="butt")
    ax.text(225, y_lab, lab, color=INK, fontsize=7.5, va="center", zorder=6,
            bbox=dict(boxstyle="square,pad=0.1", facecolor=SURF, edgecolor="none"))
ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
ax.set_xlim(0, 318)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlabel("Лаг, мс")
ax.set_ylabel("Автокорреляция LLR")
ax.set_title("а. Автокорреляция: опубликованная и модели (часть II)\nпунктир: M0 с множителем, подобранным на 20–150 мс",
             loc="left", fontsize=9)
style(ax)
strip(axs[1], ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal"], "rebound",
      ref=S["published_rebound"], ref_label="опубликованная кривая, −0,013", ylabel="Отскок Д5")
axs[1].axhline(0, color=INK2, linewidth=0.8)
title(axs[1], "б. Отскок: max(110–200 мс) − min(50–100 мс)")
axs[2].axhspan(-0.116, -0.029, color=GRID, alpha=0.6, zorder=0, linewidth=0)
strip(axs[2], ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal"], "tail",
      ref=-0.0579, ref_label="опубл. −0,058; полоса: допуск H5", ylabel="Средняя ACF на 110–200 мс")
title(axs[2], "в. Глубина хвоста")
fig.tight_layout()
fig.savefig("figures/fig1_acf.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: battery pass matrix
order = ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal", "M4_adaptation", "M4c_adaptation_tuning",
         "M5_conscious_gain", "C1_averaging", "C2_uncoordinated"]
tests = [("reconstruction", "Реконструкция\n(круговая)"), ("coordinated", "Согласованная\nреконструкция"),
         ("reversal", "Разворот\nмодуляции"), ("likelihood", "Правдоподобие\nдвух состояний")]
models = [m for m in order if m in S["part1"]]
fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(models) + 1.3))
for i, m in enumerate(models):
    for j, (k, _) in enumerate(tests):
        v = S["part1"][m]["battery"][k]
        frac = sum(v) / len(v)
        col = RAMP[min(int(round(frac * (len(RAMP) - 1))), len(RAMP) - 1)] if frac > 0 else "#f0efec"
        ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=col, linewidth=0))
        ax.text(j + 0.5, i + 0.5, f"{sum(v)}/{len(v)}", ha="center", va="center", fontsize=9,
                color="#ffffff" if frac >= 0.6 else INK)
ax.set_xlim(0, len(tests)); ax.set_ylim(len(models), 0)
ax.set_xticks(np.arange(len(tests)) + 0.5); ax.set_xticklabels([t for _, t in tests], fontsize=8)
ax.set_yticks(np.arange(len(models)) + 0.5); ax.set_yticklabels([NAMES[m] for m in models], fontsize=8)
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.xaxis.tick_top()
ax.set_title("Батарея тестов авторов: сколько наборов прошли тест", loc="left", fontsize=9, pad=34)
fig.tight_layout()
fig.savefig("figures/fig2_battery.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 3: gain asymmetry
gm = [m for m in ["M0_markov", "M4_adaptation", "M4c_adaptation_tuning", "M5_conscious_gain"] if m in S["part1"]]
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6))
axs[0].axhspan(1.10, 1.40, color=GRID, alpha=0.6, zorder=0, linewidth=0)
strip(axs[0], gm, "gain_ratio", part="part1", ref=1.21, ref_label="опубликовано, 1,21\nполоса: допуск H6 для M4", ylabel="Усиление S1C / S2S")
axs[0].set_ylim(0.95, 1.43)
title(axs[0], "а. Асимметрия усиления")
strip(axs[1], gm, "s1c_over_prime", part="part1", ylabel="Усиление S1C / прайминг")
title(axs[1], "б. Сознательное лицо против прайминга")
for m in gm:
    S["part1"][m]["early_minus_late"] = [e - l for e, l in zip(S["part1"][m]["ratio_early"], S["part1"][m]["ratio_late"])]
strip(axs[2], gm, "early_minus_late", part="part1", ref=0.0, ylabel="Отношение: рано − поздно")
title(axs[2], "в. Динамика внутри окна")
fig.tight_layout()
fig.savefig("figures/fig3_gain.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 4: discriminators
dm = [m for m in ["M0_markov", "M1_rhythm", "M1b_rhythm_k3", "M2_cont_unimodal", "M3_cont_bimodal",
                  "M6_heterogeneous", "M7_ring"] if m in S["part2"]]
fig, axs2 = plt.subplots(2, 2, figsize=(9.6, 7.6))
axs = axs2.ravel()
strip(axs[0], dm, "disc", ref=0.0, ref_label="предрегистрированный ноль; скачки дают +7…+14", ylabel="Д1, миллинат/бин")
title(axs[0], "а. Дискретность (Д1)")
strip(axs[1], dm, "k4", ref=-2.0, ref_label="−2 – ожидание для скачков (см. E4)", ylabel="Д2, стандартизованный κ4")
axs[1].axhline(0, color=INK2, linewidth=0.8)
title(axs[1], "б. Четвёртый кумулянт (Д2)")
strip(axs[2], dm, "cv", ref=1.0, ref_label="1 – экспонента; у M0 после HMM около 0,81", ylabel="Д3, CV длительностей")
title(axs[2], "в. Регулярность (Д3)")
strip(axs[3], dm, "od", ref=1.0, ref_label="1 – однородные пробы", ylabel="Д4, сверхдисперсия R")
title(axs[3], "г. Неоднородность проб (Д4)")
fig.tight_layout()
fig.savefig("figures/fig4_discriminators.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 5: exploratory E3 (kappa sweep)
try:
    E = json.load(open("results/explore_snr_summary.json"))
except FileNotFoundError:
    E = None
if E:
    em = ["M0_markov", "M1_rhythm", "M2_cont_unimodal", "M3_cont_bimodal"]
    mults = sorted({float(k.split("|")[1]) for k in E})
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.7))
    ax = axs[0]
    styles = [(m, c, "-") for m, c in zip(em, SLOT)] + [("M4_adaptation", INK2, (0, (4, 2.5)))]
    for m, c, ls in styles:
        xs = [k for k in mults if f"{m}|{k}" in E]
        if not xs:
            continue
        ys = [1e4 * np.mean(E[f"{m}|{k}"]["lik_held"]) for k in xs]
        ax.plot(xs, ys, color=c, linewidth=2, linestyle=ls, marker="o", markersize=5, markeredgecolor=SURF,
                zorder=4, label=NAMES[m])
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("κ / κ калибровки")
    ax.set_ylabel("Тест правдоподобия, 10⁻⁴ нат/бин/нейрон")
    ax.set_title("а. Тест правдоподобия батареи\nвыше нуля: две частоты лучше статичной смеси", loc="left", fontsize=9)
    ax.set_ylim(-26, 34)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.2)
    style(ax); ax.xaxis.set_major_formatter(comma)
    panels = (("occ_prime", 100, (22.9, 22.9), "Ложное состояние в прайминге, %",
               "б. Прайминг, M0 (одинаково у всех моделей без адаптации)\nпунктир: оцифровано у авторов, 22,9%"),
              ("acf10", 1, (0.35, 0.39), "ACF LLR на лаге 10 мс",
               "в. Автокорреляция на 10 мс\nполоса: оцифровано у авторов, 0,35–0,39"))
    for j, (key, scale, ref, yl, tt) in enumerate(panels):
        ax = axs[j + 1]
        for c, m in zip(SLOT, em[:1] if key == "occ_prime" else em):
            xs = [k for k in mults if f"{m}|{k}" in E]
            ys = [scale * np.mean(E[f"{m}|{k}"][key]) for k in xs]
            ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=5, markeredgecolor=SURF, zorder=4)
        if ref[0] == ref[1]:
            ax.axhline(ref[0], color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        else:
            ax.axhspan(ref[0], ref[1], color=GRID, alpha=0.9, zorder=0, linewidth=0)
        ax.set_xlabel("κ / κ калибровки")
        ax.set_ylabel(yl)
        ax.set_title(tt, loc="left", fontsize=9)
        style(ax); ax.xaxis.set_major_formatter(comma)
    fig.tight_layout()
    fig.savefig("figures/fig5_explore_kappa.png", dpi=200)
    plt.close(fig)
print("figures written")
