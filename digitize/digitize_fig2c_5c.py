"""Digitize Fig. 2c (state occupancy) and Fig. 5c (median dwell vs bin size) of Hesse et al. (2026)."""
import json
import numpy as np
from PIL import Image

def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(int)

def group(v):
    out, cur = [], [v[0]]
    for a in v[1:]:
        if a - cur[-1] <= 1:
            cur.append(a)
        else:
            out.append(float(np.mean(cur))); cur = [a]
    out.append(float(np.mean(cur))); return out

# ---------- Fig 2c ----------
im = load("digitize/fig2c_crop.png"); H, W, _ = im.shape
gray = im.mean(axis=2); dark = gray < 90
ax_col = int(np.argmax(dark[:, :600].sum(axis=0)))
ax_row = int(np.argmax(dark[300:, :].sum(axis=1)) + 300)
ty = group([r for r in range(H) if dark[r, ax_col - 14:ax_col - 2].mean() > 0.8])
tx = group([c for c in range(W) if dark[ax_row + 2:ax_row + 14, c].mean() > 0.8])
print("2c axes", ax_col, ax_row, ty, tx)
ty = sorted(ty)  # 100, 75, 50, 25, 0 top->bottom
y100, y0 = ty[0], ty[-1]
x_m1, x_p1 = tx[0], tx[-1]
def yval(r): return 100 * (y0 - r) / (y0 - y100)
def tval(c): return -1 + 2 * (c - x_m1) / (x_p1 - x_m1)
orange = (im[:, :, 0] > 220) & (im[:, :, 1] > 140) & (im[:, :, 1] < 185) & (im[:, :, 2] < 80)
purple = (im[:, :, 0] < 110) & (im[:, :, 1] < 70) & (im[:, :, 2] > 100) & (im[:, :, 2] < 160)
occ = []
for c in range(int(x_m1) + 5, int(x_p1) - 5, 3):
    ro = np.where(orange[:, c])[0]; rp = np.where(purple[:, c])[0]
    occ.append({"t": round(tval(c), 3),
                "state1_pct": float(yval(np.median(ro))) if len(ro) else None,
                "state2_pct": float(yval(np.median(rp))) if len(rp) else None})
def mean_in(key, a, b):
    v = [o[key] for o in occ if a <= o["t"] <= b and o[key] is not None]
    return float(np.mean(v)), len(v)
summ2c = {
    "priming_state1_pct_-0.9_-0.1": mean_in("state1_pct", -0.9, -0.1),
    "priming_state2_pct_-0.9_-0.1": mean_in("state2_pct", -0.9, -0.1),
    "binocular_state1_pct_0.3_0.9": mean_in("state1_pct", 0.3, 0.9),
    "binocular_state2_pct_0.3_0.9": mean_in("state2_pct", 0.3, 0.9),
    "binocular_state1_pct_0.3_0.5": mean_in("state1_pct", 0.3, 0.5),
    "binocular_state1_pct_0.7_0.9": mean_in("state1_pct", 0.7, 0.9),
}
print(json.dumps(summ2c, indent=1))

# ---------- Fig 5c (lower panel: median dwell vs bin size, log x axis) ----------
im = load("digitize/fig5c_crop.png"); H, W, _ = im.shape
gray = im.mean(axis=2); dark = gray < 90
ax_col = int(np.argmax(dark[:, :500].sum(axis=0)))
# lower panel x-axis row: dark row with many pixels below row 600
rows = dark[600:, :].sum(axis=1); ax_row = int(np.argmax(rows) + 600)
ty = group([r for r in range(400, H) if dark[r, ax_col - 14:ax_col - 2].mean() > 0.8])
print("5c axis", ax_col, ax_row, "yticks", ty)
ty = [t for t in ty if t > 420]
y200, y0 = min(ty), max(ty)  # ticks 200,100,0
def dval(r): return 200 * (y0 - r) / (y0 - y200)
# x positions of the bin-size ticks (log scale): label positions known 2,3,5,10,20,50,100
tx = group([c for c in range(W) if dark[ax_row + 2:ax_row + 14, c].mean() > 0.8])
print("x ticks (minor+major)", [round(t) for t in tx])
orange = (im[:, :, 0] > 220) & (im[:, :, 1] > 140) & (im[:, :, 1] < 190) & (im[:, :, 2] < 90)
purple = (im[:, :, 0] < 120) & (im[:, :, 1] < 80) & (im[:, :, 2] > 90) & (im[:, :, 2] < 170)
# map log10(bin) linearly between the 2 ms and 100 ms major ticks
x2, x100 = tx[0], tx[-1]
def col_of(b): return int(round(x2 + (np.log10(b) - np.log10(2)) / (np.log10(100) - np.log10(2)) * (x100 - x2)))
dw = {}
for b in [2, 3, 5, 10, 20, 50, 75, 100]:
    c = col_of(b)
    win = range(c - 4, c + 5)
    ro = [r for cc in win for r in range(int(y200) - 30, ax_row - 2) if orange[r, cc]]
    rp = [r for cc in win for r in range(int(y200) - 30, ax_row - 2) if purple[r, cc]]
    dw[b] = {"state1_ms": float(dval(np.median(ro))) if ro else None,
             "state2_ms": float(dval(np.median(rp))) if rp else None}
print(json.dumps(dw, indent=1))
json.dump({"fig2c_summary": summ2c, "fig2c_series": occ, "fig5c_dwell": dw,
           "source": "Hesse et al. 2026, Fig. 2c (p.36) and Fig. 5c lower panel (p.39), 600 dpi"},
          open("digitize/fig2c_5c.json", "w"), indent=1)
