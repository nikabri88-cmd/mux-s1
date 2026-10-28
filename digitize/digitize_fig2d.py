"""Digitize Fig. 2d of Hesse et al. (2026): LLR autocorrelation (black = data, grey = shuffled, band = SD)."""
import json
import numpy as np
from PIL import Image

im = np.asarray(Image.open("digitize/fig2d_crop.png").convert("RGB")).astype(int)
H, W, _ = im.shape
gray = im.mean(axis=2)
dark = gray < 90

# axis lines: longest dark vertical line (y axis) and horizontal line (x axis)
col_counts = dark.sum(axis=0)
row_counts = dark.sum(axis=1)
x_axis_col = int(np.argmax(col_counts[:500]))          # y-axis line column
y_axis_row = int(np.argmax(row_counts[300:]) + 300)   # x-axis line row
# tick marks on the y axis: short dark horizontal runs just left of the axis line
ticks_y = []
for r in range(H):
    seg = dark[r, x_axis_col - 14:x_axis_col - 2]
    if seg.mean() > 0.8:
        ticks_y.append(r)
# group consecutive rows
def group(v):
    out, cur = [], [v[0]]
    for a in v[1:]:
        if a - cur[-1] <= 1:
            cur.append(a)
        else:
            out.append(np.mean(cur)); cur = [a]
    out.append(np.mean(cur)); return out
ty = group(ticks_y)
ticks_x = []
for c in range(W):
    seg = dark[y_axis_row + 2:y_axis_row + 14, c]
    if seg.mean() > 0.8:
        ticks_x.append(c)
tx = group(ticks_x)
print("axis col", x_axis_col, "axis row", y_axis_row, "y ticks", ty, "x ticks", tx)
# y ticks top->bottom: 1, 0.5, 0 ; x ticks: 0, 100, 200
ty = sorted(ty)[:3]
tx = sorted(tx)
y1, y05, y0 = ty
px_per_unit_y = (y0 - y1) / 1.0
x0, x200 = tx[0], tx[-1]
px_per_ms = (x200 - x0) / 200.0

def to_val(row):
    return (y0 - row) / px_per_unit_y

def col_of(lag):
    return int(round(x0 + lag * px_per_ms))

black = (im.max(axis=2) < 40)
band = (np.abs(im[:, :, 0] - 211) < 14) & (np.abs(im[:, :, 1] - 211) < 14) & (np.abs(im[:, :, 2] - 211) < 14)
greyc = (np.abs(gray - 125) < 22) & (np.ptp(im, axis=2) < 10)

rows = {}
for lag in list(range(10, 201, 5)):
    c = col_of(lag)
    if c >= W:
        continue
    cols = range(max(c - 1, 0), min(c + 2, W))
    rr = [r for cc in cols for r in range(int(y1) - 5, y_axis_row - 2) if black[r, cc]]
    gr = [r for cc in cols for r in range(int(y1) - 5, y_axis_row - 2) if greyc[r, cc]]
    br = [r for cc in cols for r in range(int(y1) - 5, y_axis_row - 2) if band[r, cc] or black[r, cc]]
    rows[lag] = {
        "acf": float(to_val(np.median(rr))) if rr else None,
        "shuffled": float(to_val(np.median(gr))) if gr else None,
        "band_hi": float(to_val(min(br))) if br else None,
        "band_lo": float(to_val(max(br))) if br else None,
    }
for k, v in rows.items():
    print(k, v)
res = {"source": "Hesse et al. 2026 bioRxiv 10.64898/2026.10.15.759299, Fig. 2d (page 36), rendered at 600 dpi",
       "calibration": {"x_axis_col": x_axis_col, "y_axis_row": y_axis_row, "y_ticks_px": ty, "x_ticks_px": tx,
                       "px_per_unit_y": px_per_unit_y, "px_per_ms": px_per_ms},
       "digitized": rows,
       "note": "black = recorded data mean across sessions; band = SD across sessions; shuffled = time-shuffled control"}
json.dump(res, open("digitize/fig2d_acf.json", "w"), indent=1)
