"""Parse the VGG cells counting set exactly as the notebook's data cell does, and cache it.

The notebook reads `NNNcell.png` and `NNNdots.png` for NNN = 001..200, keeps the **blue** channel of
the image (`rgb[..., 2]`, the red and green channels of these files are empty) and defines a cell
centre as a pixel of the dots image that is red and not green (`dots[..., 0] > 0 & dots[..., 1] == 0`).
Both of those choices are reproduced here line for line — the count we are scored on is the sum of
that mask, so if the mask were defined differently the target would disagree with the paper's counts.

Reproduced, and also checked (see inspect_cells.py for the full report):
  * the red/green channels are not quite empty — they hold the RGB encoding of the dots and a little
    else, so the channel means are written to the CSV rather than assumed;
  * `cell_count` is recomputed as the sum of the parsed dot mask, never read from a file.

Usage:  py -I task4_cells/parse_cells.py <image-dir> <out.npz> [<records.csv>]

Writes <out.npz> with x (200, 256, 256) float32 in [0, 1], dots (200, 256, 256) uint8 0/1,
count (200,) int32, split_train / split_valid / split_test (paper's split: images 1-80 train,
81-100 validation, 101-200 test) and the per-image diagnostics of the CSV as arrays too.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CSV = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT.with_suffix(".csv")

pngs = sorted(SRC.glob("*cell.png"))
if len(pngs) != 200:
    sys.exit(f"expected 200 *cell.png files in {SRC}, found {len(pngs)}")

x, dots, rows = [], [], []
for p in pngs:
    i = int(p.stem[:3])
    dots_path = SRC / f"{i:03d}dots.png"
    rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255          # the notebook
    d = np.asarray(Image.open(dots_path).convert("RGB"))                             # 0..255, uint8
    centre = ((d[..., 0] > 0) & (d[..., 1] == 0)).astype(np.uint8)                   # ditto

    x.append(rgb[..., 2])
    dots.append(centre)
    rows.append({
        "image": i,
        "count": int(centre.sum()),
        "mean_blue": float(rgb[..., 2].mean()),
        "max_blue": float(rgb[..., 2].max()),
        "mean_red": float(rgb[..., 0].mean()),
        "mean_green": float(rgb[..., 1].mean()),
        # every red pixel of the dots file, and the ones the notebook's `green == 0` condition drops:
        "red_pixels": int((d[..., 0] > 0).sum()),
        "red_dropped_by_green": int(((d[..., 0] > 0) & (d[..., 1] > 0)).sum()),
        "green_pixels": int((d[..., 1] > 0).sum()),
        "blue_pixels_in_dots": int((d[..., 2] > 0).sum()),
        # are the dots the notebook finds ever touching the edge of the frame?
        "dots_on_border": int(centre[0, :].sum() + centre[-1, :].sum()
                              + centre[:, 0].sum() + centre[:, -1].sum()),
    })

x = np.stack(x).astype(np.float32)
dots = np.stack(dots).astype(np.uint8)
count = dots.reshape(len(dots), -1).sum(1).astype(np.int32)

train, valid, test = np.arange(0, 80), np.arange(80, 100), np.arange(100, 200)   # the notebook's
frame = pd.DataFrame(rows)
assert (frame["count"].to_numpy() == count).all(), "CSV and array counts disagree"
frame.to_csv(CSV, index=False)
np.savez_compressed(OUT, x=x, dots=dots, count=count, split_train=train, split_valid=valid,
                    split_test=test)
print(f"x {x.shape} {x.dtype} in [{x.min():.3f}, {x.max():.3f}] | dots {dots.shape} | "
      f"count {count.min()}-{count.max()}, mean {count.mean():.1f}")
print(f"splits {len(train)} / {len(valid)} / {len(test)} | "
      f"per-image stats -> {CSV} | cache -> {OUT}")
