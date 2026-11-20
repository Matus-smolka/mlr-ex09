"""Is this dataset what the notebook says it is, and how bad is the overlap it warns about?

The task's whole premise is that cells touch and merge, so a threshold followed by counting the blobs
misses most of them. That premise is measurable before any network is trained, and the answer decides
the modelling: how big a cell is, how close cells sit, how much of the image is inside a merged blob,
whether a merged blob is *bigger* in proportion to how many cells it holds (it should be, minus the
shared boundary — which is what makes a hand-crafted counter possible at all), and how wide a Gaussian
the density target needs.

Sections: 1 structure, 2 what the channel and dot-file choices of the notebook really keep, 3 the dots,
4 the cells (size and shape from the blobs that contain exactly one dot), 5 the overlap, 6 area and
brightness vs the number of cells in a blob, 7 the count distribution, 8 the baselines measured before
anything is trained.

Usage:  py -I task4_cells/inspect_cells.py <cache.npz> <figs-dir> [<records.csv>]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label as sk_label

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
d = np.load(CACHE, allow_pickle=True)
x, dots, count = d["x"], d["dots"].astype(bool), d["count"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
n, H, W = x.shape
PIX = H * W


def otsu(img, bins=256):
    """Otsu's threshold, from the histogram — the textbook rule the task's baseline is built on."""
    hist, edges = np.histogram(img, bins=bins, range=(0.0, 1.0))
    p = hist / hist.sum()
    omega = np.cumsum(p)                                    # weight of the lower class per threshold
    centres = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * centres)
    var = np.where(omega * (1 - omega) > 0, omega * (1 - omega) * (mu[-1] - mu) ** 2, 0)
    return float(centres[int(np.argmax(var))])


print("=" * 78)
print("1. STRUCTURE")
print("=" * 78)
print(f"images {x.shape} {x.dtype} in [{x.min():.4f}, {x.max():.4f}] | non-finite {int((~np.isfinite(x)).sum())}")
print(f"dot masks {dots.shape} | total dots {int(count.sum())} | per image {count.min()}-{count.max()}, "
      f"mean {count.mean():.1f}, median {np.median(count):.0f}, sd {count.std(ddof=1):.1f}")
print(f"splits {({s: len(v) for s, v in splits.items()})} | disjoint "
      f"{len(set(np.concatenate(list(splits.values())))) == n} | covers all {n}: "
      f"{sorted(np.concatenate(list(splits.values()))) == list(range(n))}")
print("count per split: " + " | ".join(
    f"{s} mean {count[v].mean():.0f} sd {count[v].std(ddof=1):.0f} range {count[v].min()}-{count[v].max()}"
    for s, v in splits.items()))
fp = np.round(x.reshape(n, -1)[:, ::37].mean(1) * 1e6).astype(np.int64)
dup = sum(1 for i in range(n) for j in range(i + 1, n) if (x[i] == x[j]).all())
print(f"images sharing a coarse fingerprint: {n - len(np.unique(fp))} | byte-identical pairs: {dup}")

print()
print("=" * 78)
print("2. WHAT THE NOTEBOOK'S TWO CHOICES KEEP")
print("=" * 78)
if len(sys.argv) > 3:
    import pandas as pd
    rec = pd.read_csv(sys.argv[3])
    print(f"image files, channel means: red {rec.mean_red.mean():.5f}, green {rec.mean_green.mean():.5f}, "
          f"blue {rec.mean_blue.mean():.5f} -> the cells are in the blue channel, the other two are dark "
          f"but not empty")
    print(f"dot files: red pixels {int(rec.red_pixels.sum())}, of which the 'green == 0' condition drops "
          f"{int(rec.red_dropped_by_green.sum())}; green pixels {int(rec.green_pixels.sum())}, blue "
          f"{int(rec.blue_pixels_in_dots.sum())}")
    print(f"cells: notebook's rule finds {int(count.sum())}, raw red would find {int(rec.red_pixels.sum())} "
          f"(difference {int(rec.red_pixels.sum()) - int(count.sum())})")

print()
print("=" * 78)
print("3. THE DOTS")
print("=" * 78)
nn, dot_lab_sizes = [], []
for i in range(n):
    lab, nb = sk_label(dots[i], connectivity=2, return_num=True)
    dot_lab_sizes.append(np.bincount(lab[dots[i]], minlength=nb + 1)[1:])
    tree = cKDTree(np.argwhere(dots[i]))
    nn.append(tree.query(tree.data, k=2)[0][:, 1])
nn, dot_lab_sizes = np.concatenate(nn), np.concatenate(dot_lab_sizes)
print(f"dot mask: {int(count.sum())} marked pixels in {dot_lab_sizes.size} connected components "
      f"(sizes {dot_lab_sizes.min()}-{dot_lab_sizes.max()}) -> {int(count.sum()) - dot_lab_sizes.size} "
      f"places where two annotated centres touch and their 1-pixel dots merge into one component")
print(f"   the notebook's count is the PIXEL sum, so it is kept here; but those "
      f"{int(count.sum()) - dot_lab_sizes.size} pairs are genuinely ambiguous at 1 px, and no density "
      f"map can resolve two centres 1 px apart - a ceiling on any localization score, not on the count")
print(f"nearest-neighbour distance between centres (px): p1 {np.percentile(nn, 1):.1f}, "
      f"p5 {np.percentile(nn, 5):.1f}, median {np.median(nn):.1f}, p95 {np.percentile(nn, 95):.1f}, "
      f"max {nn.max():.1f}")
print(f"pairs closer than 2 px (one Gaussian smear cannot separate them): {int((nn < 2).sum())} of {nn.size}")
print(f"dots on the outermost row/column: "
      f"{sum(int(dots[i][0].sum() + dots[i][-1].sum() + dots[i][:, 0].sum() + dots[i][:, -1].sum()) for i in range(n))}")

print()
print("=" * 78)
print("4. THE CELLS, FROM THE BLOBS THAT HOLD EXACTLY ONE DOT")
print("=" * 78)
thr = np.array([otsu(x[i]) for i in range(n)])
fg = np.array([x[i] > thr[i] for i in range(n)])
blob_lab, inside_lab, dots_outside = [], [], 0
for i in range(n):
    lab, nb = sk_label(fg[i], connectivity=2, return_num=True)
    inside = np.bincount(lab[dots[i]], minlength=nb + 1)            # dots per blob label
    dots_outside += int(inside[0])                                  # centre below the threshold
    blob_lab.append(lab)
    inside_lab.append(inside)
single = np.concatenate([[r.area for r in regionprops(lab)
                          if inside[r.label] == 1 and r.area >= 4] for lab, inside in zip(blob_lab, inside_lab)])
single_ecc = np.concatenate([[r.eccentricity for r in regionprops(lab)
                              if inside[r.label] == 1 and r.area >= 4] for lab, inside in zip(blob_lab, inside_lab)])
A1 = float(np.median(single))
r1 = (A1 / np.pi) ** 0.5
nblobs = int(sum(l.max() for l in blob_lab))
print(f"Otsu threshold: median {np.median(thr):.3f} (range {thr.min():.3f}-{thr.max():.3f}) | foreground "
      f"{fg.reshape(n, -1).mean():.1%} of all pixels | blobs {nblobs}")
print(f"annotated centres that fall on no foreground pixel at all: {dots_outside} of {int(count.sum())} "
      f"({dots_outside / count.sum():.2%}) - the threshold does not see every labelled cell")
print(f"blobs holding exactly one dot: {single.size} of {nblobs}")
print(f"their area (px): p5 {np.percentile(single, 5):.0f}, median {A1:.0f}, p95 "
      f"{np.percentile(single, 95):.0f} -> equivalent radius {(np.percentile(single, 5) / np.pi) ** 0.5:.1f} "
      f"/ {r1:.1f} / {np.sqrt(np.percentile(single, 95) / np.pi):.1f} px (p5 / median / p95)")
print(f"their eccentricity: median {np.median(single_ecc):.2f}, p95 {np.percentile(single_ecc, 95):.2f} "
      f"(0 = circle, 1 = slit) -> 'cells are round' is "
      f"{'supported' if np.median(single_ecc) < 0.8 else 'weaker than the task implies'}")
print(f"{count.mean():.0f} cells of {A1:.0f} px need {count.mean() * A1 / PIX:.0%} of the frame if none "
      f"overlapped; the foreground is {fg.reshape(n, -1).mean():.0%} -> overlap is not an edge case, "
      f"it is most of the image")

print()
print("=" * 78)
print("5. THE OVERLAP")
print("=" * 78)
per_dot = np.concatenate([ins[1:][ins[1:] > 0] for ins in inside_lab])       # dots per non-empty blob
merged_px = sum(int(sum(a for a, k in zip(np.bincount(l.ravel())[1:], ins[1:]) if k > 1))
                for l, ins in zip(blob_lab, inside_lab))
print(f"cells per blob: 1 -> {int((per_dot == 1).sum())}, 2 -> {int((per_dot == 2).sum())}, "
      f"3-4 -> {int(((per_dot >= 3) & (per_dot <= 4)).sum())}, 5-9 -> {int(((per_dot >= 5) & (per_dot <= 9)).sum())}, "
      f"10+ -> {int((per_dot >= 10).sum())}")
print(f"a non-empty blob holds {per_dot.mean():.2f} cells (max {per_dot.max()}); "
      f"{np.mean(np.repeat(per_dot > 1, per_dot)):.1%} of all cells sit in a blob with at least one other")
print(f"{merged_px / fg.reshape(n, -1).sum():.1%} of the foreground area is inside a multi-cell blob")
print(f"Otsu finds {nblobs} blobs for {int(count.sum())} cells -> a blob counter sees "
      f"{nblobs / count.sum():.1%} of the cells; {(nn < 2 * r1).mean():.1%} of cells have their nearest "
      f"neighbour within one cell diameter ({2 * r1:.0f} px)")
frac = fg.reshape(n, -1).mean(1)
print(f"foreground FRACTION per image: {frac.mean():.3f} +- {frac.std():.3f} "
      f"(min {frac.min():.3f}, max {frac.max():.3f}) | correlation with the true count "
      f"{np.corrcoef(frac, count)[0, 1]:+.3f}")
print("   the field is saturated: the frame is ~68 % covered whether the image holds 74 cells or 317, "
      "so")
print("   the area of the bright region cannot be a count. What varies is the area each cell gets:")
for lo, hi in ((70, 120), (120, 170), (170, 220), (220, 270), (270, 320)):
    m = (count >= lo) & (count < hi)
    if m.sum():
        print(f"      count {lo:>3d}-{hi:<3d}  n={int(m.sum()):>3d}  foreground fraction {frac[m].mean():.3f}"
              f"   foreground px per cell {np.mean(frac[m] * PIX / count[m]):5.0f} "
              f"({np.mean(frac[m] * PIX / count[m]) / A1:.1f} lone cells)")

haze = np.mean([(fg[i] & (distance_transform_edt(~dots[i]) > 12)).sum() / max(int(fg[i].sum()), 1)
                for i in range(n)])
inside_core = np.mean([fg[i][distance_transform_edt(~dots[i]) <= 3].mean() for i in range(n)])
print(f"where the threshold actually sits: Otsu {np.median(thr):.3f} is BELOW the median pixel value "
      f"({np.median(x):.3f}), so it does not separate cells from background - it cuts the background in "
      f"half. {haze:.1%} of the 'foreground' lies more than 12 px from any annotated centre (a cell is "
      f"{2 * r1:.0f} px across), while {inside_core:.0%} of the pixels within 3 px of a centre are "
      f"foreground, as they should be")

print()
print("=" * 78)
print("6. WHAT A BLOB TELLS YOU ABOUT HOW MANY CELLS IT HOLDS")
print("=" * 78)
rows = [(r.area, k, r.eccentricity) for lab, ins in zip(blob_lab, inside_lab) for r in regionprops(lab)
        for k in [ins[r.label]] if k > 0]
area, k_of, ecc = np.array([t[0] for t in rows]), np.array([t[1] for t in rows]), np.array([t[2] for t in rows])
inten = np.concatenate([[r.intensity_mean for r in regionprops(lab, intensity_image=x[i])
                         for k in [ins[r.label]] if k > 0] for i, (lab, ins) in enumerate(zip(blob_lab, inside_lab))])
print(f"{'cells in blob':>13s} {'blobs':>7s} {'median area':>12s} {'area / lone cell':>17s} "
      f"{'area per cell / lone cell':>26s} {'median intensity':>17s} {'eccentricity':>13s}")
for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 12), (13, 30), (31, 10 ** 9)):
    m = (k_of >= lo) & (k_of <= hi)
    if m.sum() < 5:
        continue
    label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")
    mid = np.median(k_of[m])                                  # typical k in the bucket, for the ratio
    print(f"{label:>13s} {int(m.sum()):>7d} {np.median(area[m]):>12.0f} {np.median(area[m]) / A1:>17.2f} "
          f"{np.median(area[m]) / (mid * A1):>26.2f} {np.median(inten[m]):>17.3f} {np.median(ecc[m]):>13.2f}")
slope = np.polyfit(k_of[k_of <= 6], area[k_of <= 6], 1)[0]
print(f"two regimes. Up to ~6 cells a blob gains {slope:.0f} px per extra cell, less than the {A1:.0f} px "
      f"a lone cell has - shared boundary, exactly the 'two overlapping cells are smaller than two "
      f"circles' the task describes. Counted over every blob, though, the foreground per cell is "
      f"{np.sum(area) / np.sum(k_of) / A1:.1f} lone cells, because the dense images are one confluent "
      f"sheet with bright haze between the cells: dividing blob area by a cell area therefore overcounts "
      f"wildly (see section 8) and dividing the frame's area by one does not see the cells at all")
print(f"blob intensity grows only from {np.median(inten[k_of == 1]):.3f} (one cell) to "
      f"{np.median(inten[k_of >= 8]) if (k_of >= 8).any() else float('nan'):.3f} (eight or more): "
      f"brightness per pixel says almost nothing about how many cells are stacked, because the sheet is "
      f"saturated - but the INTEGRAL over an image does, see section 8")

print()
print("=" * 78)
print("7. AND 8. THE COUNT DISTRIBUTION AND THE BASELINES, BEFORE ANY TRAINING")
print("=" * 78)
tr, va, te = splits["train"], splits["valid"], splits["test"]
blob_count = np.array([float(l.max()) for l in blob_lab])
fg_area = fg.reshape(n, -1).sum(1).astype(float)
scale = fg_area[tr].sum() / count[tr].sum()                      # the one parameter, fitted on train
split_blobs = np.array([float(np.maximum(1.0, np.round(np.bincount(l.ravel())[1:] / A1)).sum())
                        for l in blob_lab])
print(f"count: train mean {count[tr].mean():.1f} sd {count[tr].std(ddof=1):.1f} | test mean "
      f"{count[te].mean():.1f} sd {count[te].std(ddof=1):.1f} | the test set is the paper's, and its "
      f"images are denser: {count[te].mean() - count[tr].mean():+.0f} cells on average")
print()
integral = x.reshape(n, -1).sum(1)
above = np.array([float(v[v > t].sum()) for v, t in zip(x, thr)])
scaled = lambda f: f / float(np.median((f / count)[tr]))          # the one parameter, fitted on train
print("the two baselines the task names, then three hand-crafted ones that take overlap into account:")
for name, pred in [("constant: train mean", np.full(n, count[tr].mean())),
                   ("constant: train median", np.full(n, float(np.median(count[tr])))),
                   ("Otsu blobs, counted as they are", blob_count),
                   ("Otsu blobs, each split by its area / median cell", split_blobs),
                   (f"foreground area / {scale:.0f} px (one global scale)", fg_area / scale),
                   ("integral of the whole image (total brightness)", scaled(integral)),
                   ("integral above the image's own threshold", scaled(above))]:
    mae = lambda idx: float(np.abs(pred[idx] - count[idx]).mean())
    print(f"{name:50s} MAE train {mae(tr):6.1f}  valid {mae(va):6.1f}  TEST {mae(te):6.1f}")
print(f"\nthe last two matter: fluorescence ADDS UP, so the total brightness of an image is nearly "
      f"proportional to its cell count (correlation {np.corrcoef(integral, count)[0, 1]:.3f}). One "
      f"number fitted on the training set already beats both task baselines by 5x - which means a "
      f"count MAE alone does NOT prove a network found the cells, and this study reports a "
      f"localization metric next to it for that reason.")
print(f"\nblob count vs true count on test: correlation {np.corrcoef(blob_count[te], count[te])[0, 1]:.3f}, "
      f"mean ratio {np.mean(blob_count[te] / count[te]):.2f} -> it finds "
      f"{np.mean(blob_count[te] / count[te]):.0%} of the cells, and is exactly right on "
      f"{np.mean(blob_count[te] == count[te]):.0%} of the images")
np.savez_compressed(FIGS / "inspect_arrays.npz", fg=fg, thr=thr, blob_count=blob_count, single=single,
                    nn=nn, ecc=single_ecc, area=area, k_of=k_of, intensity=inten)

# --------------------------------------------------------------------------------------------- figures
import matplotlib                                             # noqa: E402  (after the Agg-safe imports)
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

i = int(np.argmax(count))                                     # the densest image
sl = (slice(64, 192), slice(64, 192))                         # a 128 x 128 crop, same place everywhere
fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
ax = axes[0]
ax[0].imshow(x[i][sl], cmap="gray", vmin=0, vmax=1)
r_, c_ = np.nonzero(dots[i][sl])
ax[0].plot(c_, r_, "r+", markersize=3, markeredgewidth=0.7)
ax[0].set_title(f"image {i + 1}, a 128x128 crop ({count[i]} cells overall)\nred + mark an annotated centre",
                fontsize=9)
ax[1].imshow(fg[i][sl], cmap="gray")
ax[1].set_title(f"Otsu's foreground (threshold {thr[i]:.2f})\n{frac[i]:.0%} of the whole frame is "
                f"'foreground' - it is a sheet, not cells", fontsize=9)
ax[2].imshow(blob_lab[i][sl], cmap="tab20", interpolation="nearest")
ax[2].set_title(f"its connected components ({blob_lab[i].max()} in the crop)\none blue sheet holds "
                f"nearly all of it - counting these is hopeless", fontsize=9)
for a in ax:
    a.set_xticks([]), a.set_yticks([])

ax = axes[1]
small = k_of <= 8
ax[0].scatter(k_of[small], area[small], s=3, alpha=0.3, label="blobs of 1-8 cells")
big = k_of > 8
ax[0].scatter(k_of[big], area[big], s=3, alpha=0.5, color="crimson", label="blobs of 9+ cells")
ax[0].plot([1, 8], [A1, A1 * 8], "k--", lw=0.8, label="area = k x a lone cell")
ax[0].set_xlabel("cells inside the blob"), ax[0].set_ylabel("blob area (px)")
ax[0].set_yscale("log"), ax[0].legend(fontsize=7)
ax[0].set_title("area grows with cell number, but sublinearly\nand stops being meaningful once the "
                "sheet merges", fontsize=9)
ax[1].scatter(count, frac, s=8, alpha=0.6)
ax[1].axhline(frac.mean(), color="k", lw=0.8, ls="--")
ax[1].set_xlabel("true count"), ax[1].set_ylabel("foreground fraction")
ax[1].set_title(f"coverage is {frac.mean():.3f} whatever the count\n"
                f"(correlation {np.corrcoef(frac, count)[0, 1]:+.3f}) - area carries nothing", fontsize=9)
ax[2].scatter(count, integral, s=8, alpha=0.6)
b, a0 = np.polyfit(count, integral, 1)
ax[2].plot([count.min(), count.max()], [a0 + b * count.min(), a0 + b * count.max()], "k--", lw=0.8)
ax[2].set_xlabel("true count"), ax[2].set_ylabel("integral of the image")
ax[2].set_title(f"total brightness vs count: r = {np.corrcoef(integral, count)[0, 1]:.3f}\nthe answer "
                f"leaks through the exposure", fontsize=9)
fig.suptitle("The VGG cells counting set: the overlap, the saturation of the field, and the "
             "brightness cue", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "inspect_overlap.png", dpi=130)
plt.close("all")
print(f"figures -> {FIGS / 'inspect_overlap.png'} and arrays -> {FIGS / 'inspect_arrays.npz'}")
