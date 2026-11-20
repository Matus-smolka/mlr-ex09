"""The figures and the tables Task 4 has to hand in, drawn from what train_cells.py saved.

Nothing is trained here: the maps of the chosen network are read from figs/preds_test.npz and the
numbers from figs/results_net.json (network) and figs/results_classical.json (baselines), so the
figures can be redrawn without touching the GPU and always show the run that was scored.

    figs/target_sum_check.png   the notebook's check: the target of a few training images, with its sum
    figs/scatter_test.png       predicted against true count for all 100 test images, with the diagonal,
                                next to the blob baseline the task names and the brightness rule
    figs/one_test_map.png       one test image: the image, its centres, its target, its prediction, each summed
    figs/worst_test_maps.png    the three test images the network counts worst, and what they look like
    figs/localization.png       which cells the predicted map really found (hit / invented / missed)
    figs/sweep_validation.png   every arm tried on validation, by count MAE

and a printed table of how much the localization score depends on its two knobs.

Usage:  py -I task4_cells/report_cells.py <cache.npz> <figs-dir>
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from scipy import ndimage                                # noqa: E402

from cells_metric import evaluate, find_peaks, locate_detail, localize    # noqa: E402

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(CACHE, allow_pickle=True)
x, dots, count = d["x"], d["dots"], d["count"].astype(np.float32)
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
te = splits["test"]
res = json.loads((FIGS / "results_net.json").read_text())
P = np.load(FIGS / "preds_test.npz")
seeds = sorted(int(re.match(r"maps_seed(\d+)_test$", k).group(1)) for k in P.files
               if re.match(r"maps_seed\d+_test$", k))
SE = np.mean([P[f"maps_seed{s}_test"] for s in seeds], 0)          # the ensemble's 100 test maps
SIGMA = float(P["sigma"])
BASE = json.loads((FIGS / "results_classical.json").read_text())
CP = np.load(FIGS / "classical_predictions.npz")


def key_of(name):
    """How classical_cells.py named its predictions inside classical_predictions.npz."""
    return name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")


def pick_base(*needles):
    for name in BASE:
        if all(n in name for n in needles):
            return name
    return None


def sum_of(maps):
    return maps.reshape(len(maps), -1).sum(1)


def target(i):
    return ndimage.gaussian_filter(dots[i].astype(np.float64), SIGMA)


ens = evaluate(count[te], sum_of(SE), SE, dots[te])
B_CONST, B_BLOBS = pick_base("constant"), pick_base("Otsu")
B_LEAK = pick_base("brightness sum")
print(f"chosen arm {json.dumps(res['arm'])} | {res['arms_tried_on_validation']} arms tried on validation "
      f"| {res['parameters'] / 1e6:.1f} M parameters | sigma {SIGMA}")
print(f"ensemble of seeds {seeds} on the test split: MAE {ens['mae']:.2f}, RMSE {ens['rmse']:.2f}, "
      f"bias {ens['bias']:+.2f}, R2 {ens['r2']:.3f}, worst {ens['worst']:.0f}, recall {ens['recall']:.3f}, "
      f"precision {ens['precision']:.3f}, F1 {ens['f1']:.3f}")
for s in seeds:
    r = res["seeds"][str(s)]["test"]
    print(f"   seed {s}: MAE {r['mae']:.2f}, bias {r['bias']:+.2f}, recall {r['recall']:.3f}")

# --------------------------------------------------------------------------- 1. the target's sum
fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
for col, i in enumerate([0, 1, int(np.argmax(count[splits["train"]]))]):
    axes[0, col].imshow(x[i], cmap="gray")
    axes[0, col].set_title(f"training image {i + 1}: {count[i]:.0f} cells", fontsize=9)
    axes[1, col].imshow(target(i), cmap="magma")
    axes[1, col].set_title(f"target, sum {target(i).sum():.3f} "
                           f"(ratio to the count {target(i).sum() / count[i]:.5f})", fontsize=9)
    for ax in (axes[0, col], axes[1, col]):
        ax.axis("off")
fig.suptitle(f"the target is a Gaussian (sigma {SIGMA}) on every annotated centre, so its sum IS the count; "
             "scipy's reflecting border keeps that exact", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "target_sum_check.png", dpi=130)

# --------------------------------------------------------------------------- 2. scatter, test split
lo, hi = float(count[te].min()) - 15, float(count[te].max()) + 15
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
axes[0].plot([lo, hi], [lo, hi], "k", lw=0.9)
axes[0].axhline(float(count[splits["train"]].mean()), color="0.45", ls="--", lw=1,
                label=f"{B_CONST} (MAE {BASE[B_CONST]['test']['mae']:.1f})")
axes[0].scatter(count[te], sum_of(SE), s=20, c="tab:blue", alpha=0.8, label="network, 3 seeds averaged")
axes[0].set_xlim(lo, hi)
axes[0].set_ylim(lo, hi)
axes[0].set_title(f"network, test split: MAE {ens['mae']:.1f}, R2 {ens['r2']:.3f}", fontsize=9)
axes[0].legend(fontsize=7, loc="lower right")
for ax, name, colour, ymax in ((axes[1], B_BLOBS, "tab:red", hi),
                               (axes[2], B_LEAK, "tab:green", hi)):
    pred = CP[key_of(name)]
    ax.plot([lo, hi], [lo, hi], "k", lw=0.9)
    ax.scatter(count[te], pred[te], s=20, c=colour, alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, max(ymax, float(pred.max()) + 10))
    ax.set_title(f"{name}\nMAE {BASE[name]['test']['mae']:.1f}, correlation with the true count "
                 f"{np.corrcoef(count[te], pred[te])[0, 1]:+.3f}", fontsize=9)
for ax in axes:
    ax.set_xlabel("true count")
    ax.set_ylabel("the answer given")
fig.suptitle("all 100 test images. Left: the network. Middle: the baseline the task names (touching cells "
             "merge, so it falls below the diagonal and correlates NEGATIVELY). Right: total brightness, "
             "which is nearly the count itself", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "scatter_test.png", dpi=130)

# --------------------------------------------------------------------------- 3. one test image, four views
err = np.abs(sum_of(SE) - count[te])
median_count = float(np.median(count[te]))
i_mid = int(np.argmin(np.abs(count[te] - median_count) + np.abs(err - np.median(err))))
j = int(te[i_mid])
xy, pairs, fp, fn = locate_detail(SE[i_mid], dots[j])
tp_xy = np.array([xy[p] for p, _ in pairs]).reshape(-1, 2)
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
axes[0].imshow(x[j], cmap="gray")
axes[0].set_title(f"test image {j + 1}", fontsize=9)
axes[1].imshow(x[j], cmap="gray")
rows, cols = np.nonzero(dots[j])
axes[1].plot(cols, rows, "r.", markersize=2)
axes[1].set_title(f"the annotated {count[j]:.0f} centres", fontsize=9)
axes[2].imshow(target(j), cmap="magma")
axes[2].set_title(f"target map, sum {target(j).sum():.1f}", fontsize=9)
axes[3].imshow(SE[i_mid], cmap="magma")
axes[3].plot(tp_xy[:, 1], tp_xy[:, 0], "o", mfc="none", mec="lime", ms=5, mew=0.8)
if len(fp):
    axes[3].plot(xy[fp][:, 1], xy[fp][:, 0], "x", color="red", ms=5, mew=1.2)
if len(fn):
    axes[3].plot(xy[fn][:, 1], xy[fn][:, 0], "s", mfc="none", mec="deepskyblue", ms=5, mew=1.0)
axes[3].set_title(f"predicted map, sum {sum_of(SE)[i_mid]:.1f} (true {count[j]:.0f}, "
                  f"error {sum_of(SE)[i_mid] - count[j]:+.1f})", fontsize=9)
for ax in axes:
    ax.axis("off")
fig.suptitle("one test image with its target and its predicted density map, each with its sum; on the last "
             "panel green circle = a peak that lands on a centre, red cross = one that lands nowhere, "
             "blue square = a centre with no peak near it", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "one_test_map.png", dpi=130)

# --------------------------------------------------------------------------- 4. the worst three
worst = np.argsort(-err)[:3]
fig, axes = plt.subplots(3, 3, figsize=(11, 11))
for r, k in enumerate(worst):
    j = int(te[k])
    axes[r, 0].imshow(x[j], cmap="gray")
    axes[r, 0].set_title(f"image {j + 1}: {count[j]:.0f} cells", fontsize=9)
    axes[r, 1].imshow(target(j), cmap="magma")
    axes[r, 1].set_title(f"target, sum {target(j).sum():.0f}", fontsize=9)
    axes[r, 2].imshow(SE[k], cmap="magma")
    axes[r, 2].set_title(f"predicted, sum {sum_of(SE)[k]:.0f} "
                         f"({sum_of(SE)[k] - count[j]:+.0f} cells)", fontsize=9)
    for c in range(3):
        axes[r, c].axis("off")
fig.suptitle("the three test images the ensemble counts worst", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "worst_test_maps.png", dpi=130)

# --------------------------------------------------------------------------- 5. localization evidence
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
show = np.argsort([len(locate_detail(SE[k], dots[int(te[k])])[3]) for k in range(len(te))])[::-1][:3]
for col, k in enumerate(show):
    j = int(te[k])
    xy, pairs, fp, fn = locate_detail(SE[k], dots[j])
    canvas = np.stack([x[j], 0.35 * x[j], 0.35 * x[j], ], -1)
    axes[col].imshow(canvas)
    tp_xy = np.array([xy[p] for p, _ in pairs]).reshape(-1, 2)
    if len(tp_xy):
        axes[col].plot(tp_xy[:, 1], tp_xy[:, 0], "o", mfc="none", mec="lime", ms=6, mew=1.0)
    if len(fp):
        axes[col].plot(xy[fp][:, 1], xy[fp][:, 0], "x", color="red", ms=6, mew=1.3)
    if len(fn):
        axes[col].plot(xy[fn][:, 1], xy[fn][:, 0], "s", mfc="none", mec="deepskyblue", ms=6, mew=1.0)
    axes[col].set_title(f"image {j + 1}: {len(pairs)} of {count[j]:.0f} cells found, {len(fp)} invented, "
                        f"{len(fn)} missed\nsum {sum_of(SE)[k]:.0f} against {count[j]:.0f}", fontsize=9)
    axes[col].axis("off")
fig.suptitle("did it count for the right reason? the three test images where the predicted map misses the "
             "most cells", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "localization.png", dpi=130)

# --------------------------------------------------------------------------- 6. the arms, on validation
sweep = res["validation_sweep"]
keys = sorted(sweep, key=lambda k: sweep[k]["valid_mae"])
fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(keys) + 1.8))
ax.barh(range(len(keys)), [sweep[k]["valid_mae"] for k in keys], color="tab:blue")
ax.set_yticks(range(len(keys)), keys, fontsize=7)
ax.invert_yaxis()
for name, colour in ((B_BLOBS, "tab:red"), (B_CONST, "0.45")):
    ax.axvline(BASE[name]["valid_mae"], color=colour, ls="--", lw=1,
               label=f"{name} (valid {BASE[name]['valid_mae']:.1f})")
ax.set_xlabel("count MAE on the 20 validation images")
ax.set_title("every arm tried before the network was fixed; the best is on top", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "sweep_validation.png", dpi=130)

# --------------------------------------------------------------------------- 7. how solid is that score?
print("\nlocalization on the test split against its two knobs (tol = how close to a centre a peak has to "
      "land,\nmin distance = how far apart two peaks must be to count as two cells, level = the peak "
      "height kept\nas a fraction of that map's own maximum):")
print(f"{'map':22s} {'tol':>4s} {'min dist':>9s} {'level':>6s} {'recall':>7s} {'precision':>10s} {'F1':>6s}")
tgt_te = np.stack([target(int(i)) for i in te])
sens = {}
for what, maps in (("predicted map", SE), ("target map (ceiling)", tgt_te)):
    for tol in (3.0, 5.0, 8.0):
        for md in (3, 6):
            for frac in (0.10, 0.25, 0.40):
                r = localize(maps, dots[te], tol=tol, frac=frac, min_distance=md)
                sens[f"{what}|tol{tol:g}|mindist{md}|level{frac:g}"] = r
                if tol == 5.0 and md == 3:
                    print(f"{what:22s} {tol:4.0f} {md:9d} {frac:6.2f} {r['recall']:7.3f} "
                          f"{r['precision']:10.3f} {r['f1']:6.3f}")
peaks = np.array([len(find_peaks(SE[k], 3, 0.25 * SE[k].max())[0]) for k in range(len(te))])
by_peak = evaluate(count[te], peaks)
print(f"\nthe same maps read two ways: the sum of the map -> MAE {ens['mae']:.2f}; counting its peaks -> "
      f"MAE {by_peak['mae']:.2f} (bias {by_peak['bias']:+.1f}). The task asks for the sum, and the loss "
      f"optimizes it; the peak count is what a biologist would do with the picture.")

out = {"ensemble_test": ens, "per_seed_test": {str(s): res["seeds"][str(s)]["test"] for s in seeds},
       "counted_by_peaks": by_peak, "localization_sensitivity": sens,
       "worst_test_images": [int(te[k]) for k in worst], "shown_test_image": int(te[i_mid])}
(FIGS / "results_report.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote target_sum_check.png, scatter_test.png, one_test_map.png, worst_test_maps.png, "
      f"localization.png, sweep_validation.png and results_report.json to {FIGS}")
