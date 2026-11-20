"""The baselines, measured before any network is trained — and why the two the task names are weak.

The task prescribes two: answer the training mean for every image, and threshold with Otsu and count the
connected blobs (MAE about 56 and 95 on the test set, it says; both are reproduced). Then come the ones
that take the overlap into account by hand, which is what the exercise is really about:

  bright cores   threshold at a level that isolates the cores rather than Otsu's, count the blobs.
                 Otsu lands *below* the median pixel here (0.25 vs 0.29), so it splits the background
                 rather than the cells; a fixed higher level is the same rule used honestly.
  area split     count = sum over blobs of round(area / area-of-a-cell) — the direct hand-crafted answer
                 to "a big blob is probably two cells".
  brightness     count = integral of the image above a floor, divided by one fitted number. Fluorescence
                 adds up, so this is nearly the count itself; it is included precisely because it shows
                 how little a count MAE alone proves.
  peaks          smooth with a Gaussian and count the maxima — the classical density-free counter, and
                 the only baseline that also gets a localization score.
  target         the density map we will train on (Gaussians on the annotated centres), scored as a
                 counter and as a detector: its recall is the ceiling of peak counting, and its sum
                 shows how much mass a Gaussian loses off the edge of the frame.

Everything with a knob is chosen on the validation split; the test split is evaluated once at the end.

Usage:  py -I task4_cells/classical_cells.py <cache.npz> <figs-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import label as sk_label

sys.path.insert(0, str(Path(__file__).parent))
from cells_metric import evaluate, find_peaks, localize                  # noqa: E402

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
d = np.load(CACHE, allow_pickle=True)
x, dots, count = d["x"], d["dots"], d["count"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
tr, va, te = splits["train"], splits["valid"], splits["test"]
n = len(x)
A1 = 126.0                                      # median area of a blob holding one dot (inspect_cells.py)


def otsu(img, bins=256):
    hist, edges = np.histogram(img, bins=bins, range=(0.0, 1.0))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    centres = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * centres)
    var = np.where(omega * (1 - omega) > 0, omega * (1 - omega) * (mu[-1] - mu) ** 2, 0)
    return float(centres[int(np.argmax(var))])


thr = np.array([otsu(v) for v in x])
BLOBS = {}                                      # (threshold, min area) -> blob counts, cached


def blobs(t, min_area=0):
    """Number of connected bright regions per image at a fixed level. Cached; 8-connectivity."""
    key = (round(t, 3), min_area)
    if key not in BLOBS:
        out = np.zeros(n)
        for i in range(n):
            lab, nb = sk_label(x[i] > t, connectivity=2, return_num=True)
            if min_area:
                sizes = np.bincount(lab.ravel())[1:]
                nb = int((sizes >= min_area).sum())
            out[i] = nb
        BLOBS[key] = out
    return BLOBS[key]


def blob_labels(t):
    return [sk_label(x[i] > t, connectivity=2) for i in range(n)]


def area_split(t, cell_area):
    lab = blob_labels(t)
    return np.array([float(np.maximum(1.0, np.round(np.bincount(l.ravel())[1:] / cell_area)).sum())
                     for l in lab])


mae = lambda pred, idx: float(np.abs(pred[idx] - count[idx]).mean())
rows = {}                                       # name -> (predictions, tuned-from, extra info)

print("=" * 78)
print("1. THE TWO BASELINES THE TASK NAMES")
print("=" * 78)
rows["constant, train mean"] = (np.full(n, count[tr].mean()), "no parameters")
rows["constant, train median"] = (np.full(n, float(np.median(count[tr]))), "no parameters")
otsu_blobs = np.array([float(sk_label(x[i] > thr[i], connectivity=2, return_num=True)[1]) for i in range(n)])
rows["Otsu threshold, count blobs"] = (otsu_blobs, "no parameters (Otsu per image)")
for name, (pred, how) in rows.items():
    print(f"{name:44s} MAE train {mae(pred, tr):6.1f}  valid {mae(pred, va):6.1f}  TEST {mae(pred, te):6.1f}"
          f"   [{how}]")

print()
print("=" * 78)
print("2. THRESHOLD AND BLOB COUNT, SWEEPED ON VALIDATION")
print("=" * 78)
print(f"{'threshold':>9s} {'min area':>9s} {'blobs/image':>12s} {'MAE valid':>10s} {'MAE test':>9s}")
best = None
for t in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
    for min_area in (0, 20, 40):
        pred = blobs(t, min_area)
        m_v, m_t = mae(pred, va), mae(pred, te)
        if best is None or m_v < best[0]:
            best = (m_v, t, min_area, pred)
        if min_area == 0 or t in (0.5, 0.6, 0.7, 0.8):
            print(f"{t:9.2f} {min_area:9d} {pred.mean():12.1f} {m_v:10.1f} {m_t:9.1f}")
print(f"best on validation: threshold {best[1]:.2f}, min area {best[2]} -> valid {best[0]:.1f}, "
      f"test {mae(best[3], te):.1f} (it counts {best[3].mean():.0f} blobs per image against "
      f"{count.mean():.0f} cells)")
rows[f"bright cores at {best[1]:.2f}, blobs counted"] = (best[3], f"tuned on validation: t={best[1]:.2f}, "
                                                         f"min area {best[2]}")

print()
print("=" * 78)
print("3. AREA SPLITTING: 'A BIG BLOB IS PROBABLY TWO CELLS'")
print("=" * 78)
best_split = None
for t in (0.40, 0.50, 0.60, 0.70):
    for cell_area in (60, 90, 126, 160, 200, 250):
        pred = area_split(t, cell_area)
        if best_split is None or mae(pred, va) < best_split[0]:
            best_split = (mae(pred, va), t, cell_area, pred)
    print(f"  threshold {t:.2f}: best so far valid {best_split[0]:6.1f} at cell area {best_split[2]:3d} px")
bias_split = float((best_split[3][va] - count[va]).mean())
print(f"best: threshold {best_split[1]:.2f}, divisor {best_split[2]} px -> valid {best_split[0]:.1f}, "
      f"test {mae(best_split[3], te):.1f}, bias {bias_split:+.1f} cells. This is the hypothesis the task "
      f"is built on, working as an explicit rule: a bright region of twice the area is counted as two "
      f"cells. Counting the same blobs as they are gives {mae(blobs(best_split[1]), te):.1f} on test, so "
      f"the split alone is worth {mae(blobs(best_split[1]), te) - mae(best_split[3], te):.1f} cells. The "
      f"divisor it wants ({best_split[2]} px) is smaller than a lone cell ({A1:.0f} px), because at this "
      f"threshold every cell sits inside a fattened sheet - section 5's saturation, absorbed into one "
      f"fitted number.")
rows[f"blobs at {best_split[1]:.2f}, area / {best_split[2]} px"] = (
    best_split[3], f"tuned on validation: t={best_split[1]:.2f}, divisor {best_split[2]} px")

print()
print("=" * 78)
print("4. TOTAL BRIGHTNESS (the leak): count = sum(relu(image - floor)) / scale")
print("=" * 78)
best_int = None
for t in (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
    f = np.array([np.maximum(v - t, 0).sum() for v in x])
    scale = float(np.median((f / count)[tr]))
    pred = f / scale
    if best_int is None or mae(pred, va) < best_int[0]:
        best_int = (mae(pred, va), t, scale, pred)
    print(f"  floor {t:.2f}: scale {scale:8.1f}, MAE valid {mae(pred, va):5.1f}, test {mae(pred, te):5.1f}")
print(f"best: floor {best_int[1]:.2f} -> valid {best_int[0]:.1f}, test {mae(best_int[3], te):.1f} with "
      f"exactly ONE fitted number. No count MAE below ~10 here means the model found cells; it may have "
      f"found the exposure instead.")
rows[f"brightness sum above {best_int[1]:.2f} / {best_int[2]:.0f}"] = (
    best_int[3], f"one parameter fitted on train, floor chosen on validation")

print()
print("=" * 78)
print("5. PEAKS OF A SMOOTHED IMAGE (the classical density-free counter, and it can be localized)")
print("=" * 78)
print(f"{'sigma':>6s} {'frac':>6s} {'peaks/image':>12s} {'MAE valid':>10s} {'recall':>7s} {'precision':>10s}")
best_pk = None
for sigma in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    sm = np.array([ndimage.gaussian_filter(v, sigma) for v in x])
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        xy = [find_peaks(sm[i], 3, frac * sm[i].max())[0] for i in range(n)]
        pred = np.array([len(p) for p in xy], float)
        if best_pk is None or mae(pred, va) < best_pk[0]:
            loc = localize(sm[va], dots[va], tol=5.0, frac=frac, min_distance=3)
            best_pk = (mae(pred, va), sigma, frac, pred, loc)
        if sigma in (1.0, 2.0, 3.0) and frac in (0.3, 0.5, 0.7):
            loc = localize(sm[va], dots[va], tol=5.0, frac=frac, min_distance=3)
            print(f"{sigma:6.1f} {frac:6.2f} {pred.mean():12.1f} {mae(pred, va):10.1f} "
                  f"{loc['recall']:7.3f} {loc['precision']:10.3f}")
sm_best = np.array([ndimage.gaussian_filter(v, best_pk[1]) for v in x])
loc_test = localize(sm_best[te], dots[te], tol=5.0, frac=best_pk[2], min_distance=3)
print(f"best on validation: sigma {best_pk[1]:.1f}, level {best_pk[2]:.2f} x max -> valid "
      f"{best_pk[0]:.1f}, test {mae(best_pk[3], te):.1f}; on the validation split it finds "
      f"{best_pk[4]['recall']:.1%} of the centres at {best_pk[4]['precision']:.1%} precision, on test "
      f"{loc_test['recall']:.1%} at {loc_test['precision']:.1%}")
rows[f"peaks of a sigma={best_pk[1]:.1f} blur"] = (best_pk[3], f"tuned on validation: sigma={best_pk[1]:.1f}, "
                                                   f"level={best_pk[2]:.2f}")

print()
print("=" * 78)
print("6. THE DENSITY TARGET ITSELF: what a perfect map could score")
print("=" * 78)
print(f"{'sigma':>6s} {'sum - count, reflect':>19s} {'sum - count, zeros':>19s} {'mean, zeros':>12s} "
      f"{'recall (val)':>13s} {'precision (val)':>16s} {'F1 (val)':>9s}")
best_sig = None
for sigma in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    maps = np.array([ndimage.gaussian_filter(dd.astype(np.float64), sigma) for dd in dots])
    cut = np.array([ndimage.gaussian_filter(dd.astype(np.float64), sigma, mode="constant") for dd in dots])
    rel = np.abs(maps.reshape(n, -1).sum(1) - count) / count
    rel_c = np.abs(cut.reshape(n, -1).sum(1) - count) / count
    loc = localize(maps[va], dots[va], tol=5.0, frac=0.25, min_distance=3)
    if best_sig is None or loc["f1"] > best_sig[0]:
        best_sig = (loc["f1"], sigma, maps)
    print(f"{sigma:6.1f} {rel.max() * 100:18.3f}% {rel_c.max() * 100:18.3f}% {rel_c.mean() * 100:11.3f}% "
          f"{loc['recall']:13.3f} {loc['precision']:16.3f} {loc['f1']:9.3f}")
loc_t = localize(best_sig[2][te], dots[te], tol=5.0, frac=0.25, min_distance=3)
print(f"sigma {best_sig[1]:.1f} has the best peak F1 on validation; on test the same map reaches recall "
      f"{loc_t['recall']:.3f}, precision {loc_t['precision']:.3f}, F1 {loc_t['f1']:.3f}. That is the "
      f"CEILING of peak counting, set by the "
      f"{int(count.sum()) - sum(int(sk_label(dd, connectivity=2, return_num=True)[1]) for dd in dots)} "
      f"pairs of centres closer than a pixel apart and by how wide the bumps are - not by the network.")
print("The two sum columns answer the notebook's 'check that your target sums to the count': scipy's "
      "gaussian_filter\nreflects at the border by default, which folds the truncated tail back in, so the "
      "map sums to the\ncount to 5 decimals at every sigma tried. Cutting the kernel off instead (mode="
      "'constant', what a\nconvolution layer with zero padding does) loses up to the printed percentage for "
      "the worst of the\n441 border centres - so the target is built on the CPU with gaussian_filter and "
      "the network is asked\nto reproduce a map whose sum is already exact, rather than being left to "
      "reconstruct mass that its\nown convolutions dropped at the edge.")

print()
print("=" * 78)
print("7. ALL OF THEM ON THE TEST SPLIT, ONCE")
print("=" * 78)
LOCALIZED = {f"peaks of a sigma={best_pk[1]:.1f} blur": loc_test}      # on the test split
res = {}
for name, (pred, how) in rows.items():
    r = evaluate(count[te], pred[te])
    if name in LOCALIZED:
        r.update(LOCALIZED[name])
    res[name] = {"how": how, "test": r, "valid_mae": mae(pred, va)}
    extra = (f"  recall {r['recall']:.3f} precision {r['precision']:.3f} F1 {r['f1']:.3f}"
             if "recall" in r else "")
    print(f"{name:44s} MAE {r['mae']:6.1f}  RMSE {r['rmse']:6.1f}  bias {r['bias']:+7.1f}  "
          f"R2 {r['r2']:6.3f}  worst {r['worst']:5.0f}{extra}")
print(f"\nthe task's targets: constant about 56, Otsu blobs about 95, and 'well under 10' for a network. "
      f"Best hand rule here: {min(res, key=lambda k: res[k]['test']['mae'])} at MAE "
      f"{min(r['test']['mae'] for r in res.values()):.1f} — so a network must beat ~11 and must do it "
      f"with localization, or it has learned the exposure.")
(FIGS / "results_classical.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(FIGS / "classical_predictions.npz",
                    **{k.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""): v[0]
                       for k, v in rows.items()})
print(f"wrote {FIGS / 'results_classical.json'} and classical_predictions.npz")
