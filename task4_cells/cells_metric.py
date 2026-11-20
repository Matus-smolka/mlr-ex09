"""How well did it count — and did it count for the right reason?

The task scores one number: the mean absolute error of the count. On this data that number can be won
without looking at a single cell, because the integral of an image correlates 0.991 with its cell count
(see inspect_cells.py section 8). So every count here is reported next to a **localization** score: turn
the map into peaks, match each peak greedily — strongest first — to the nearest unused annotated centre
within `tol` px, and ask how many of the cells were actually found. A model that only adds up brightness
scores a perfect count and zero recall; a model that sees the cells scores both.

`evaluate` also scores the *target* map itself, which is the ceiling of peak counting: two cells whose
centres are 1 px apart cannot be separated by any smooth map, and the target's recall says so.

Usage: as a module.  py -I task4_cells/cells_metric.py runs the self-check.
"""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree


def find_peaks(density, min_distance=3, threshold=0.0):
    """Positions and strengths of the local maxima of a map, in a (2*d+1)^2 neighbourhood.

    A maximum is a pixel that is at least as bright as every pixel within `min_distance` of it. Ties
    (a flat ridge, a plateau) would otherwise be reported as several peaks, so the maxima are thinned
    strongest-first: a kept maximum suppresses everything still within `min_distance` of it.
    """
    density = np.asarray(density, dtype=np.float64)
    peak = (density >= maximum_filter(density, size=2 * int(min_distance) + 1, mode="nearest")) \
        & (density > threshold)
    rows, cols = np.nonzero(peak)
    xy, strength = np.stack([rows, cols], 1), density[rows, cols]
    if len(xy) < 2:
        return xy, strength
    tree, alive = cKDTree(xy), np.ones(len(xy), bool)
    for i in np.argsort(-strength):                       # strongest first, plateaus collapse to one
        if alive[i]:
            for j in tree.query_ball_point(xy[i], min_distance - 1e-6):
                alive[j] = j == i
    return xy[alive], strength[alive]


def match_detail(peak_xy, strength, true_xy, tol=5.0):
    """Greedy strongest-first matching, each centre used once.

    Returns (pairs, unmatched peaks, unmatched centres): `pairs` is one (peak row, centre row) per
    cell the map actually found, so a figure can draw what was hit, what was invented and what was
    missed instead of only counting them. A peak looks at its nearest centres in order and takes the
    first still free one within `tol`.
    """
    peak_xy, strength = np.asarray(peak_xy).reshape(-1, 2), np.asarray(strength).ravel()
    true_xy = np.asarray(true_xy).reshape(-1, 2)
    if len(peak_xy) == 0:
        return [], [], list(range(len(true_xy)))
    tree = cKDTree(true_xy)
    k = min(8, len(true_xy))
    dist, idx = tree.query(peak_xy, k=k)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    taken, pairs = np.zeros(len(true_xy), bool), []
    for p in np.argsort(-strength):
        for dd, jj in zip(dist[p], idx[p]):
            if dd > tol:
                break
            if not taken[jj]:
                taken[jj] = True
                pairs.append((p, jj))
                break
    matched_peaks = {p for p, _ in pairs}
    return pairs, [p for p in range(len(peak_xy)) if p not in matched_peaks], \
        [j for j in range(len(true_xy)) if not taken[j]]


def match(peak_xy, strength, true_xy, tol=5.0):
    """The same matching as a triple of numbers: (tp, n_peaks, n_true)."""
    peak_xy = np.asarray(peak_xy).reshape(-1, 2)
    pairs, fp, fn = match_detail(peak_xy, strength, true_xy, tol)
    return len(pairs), len(peak_xy), len(true_xy)


def locate_detail(m, d, tol=5.0, frac=0.25, min_distance=3):
    """One map against one dot image: (peak, centre) pairs, spurious peaks, missed centres."""
    xy, s = find_peaks(np.asarray(m, dtype=np.float64), min_distance, frac * float(np.max(m)))
    pairs, fp, fn = match_detail(xy, s, np.argwhere(np.asarray(d) > 0), tol)
    return xy, pairs, fp, fn


def localize(maps, dots, tol=5.0, frac=0.25, min_distance=3):
    """Micro precision / recall / F1 of the peaks of every map against the annotated centres.

    `frac` is relative to each map's own maximum, so the score does not depend on the units the network
    happens to paint in. `min_distance` is what two peaks must be apart to count as two cells: a cell is
    ~13 px across, so 3 px is deliberately permissive and 6 px is the honest physical limit.
    """
    tp = fp = fn = ntr = 0
    for m, d in zip(maps, dots):
        _, pairs, f, g = locate_detail(m, d, tol, frac, min_distance)
        tp += len(pairs)
        fp += len(f)
        fn += len(g)
        ntr += int((np.asarray(d) > 0).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"tp": int(tp), "n_peaks": int(tp + fp), "n_true": int(ntr),
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9))}


def evaluate(count_true, count_pred, maps=None, dots=None, tol=5.0, frac=0.25, min_distance=3):
    """Count MAE / RMSE / bias, and — when the maps are given — the localization score."""
    a, b = np.asarray(count_true, float), np.asarray(count_pred, float)
    err = b - a
    spread = float(((a - a.mean()) ** 2).sum())
    out = {"n": int(len(a)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "bias": float(err.mean()), "r2": float(1 - (err ** 2).sum() / spread) if spread > 0 else float("nan"),
           "rel_mae": float((np.abs(err) / np.maximum(a, 1)).mean()), "worst": float(np.abs(err).max())}
    if maps is not None:
        out.update(localize(maps, dots, tol, frac, min_distance))
    return out


def fmt(res, keys=("mae", "rmse", "bias", "r2", "precision", "recall", "f1")):
    return "  ".join(f"{k} {res[k]:.3f}".rstrip("0").rstrip(".") if isinstance(res[k], float)
                     else f"{k} {res[k]}" for k in keys if k in res)


if __name__ == "__main__":
    # a map with two bumps and one spurious one, against two centres 20 px apart
    tgt = np.zeros((64, 64))
    tgt[20, 20] = tgt[40, 40] = 1.0
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(tgt, 3)
    good = evaluate([2], [2], [smooth], [tgt], tol=5)
    off_by_one = evaluate([2], [3])
    shifted = np.roll(smooth, 9, axis=0)                      # bumps 9 px away from the centres
    bad = evaluate([2], [2], [shifted], [tgt], tol=5)
    print("perfect map        ", fmt(good))
    assert good["recall"] == 1.0 and good["mae"] == 0.0, good
    print("count off by one   ", fmt(off_by_one))
    assert off_by_one["mae"] == 1.0 and "recall" not in off_by_one
    print("map shifted 9 px   ", fmt(bad))
    assert bad["mae"] == 0.0 and bad["recall"] == 0.0, bad      # the count lies, the localization does not
    # two centres 1 px apart: no smooth map, and no peak finder with a 3 px min distance, can split them
    pair = np.zeros((64, 64))
    pair[30, 30] = pair[30, 31] = 1.0
    res = evaluate([2], [2], [gaussian_filter(pair, 3)], [pair], tol=2, min_distance=3)
    print("centres 1 px apart  ", fmt(res), "<- the ceiling peak counting sets on such pairs:",
          f"recall {res['recall']:.2f}, because the map has one bump, not two")
    assert res["recall"] == 0.5, res
    print("cells_metric self-check ok")
