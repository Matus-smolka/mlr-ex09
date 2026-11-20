"""How much of the ECG is a real beat but has no label, and where is it? (Task 1 data repair)

Two detectors find QRS-like peaks; a peak more than `tol` samples from any annotated QRS is
"unlabelled". Every such peak is then located relative to the span the cardiologists annotated
[first onset, last offset]:

  head / tail   outside it  -> fixed by trimming the record (or masking the edge)
  interior      inside it   -> trimming does NOT help; needs masking or dropping the record

Usage:  py -I task1_ludb/audit_unlabelled.py <cache.npz> <figs-dir>
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks

FS = 500
WAVES = ["background", "P", "QRS", "T"]
CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
d = np.load(CACHE, allow_pickle=True)
x, y, waves = d["x"], d["y"], d["waves"]
rec_of = waves[:, 0]
n, T = y.shape
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}

B, A = butter(3, [8 / (FS / 2), 20 / (FS / 2)], btype="band")


def qrs_peaks(signals, height):
    """Sensitive QRS detector: envelope of the 8-20 Hz band, peaks above `height` of its own maximum."""
    out = []
    for s in signals:
        env = gaussian_filter1d(np.abs(filtfilt(B, A, s)), 2.0)
        out.append(find_peaks(env, height=height * env.max(), distance=int(0.35 * FS))[0])
    return out


def annotated_qrs(r):
    w = waves[(rec_of == r) & (waves[:, 1] == 2)]
    return np.sort(w[:, 3]) if len(w) else np.array([], int), w


def explain(peak, w):
    """Is this peak inside an annotated wave of any class? (a T wave can also look QRS-ish)"""
    return any(o <= peak <= f for _, _, o, _, f in w)


print("detecting QRS-like peaks that carry no label (tolerance +-50 samples):")
rows = []
for height in (0.3, 0.5):
    head_i = tail_i = interior = 0
    per_record = []
    for r in range(n):
        pk = qrs_peaks(x[r:r + 1], height)[0]
        q, w = annotated_qrs(r)
        un = [p for p in pk if (len(q) == 0 or np.min(np.abs(q - p)) > 50) and not explain(p, w)]
        lo, hi = (w[:, 2].min(), w[:, 4].max()) if len(w) else (0, T - 1)
        h = int(sum(p < lo for p in un))
        t = int(sum(p > hi for p in un))
        i = len(un) - h - t
        head_i, tail_i, interior = head_i + h, tail_i + t, interior + i
        per_record.append((i, h + t, len(q), r))
    print(f"\n  detector threshold {height}: unlabelled beats -> in the head/tail {head_i + tail_i}, "
          f"INTERIOR (trimming will not remove them) {interior}")
    worst = sorted(per_record, reverse=True)[:6]
    print("    records with the most interior unlabelled beats (interior, edge, annotated QRS, record):", worst)
    print("    records with at least one interior unlabelled beat:",
          sum(1 for i, *_ in per_record if i > 0), f"/{n}")
    rows.append((height, per_record))

height = 0.4
pk_all = qrs_peaks(x, height)
interior_recs = []
for r in range(n):
    q, w = annotated_qrs(r)
    lo, hi = (w[:, 2].min(), w[:, 4].max()) if len(w) else (0, T - 1)
    un = [p for p in pk_all[r] if (len(q) == 0 or np.min(np.abs(q - p)) > 50) and not explain(p, w)]
    if any(lo <= p <= hi for p in un):
        interior_recs.append(r)
print(f"\nrecords with >=1 interior unlabelled beat at threshold {height}: {len(interior_recs)} -> "
      f"{interior_recs[:20]}")
print("which split:", {s: sum(1 for r in interior_recs if r in splits[s]) for s in splits})

# how much data does the trim keep?
lo_all, hi_all = [], []
for r in range(n):
    w = waves[rec_of == r]
    lo_all.append(w[:, 2].min() if len(w) else 0)
    hi_all.append(w[:, 4].max() if len(w) else T - 1)
lo_all, hi_all = np.array(lo_all), np.array(hi_all)
print(f"\ntrim to the annotated span keeps {np.median(hi_all - lo_all + 1):.0f} of 5000 samples on average "
      f"(min {int((hi_all - lo_all + 1).min())}, max {int((hi_all - lo_all + 1).max())}); "
      f"that drops {100 * (1 - (hi_all - lo_all + 1).mean() / T):.1f} % of the signal")
pk_all_by_rec = qrs_peaks(x, 0.4)
for pad in (0, 100, 250, 500):
    covered = 0
    for r in range(n):
        q, w = annotated_qrs(r)
        for p in pk_all_by_rec[r]:
            if not (lo_all[r] - pad <= p <= hi_all[r] + pad):
                continue
            if (len(q) == 0 or np.min(np.abs(q - p)) > 50) and not explain(p, w):
                covered += 1
    print(f"  buffer {pad:4d} samples ({pad * 2:4d} ms) around the span re-admits {covered:3d} unlabelled beats")

# --- figures: one clean record, one with an interior unlabelled beat, one with a broken annotation
for r, tag in ((int(splits["test"][0]), "test"), (int(interior_recs[0]), "interior"), (103, "broken lead II")):
    q, w = annotated_qrs(r)
    lo, hi = (w[:, 2].min(), w[:, 4].max()) if len(w) else (0, T - 1)
    pk = pk_all_by_rec[r]
    un = [p for p in pk if (len(q) == 0 or np.min(np.abs(q - p)) > 50) and not explain(p, w)]
    t = np.arange(T) / FS
    plt.figure(figsize=(13, 2.6))
    plt.plot(t, x[r], "k", linewidth=0.6)
    for c, col in ((1, "tab:green"), (2, "tab:red"), (3, "tab:blue")):
        plt.fill_between(t, -4, 4, where=y[r] == c, color=col, alpha=0.35, linewidth=0)
    plt.plot(np.array(un) / FS, x[r][un], "m^", markersize=7, label="QRS-like peak with no label")
    plt.axvline(lo / FS, color="gray", ls=":")
    plt.axvline(hi / FS, color="gray", ls=":")
    plt.ylim(-4.5, 4.5)
    plt.legend(fontsize=8, loc="upper right")
    plt.title(f"record {r} ({tag}): dotted lines = the annotated span [{lo / FS:.2f} s, {hi / FS:.2f} s]", fontsize=10)
    plt.xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(FIGS / f"unlabelled_{tag}_{r}.png", dpi=130)
    print("wrote", FIGS / f"unlabelled_{tag}_{r}.png")
