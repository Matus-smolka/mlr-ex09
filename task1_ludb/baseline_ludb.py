"""Primitive (no-network) baseline for Task 1: ECG delineation of P, QRS and T.

Find the R peaks with find_peaks, then paint P, QRS and T at the median distances from the R peak
measured on the TRAINING patients. Every choice (detector, threshold, refractory gap, smoothing)
is made on the VALIDATION split; the test split is evaluated once, at the end.

Reported twice from the same predictions:
  judged   - ignoring the samples repair_ludb.py marked -1 (the unannotated beats at the record
             edges). This is the number the network has to beat.
  as parsed- every sample counted, the way the notebook's own reference baseline does it, so the
             two are comparable with the reference quoted in the notebook.

Usage:  py -I task1_ludb/baseline_ludb.py <masked_cache.npz> <figs-dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))           # so ludb_metric imports however this is run

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classical import FS, detect, median_offsets, paint
from ludb_metric import WAVES, evaluate, fmt

CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
d = np.load(CACHE, allow_pickle=True)
x, waves, y_raw = d["x"], d["waves"], d["y_raw"]
# every way of deciding which samples count; the first is the one we report (repair_ludb.py explains them)
CONVENTIONS = {"span + buffer": d["y"], "span, no buffer": d["y_tight"], "signal edges": d["ig_signal"],
               "as parsed": None}
PRIMARY = "span + buffer"
for name, m in CONVENTIONS.items():
    if m is not None:
        assert (m == -1).any() and (m >= 0).any(), f"{name}: a 0/1 mask silently counts every sample"
assert (CONVENTIONS["span, no buffer"] == -1).sum() > (CONVENTIONS[PRIMARY] == -1).sum()  # buffer keeps more
mask = CONVENTIONS[PRIMARY]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of = waves[:, 0]
T = x.shape[1]


def score(idx, kind, frac, distance, sigma, offsets, rr_ref=None, conv=PRIMARY, preds=None):
    """Predictions from one detector setting, scored under one convention (default: the one we report)."""
    if preds is None:
        preds = np.array([paint(p, offsets, rr_ref) for p in detect(x[idx], kind, frac, distance, sigma)])
    m = CONVENTIONS[conv]
    return evaluate(y_raw[idx], preds, None if m is None else m[idx]), preds


# --- the wave intervals, measured on the training patients against their annotated R peaks
offsets = median_offsets(waves, rec_of, splits["train"])
print("median wave intervals from the 140 training patients (samples from the R peak; 1 sample = 2 ms):")
for c in (1, 2, 3):
    a, b = offsets[c]
    print(f"  {WAVES[c]:4s} onset {a:+4d}  offset {b:+4d}   -> {b - a} samples = {(b - a) * 2} ms wide")
RR_REF = float(np.median(np.concatenate([np.diff(np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3]))
                                         for r in splits["train"] if ((rec_of == r) & (waves[:, 1] == 2)).sum() > 1])))
print(f"median R-R interval of the training patients: {RR_REF:.0f} samples = {RR_REF * 2:.0f} ms")


def label(kind):
    return f"{kind[0]}{list(kind[1])}" if isinstance(kind, tuple) else kind


print(f"\nsearching the detector on the validation split, judged under the '{PRIMARY}' convention:")
def under(idx, preds):
    """The same predictions scored under every convention - choosing the region is cheap, detecting is not."""
    return {c: evaluate(y_raw[idx], preds, None if m is None else m[idx]) for c, m in CONVENTIONS.items()}


def mean_macro(s):
    return float(np.mean([v["macro"] for v in s.values()]))


results = []
for kind in ["thresh", "smooth", "energy", ("bandpass", (4, 15)), ("bandpass", (8, 20)), ("bandpass", (1, 30))]:
    for sigma in [0, 2, 4]:
        for distance in [150, 200, 250, 300]:
            for frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for rr_ref in [None, RR_REF]:
                    preds = np.array([paint(p, offsets, rr_ref)
                                      for p in detect(x[splits["valid"]], kind, frac, distance, sigma)])
                    per = under(splits["valid"], preds)
                    results.append((mean_macro(per[PRIMARY]), kind, frac, distance, sigma,
                                    rr_ref is not None, per[PRIMARY], per))
results.sort(key=lambda t: -t[0])
print(f"{'detector':17s} {'frac':>5s} {'dist':>5s} {'sigma':>6s} {'RR':>4s}   validation macro Dice P/QRS/T    mean")
for t in results[:8]:
    mean, kind, frac, distance, sigma, rr_on, s = t[:7]
    print(f"{label(kind):17s} {frac:5.2f} {distance:5d} {sigma:6.1f} {'yes' if rr_on else 'no':>4s}   "
          + " / ".join(f"{s[WAVES[c]]['macro']:.3f}" for c in (1, 2, 3)) + f"    {mean:.3f}")
print("  ...and the winner under each other way of choosing the judged region (does the choice of")
print("     convention change which detector we would have picked?)")
for c in CONVENTIONS:
    t = max(results, key=lambda t: mean_macro(t[7][c]))
    print(f"     {c:16s} {label(t[1]):17s} frac {t[2]:.2f} dist {t[3]:3d} sigma {t[4]:.1f} "
          f"RR {'yes' if t[5] else 'no':>3s}   " + fmt(t[7][c]) + f"    {mean_macro(t[7][c]):.3f}")

naive = next(r for r in results if r[1] == "thresh" and r[2] == 0.6 and r[3] == 250 and r[4] == 0 and not r[5])
print(f"\nyour rule as proposed (thresh at 60 % of the record maximum, 250-sample gap, fixed windows): "
      + fmt(naive[6]) + f"   mean {naive[0]:.3f}")

# --- the test split: the naive rule, the tuned one and the ceiling, under every convention
best_mean, kind, frac, distance, sigma, rr_on = results[0][:6]
rr_ref = RR_REF if rr_on else None
print(f"\nTEST split (30 patients), tuned detector {label(kind)} frac={frac} distance={distance} "
      f"sigma={sigma} RR-scaled={rr_on}")
runs = {}
for name, cfg in [("your rule 60 % of max", ("thresh", 0.6, 250, 0, None)),
                  (f"tuned {label(kind)}", (kind, frac, distance, sigma, rr_ref)),
                  ("ceiling: true R peaks + same windows", None)]:
    if cfg is None:
        peaks = [np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3]) for r in splits["test"]]
        preds = np.array([paint(p, offsets, rr_ref) for p in peaks])
    else:
        _, preds = score(splits["test"], *cfg[:4], offsets, cfg[4])
    runs[name] = under(splits["test"], preds)
for c in CONVENTIONS:
    print(f"\n  judged as '{c}'" + ("  <- the convention we report" if c == PRIMARY else ""))
    for name in runs:
        print(f"    {name:36s} {fmt(runs[name][c])}")
print("  reference in the notebook (as parsed, unmasked): P 0.74 / QRS 0.89 / T 0.85")


def r_stats(idx, kind, frac, distance, sigma):
    tp = fp = fn = 0
    for r, found in zip(idx, detect(x[idx], kind, frac, distance, sigma)):
        free = list(found)
        for q in np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3]):
            if free and min(abs(q - f) for f in free) <= 25:
                free.pop(int(np.argmin([abs(q - f) for f in free])))
                tp += 1
            else:
                fn += 1
        fp += len(free)
    return tp, fp, fn


for desc, cfg in [("your rule", ("thresh", 0.6, 250, 0)), (f"chosen ({label(kind)})", (kind, frac, distance, sigma))]:
    tp, fp, fn = r_stats(splits["test"], *cfg)
    print(f"  R peaks, {desc:22s}: {tp + fn} annotated, {tp + fp} detected, sensitivity {tp / (tp + fn):.3f}, "
          f"precision {tp / max(tp + fp, 1):.3f} ({fn} missed, {fp} false alarms)")

# --- figure: one test record, annotated against baseline, with the ignored stretch in grey
r = int(splits["test"][0])
found = detect(x[r:r + 1], kind, frac, distance, sigma)[0]
pred = paint(found, offsets, rr_ref)
t = np.arange(T) / FS
fig, axes = plt.subplots(3, 1, figsize=(13, 5.2), sharex=True)
axes[0].plot(t, x[r], "k", linewidth=0.7)
axes[0].set_ylabel("lead II")
for ax, lab, arr in ((axes[1], "annotated", y_raw[r]), (axes[2], "baseline", pred)):
    ax.fill_between(t, -4, 4, where=mask[r] == -1, color="0.75", linewidth=0)
    ax.fill_between(t, -4, 4, where=(CONVENTIONS["span, no buffer"][r] == -1) & (mask[r] >= 0),
                    color="0.88", linewidth=0)
    ax.plot(t, x[r], "k", linewidth=0.4, alpha=0.6)
    for c in (1, 2, 3):
        ax.fill_between(t, -4, 4, where=arr == c, color=["none", "tab:green", "tab:red", "tab:blue"][c], alpha=0.45)
    ax.set_ylim(-4, 4)
    ax.set_ylabel(lab)
axes[2].plot(found / FS, x[r][found], "kv", markersize=4)
axes[2].set_xlabel("time (s)")
fig.suptitle(f"test record {r}: the cardiologists' waves (middle) against the classical baseline (bottom); "
             f"green P, red QRS, blue T; dark grey = ignored by loss and metric, light grey = the buffer, judged",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "baseline_one_test_record.png", dpi=130)
print("\nwrote", FIGS / "baseline_one_test_record.png")
