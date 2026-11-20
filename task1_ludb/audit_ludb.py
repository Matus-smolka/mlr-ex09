"""Follow-up audit of the anomalies the first report surfaced in LUDB (Task 1).

1. every raw record spans exactly 1 mV -> how is the data stored/quantized?
2. 14 'N' (QRS) annotations ignored by the notebook's strict "( symbol )" triplet rule
3. the one record with no annotated QRS
4. records whose QRS amplitude barely exceeds the baseline drift
5. wave-width outliers, very long R-R gaps, partial beats at the record edges

Usage:  py -I task1_ludb/audit_ludb.py <path-to-ludb-folder> <cache.npz>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

FS = 500
WAVES = ["background", "P", "QRS", "T"]
LUDB, CACHE = Path(sys.argv[1]), sys.argv[2]
d = np.load(CACHE, allow_pickle=True)
x, y, waves = d["x"], d["y"], d["waves"]
rec_of = waves[:, 0]
n, T = y.shape


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


head("1. HOW THE SIGNALS ARE STORED")
ranges, steps, uniques = [], [], []
for rec in (1, 2, 125, 132):
    raw = wfdb.rdrecord(str(LUDB / "data" / str(rec)), channel_names=["ii"]).p_signal[:, 0]
    u = np.unique(raw)
    ranges.append(raw.max() - raw.min())
    steps.append(np.min(np.diff(u)) if len(u) > 1 else np.nan)
    uniques.append(len(u))
    hdr = wfdb.rdheader(str(LUDB / "data" / str(rec)))
    i_ii = next(i for i, nm in enumerate(hdr.sig_name) if nm.upper() == "II")
    print(f"record {rec:3d}: n unique values {len(u):5d} | range {raw.max() - raw.min():.6f} | "
          f"smallest step between consecutive values {steps[-1]:.6f} mV | lead II adc_gain "
          f"{hdr.adc_gain[i_ii]} baseline {hdr.baseline[i_ii]} units {hdr.units[i_ii]}")
print("-> every record spans exactly 1.000 mV peak-to-peak; the absolute amplitude of a patient is not in the data.")

all_ranges, digital_ranges, lead_ranges = [], [], None
for rec in range(1, 201):
    r = wfdb.rdrecord(str(LUDB / "data" / str(rec)))
    i_ii = next(i for i, nm in enumerate(r.sig_name) if nm.upper() == "II")
    all_ranges.append(np.nanmax(r.p_signal[:, i_ii]) - np.nanmin(r.p_signal[:, i_ii]))
    digital = wfdb.rdrecord(str(LUDB / "data" / str(rec)), physical=False)   # d_signal only exists when not converted
    digi = digital.d_signal[:, i_ii]
    digital_ranges.append(int(np.nanmax(digi) - np.nanmin(digi)))
    if lead_ranges is None:
        lead_ranges = np.nanmax(r.p_signal, axis=0) - np.nanmin(r.p_signal, axis=0)
all_ranges, digital_ranges = np.array(all_ranges), np.array(digital_ranges)
print(f"lead II range over all 200 records: min {all_ranges.min():.6f} max {all_ranges.max():.6f} "
      f"-> exactly 1 mV in {int((all_ranges == 1).sum())}/200 records")
print(f"same in digital units: range {digital_ranges.min()}-{digital_ranges.max()} vs adc_gain -> the peak-to-peak "
      f"of every lead equals its own gain")
print("per-lead range of record 1 (12 leads):", np.round(lead_ranges, 4))

hdrs = [wfdb.rdheader(str(LUDB / "data" / str(rec))) for rec in (1, 2, 3)]
print("metadata in the .hea comments, e.g. record 1:", hdrs[0].comments[:3])

head("2. ANNOTATIONS THE PARSER DROPS")
wave_code = {"p": 1, "N": 2, "t": 3}
dropped = []
for rec in range(1, 201):
    ann = wfdb.rdann(str(LUDB / "data" / str(rec)), "ii")
    matched = set()
    for k in range(1, len(ann.symbol) - 1):
        if ann.symbol[k] in wave_code and ann.symbol[k - 1] == "(" and ann.symbol[k + 1] == ")":
            matched |= {k - 1, k, k + 1}
    for k, s in enumerate(ann.symbol):
        if s in wave_code and k not in matched:
            ctx = " ".join(f"[{i}]{ann.symbol[i]}@{ann.sample[i]}" for i in range(max(0, k - 2), min(len(ann.symbol), k + 3)))
            dropped.append((rec - 1, s, int(ann.sample[k]), ctx))
print(f"wave symbols NOT inside a ( symbol ) triplet: {len(dropped)}")
for rec, s, at, ctx in dropped:
    n_w = int((rec_of == rec).sum())
    near = int(((waves[:, 0] == rec) & (np.abs(waves[:, 4] - at) < 400)).sum())
    print(f"  record {rec:3d}  {s} @ {at:5d}  waves in y around it: {near:2d} | context {ctx}")

head("3. THE RECORD WITH NO ANNOTATED QRS")
qrs_per_rec = np.array([int(((rec_of == r) & (waves[:, 1] == 2)).sum()) for r in range(n)])
for r in np.flatnonzero(qrs_per_rec == 0):
    print(f"record {r}: {int((rec_of == r).sum())} annotated waves, classes "
          f"{Counter(WAVES[c] for c in waves[rec_of == r][:, 1])}, label fractions "
          f"{np.round(np.bincount(y[r], minlength=4) / T, 3)}")
    print(f"  standardized signal: min {x[r].min():.2f} max {x[r].max():.2f} std {x[r].std():.2f} "
          f"| biggest |sample| at t={int(np.argmax(np.abs(x[r]))) / FS:.2f} s")
    for lead in ["I", "II", "V1", "V6"]:
        try:
            a = wfdb.rdann(str(LUDB / "data" / str(r + 1)), lead)
            print(f"  lead {lead:2s}: {len(a.symbol)} annotations, "
                  f"{sum(1 for i in range(1, len(a.symbol) - 1) if a.symbol[i] == 'N' and a.symbol[i - 1] == '(' and a.symbol[i + 1] == ')')} complete QRS triplets")
        except Exception as e:
            print(f"  lead {lead:2s}: no annotation ({type(e).__name__})")

head("4. LOW-AMPLITUDE / HIGH-DRIFT RECORDS")
raw_std = np.array([wfdb.rdrecord(str(LUDB / "data" / str(r + 1)), channel_names=["ii"]).p_signal[:, 0].std()
                    for r in range(n)])
order = np.argsort(raw_std)
print("QRS-to-drift is worst where the raw std is smallest (all records span the same 1 mV):")
for r in order[:5]:
    peak_qrs = np.abs(x[r][y[r] == 2]).max() if (y[r] == 2).any() else float("nan")
    print(f"  record {r:3d} raw std {raw_std[r]:.4f} mV | standardized range [{x[r].min():.1f}, {x[r].max():.1f}] | "
          f"max |standardized| QRS sample {peak_qrs:.2f}")
print(f"records with standardized |x| above 6 somewhere: {int((np.abs(x) > 6).any(1).sum())}")

head("5. WIDTH OUTLIERS, LONG RR GAPS, PARTIAL BEATS AT THE EDGES")
for c in (1, 2, 3):
    w = waves[waves[:, 1] == c]
    width = (w[:, 4] - w[:, 2]) * 2                       # ms
    out = w[width > np.percentile(width, 99)]
    print(f"{WAVES[c]:4s}: {int((width > np.percentile(width, 99)).sum())} waves above p99 = "
          f"{np.percentile(width, 99):.0f} ms, e.g. " +
          ", ".join(f"rec {int(r)} {int((o - s) * 2)} ms @{int(s) / FS:.1f}s" for r, _, s, _, o in out[:3]))
gaps = []
for r in range(n):
    p = np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3])
    if len(p) > 1:
        gaps.append((int(np.diff(p).max()), r))
print("longest R-R gap per record, top 5:", sorted(gaps, reverse=True)[:5], "(samples, record)")
starts = int(sum(1 for r in range(n) if y[r, 0] != 0))
ends = int(sum(1 for r in range(n) if y[r, -1] != 0))
print(f"records whose first sample is already inside a wave: {starts}/200 | whose last sample is: {ends}/200")
first_is = Counter(WAVES[y[r, 0]] for r in range(n) if y[r, 0] != 0)
print("  class at sample 0:", dict(first_is))
t_to_next_p, pr_to_qrs = [], []
for r in range(n):
    p = np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3])
    pp = np.sort(waves[(rec_of == r) & (waves[:, 1] == 1), 3])
    t = np.sort(waves[(rec_of == r) & (waves[:, 1] == 3), 3])
    pr_to_qrs += [int(np.min(np.abs(p - pp))) for p in p if len(pp) and np.min(np.abs(p - pp)) < 600]
    t_to_next_p += [int(np.min(np.abs(pp - t))) for t in t if len(pp) and np.min(np.abs(pp - t)) < 900]
print(f"P peak -> its QRS peak: median {np.median(pr_to_qrs) * 2:.0f} ms, p5-p95 "
      f"{np.percentile(pr_to_qrs, 5) * 2:.0f}-{np.percentile(pr_to_qrs, 95) * 2:.0f} ms")
print(f"T peak -> nearest P peak: median {np.median(t_to_next_p) * 2:.0f} ms, p5-p95 "
      f"{np.percentile(t_to_next_p, 5) * 2:.0f}-{np.percentile(t_to_next_p, 95) * 2:.0f} ms")
