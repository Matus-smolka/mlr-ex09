"""Validity inspection of the parsed LUDB data (Task 1: ECG delineation).

Reads the cache written by parse_ludb.py plus the per-record diagnostics, and reports on
structure, signal quality, annotation completeness, label collisions, wave geometry,
class balance and the train/valid/test split.

Usage:  py -I task1_ludb/inspect_ludb.py <cache.npz> <ludb_records.csv>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FS = 500
WAVES = ["background", "P", "QRS", "T"]


def head(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def q(a, pcts=(0, 5, 50, 95, 100)):
    a = np.asarray(a, dtype=float)
    return f"n={len(a)} " + " ".join(f"p{p}={np.percentile(a, p):.3g}" for p in pcts) if len(a) else "empty"


def wave_rows(record, cls=None):
    """The annotated waves of one record, optionally of one class, sorted by onset."""
    keep = record_of_wave == record
    if cls is not None:
        keep &= waves[:, 1] == cls
    return waves[keep][np.argsort(waves[keep, 2])]


def peaks(record, cls):
    return np.sort(wave_rows(record, cls)[:, 3])


d = np.load(sys.argv[1], allow_pickle=True)
rec_info = pd.read_csv(sys.argv[2], encoding="utf-8")
x, y, waves, rhythm = d["x"], d["y"], d["waves"], list(d["rhythm"])
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
record_of_wave = waves[:, 0]
n, T = y.shape
n_waves_rec = np.array([len(waves[record_of_wave == r]) for r in range(n)])

head("1. STRUCTURE")
print(f"records {x.shape} signals, {y.shape} labels | float32 signals, int64 labels")
print("samples per record:", rec_info["n_samples"].unique(), "| sampling rate:", rec_info["fs"].unique(), "Hz")
print(f"label values present: {sorted(np.unique(y))} = {WAVES}")
print(f"NaN in the raw signals: {rec_info['nan'].sum()}, records with NaN: {(rec_info['nan'] > 0).sum()}")
print(f"standardized signals: min {x.min():.2f}, max {x.max():.2f}, non-finite values: {(~np.isfinite(x)).sum()}")
print("per-record mean after standardization:", q(x.mean(1), (0, 50, 100)))
print("per-record std  after standardization:", q(x.std(1), (0, 50, 100)))

head("2. RAW SIGNAL QUALITY (lead II, in mV, before standardization)")
print("std   :", q(rec_info["raw_std"], (0, 1, 50, 99, 100)))
print("range :", q(rec_info["raw_max"] - rec_info["raw_min"], (0, 50, 100)))
print("|median|:", q(np.abs(rec_info["raw_median"]), (0, 50, 100)))
print("three lowest-amplitude records:\n",
      rec_info.nsmallest(3, "raw_std")[["record", "raw_std", "raw_max", "raw_min"]].to_string(index=False))
finger = np.round(x[:, ::50].sum(1), 3)
print(f"records whose coarse fingerprint equals another record's: {pd.Series(finger).duplicated(keep=False).sum()}")

head("3. ANNOTATION COMPLETENESS")
total_ann = int(rec_info["n_annotations"].sum())
unused = Counter(s for row in rec_info["unused_symbols"] for s in str(row).split(",") if s and s != "nan")
matched = total_ann - sum(unused.values())
print(f"annotation marks in lead II: {total_ann} | used in a ( sym ) triplet: {matched} ({matched / total_ann:.1%})"
      f" | ignored: {sum(unused.values())}")
print("ignored symbols:", dict(unused.most_common(10)) or "none")
print(f"waves per record: {q(n_waves_rec)} | records with no wave at all: {(n_waves_rec == 0).sum()}")

head("4. WAVE GEOMETRY (the cardiologists' triplets)")
rows = []
for c in (1, 2, 3):
    w = waves[waves[:, 1] == c]
    width = w[:, 4] - w[:, 2]
    rows.append({"class": WAVES[c], "waves": len(w),
                 "width ms p5/p50/p95": f"{np.percentile(width, 5) * 2:.0f} / {np.median(width) * 2:.0f} / "
                                        f"{np.percentile(width, 95) * 2:.0f}",
                 "max width ms": width.max() * 2, "peak inside [onset,offset]":
                 f"{np.mean((w[:, 3] >= w[:, 2]) & (w[:, 3] <= w[:, 4])):.3f}",
                 "non-positive width": int((width <= 0).sum())})
print(pd.DataFrame(rows).to_string(index=False))
mono = sum(bool(np.all(np.diff(waves[record_of_wave == r][:, 2]) >= 0)) for r in range(n) if n_waves_rec[r] > 1)
n_multi = sum(n_waves_rec[r] > 1 for r in range(n))
print(f"records whose waves are listed in temporal order: {mono}/{n_multi}")
print(f"waves starting before the record: {int((waves[:, 2] < 0).sum())} | ending after sample {T - 1}: "
      f"{int((waves[:, 4] > T - 1).sum())}")

head("5. BEAT STRUCTURE")
rr = np.concatenate([np.diff(peaks(r, 2)) for r in range(n) if len(peaks(r, 2)) > 1])
print(f"R-R intervals: {q(rr / FS * 60, (0, 5, 50, 95, 100))} heart rate (bpm)")
print("QRS per record:", q([len(peaks(r, 2)) for r in range(n)]))
paired = [len(peaks(r, 2)) and sum(1 for onset in peaks(r, 2) if ((peaks(r, 1) < onset) & (peaks(r, 1) > onset - 600)).any())
          for r in range(n)]
n_qrs = int(sum(len(peaks(r, 2)) for r in range(n)))
n_paired = int(sum(paired))
print(f"QRS preceded by a P wave within 600 samples (1.2 s): {n_paired}/{n_qrs} = {n_paired / n_qrs:.3f}"
      f" | records where every QRS has one: {sum(p == len(peaks(r, 2)) for p, r in zip(paired, range(n)) if len(peaks(r, 2)))}/{n}")

head("6. LABEL / ANNOTATION AGREEMENT (does y really encode the waves?)")
overwritten, lost_records = Counter(), []
for r in range(n):
    lost = 0
    for _, c, onset, _, offset in wave_rows(r):
        seg = y[r, max(onset, 0):min(offset, T - 1) + 1]
        bad = int((seg != c).sum())
        overwritten[WAVES[c]] += bad
        lost += bad
    if lost:
        lost_records.append((r, lost))
total_ann_samples = int(sum(min(o, T - 1) - max(s, 0) + 1 for _, _, s, _, o in waves))
lost_total = sum(overwritten.values())
print(f"annotated samples overwritten by a later wave: {lost_total} of {total_ann_samples} "
      f"({lost_total / total_ann_samples:.2%}) -> {dict(overwritten)}")
lost_records.sort(key=lambda t: -t[1])
print("records most affected (record, samples lost):", lost_records[:6])
runs = np.zeros((n, 3), dtype=int)
for r in range(n):
    prev = np.concatenate(([-1], y[r, :-1]))
    for j, c in enumerate((1, 2, 3)):
        runs[r, j] = int(((y[r] == c) & (prev != c)).sum())
for j, c in enumerate((1, 2, 3)):
    print(f"{WAVES[c]:4s}: {int((waves[:, 1] == c).sum())} annotated waves -> {runs[:, j].sum()} contiguous blocks "
          f"of that class in y ({runs[:, j].sum() - int((waves[:, 1] == c).sum()):+d} fragments)")

head("7. CLASS BALANCE, MISSING CLASSES, SPLITS")
print("overall sample fractions:", dict(zip(WAVES, np.round(np.bincount(y.ravel(), minlength=4) / y.size, 3))))
for c in (1, 2, 3):
    empty = [r for r in range(n) if len(peaks(r, c)) == 0]
    print(f"records with no {WAVES[c]:4s} wave: {len(empty):3d} -> rhythms: "
          f"{Counter(rhythm[r] for r in empty).most_common(3)}")
for s, idx in splits.items():
    f = np.bincount(y[idx].ravel(), minlength=4) / y[idx].size
    hr = [np.median(np.diff(peaks(r, 2))) / FS * 60 for r in idx if len(peaks(r, 2)) > 1]
    print(f"{s:5s} {len(idx):3d} records, {int(n_waves_rec[idx].sum()):4d} waves, fractions {np.round(f, 3)}, "
          f"median heart rate {np.median(hr):.0f} bpm, rhythms {dict(Counter(rhythm[r] for r in idx).most_common(3))}")

head("8. RHYTHMS (the only per-patient metadata)")
print("distinct first-listed rhythms:", len(set(rhythm)))
by_rhythm = {}
for r in range(n):
    by_rhythm.setdefault(rhythm[r], []).append(r)
for name, rs in sorted(by_rhythm.items(), key=lambda kv: -len(kv[1])):
    where = "".join(s[0].upper() if any(r in splits[s] for r in rs) else "." for s in ("train", "valid", "test"))
    print(f"  {len(rs):3d}  {name[:50]:50s} [{where}]  (T=v a=l i=n)")

head("9. SPLIT INTEGRITY")
union = set(splits["train"]) | set(splits["valid"]) | set(splits["test"])
print("sizes:", {s: len(v) for s, v in splits.items()},
      "| pairwise disjoint:", not (set(splits["train"]) & set(splits["valid"]) or set(splits["train"]) & set(splits["test"])
                                   or set(splits["valid"]) & set(splits["test"])),
      "| covers every record:", sorted(union) == list(range(n)))
print("per-class test fractions:", dict(zip(WAVES, np.round(np.bincount(y[splits['test']].ravel(), minlength=4) / y[splits['test']].size, 3))))
