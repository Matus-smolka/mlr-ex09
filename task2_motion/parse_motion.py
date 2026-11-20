"""Parse UCI HAPT exactly as the notebook's data cell does, and cache it.

The notebook reads every `acc_expNN_userNN.txt` as float32 (T, 3) in g, writes -1 over a whole
recording and then paints the label rows of `labels.txt` onto it in file order -

    for _, _, activity, first, last in labels_txt[labels_txt[:, 0] == exp]:
        lab[first - 1:last] = activity - 1 if activity <= 6 else 6

- so activities 1-6 become classes 0-5, the six postural transitions 7-12 all become class 6
 (`transition`), and the samples between the labelled activities keep -1. Two details of that loop are
 measured rather than assumed, because both change the target:

   * it selects the label rows by **experiment only**, not by user, so if an experiment ever appeared
     for two users the rows would be painted onto both;
   * the rows are painted in file order, so where two intervals overlap the later row wins.

Everything is written to one flat array with per-recording offsets, because the 61 recordings have
different lengths.

Usage:  py -I task2_motion/parse_motion.py <hapt-dir> <out.npz> [<records.csv>]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CSV = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT.with_suffix(".csv")

files = sorted(SRC.glob("acc_exp*_user*.txt"))
if len(files) != 61:
    sys.exit(f"expected 61 acc_exp*_user*.txt files in {SRC}, found {len(files)}")
labels_txt = np.loadtxt(SRC / "labels.txt", dtype=int)     # experiment, user, activity, first, last (from 1)

acc, lab, rows, offsets, off = [], [], [], [0], 0
for f in files:
    exp, user = int(f.stem[7:9]), int(f.stem[-2:])         # the notebook's two slices, unchanged
    a = np.loadtxt(f, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        sys.exit(f"{f.name}: expected (T, 3), got {a.shape}")
    y = np.full(len(a), -1, dtype=np.int8)
    sel = labels_txt[labels_txt[:, 0] == exp]
    painted, overwritten, outside = 0, 0, 0
    taken = np.zeros(len(a), bool)
    for _, u_row, activity, first, last in sel:
        lo, hi = int(first) - 1, int(last)                 # the notebook's slice, 1-based inclusive
        outside += max(0, hi - len(a))
        new = np.zeros(len(a), bool)
        new[lo:hi] = True
        overwritten += int((new & taken).sum())
        y[lo:hi] = int(activity) - 1 if activity <= 6 else 6
        taken |= new
        painted += int(new.sum())
        if int(u_row) != user:
            print(f"  NOTE labels row user {int(u_row)} painted onto recording user {user} (exp {exp})")
    acc.append(a)
    lab.append(y)
    offsets.append(off + len(a))
    off += len(a)
    rows.append({
        "file": f.name, "experiment": exp, "user": user, "samples": len(a), "seconds": len(a) / 50.0,
        "label_rows": len(sel), "painted": painted, "overwritten_samples": overwritten,
        "painted_past_end": outside,
        "labelled": int((y >= 0).sum()), "unlabelled": int((y < 0).sum()),
        "n_classes": int(len(set(y[y >= 0].tolist()))),
        "x_mean": float(a[:, 0].mean()), "y_mean": float(a[:, 1].mean()), "z_mean": float(a[:, 2].mean()),
        "x_std": float(a[:, 0].std()), "y_std": float(a[:, 1].std()), "z_std": float(a[:, 2].std()),
        "abs_min": float(np.abs(a).min()), "abs_max": float(np.abs(a).max()),
        "finite": int(np.isfinite(a).all(axis=1).sum()),
    })

acc = np.concatenate(acc)
lab = np.concatenate(lab)
offsets = np.array(offsets)
users = {"train": range(1, 21), "valid": range(21, 26), "test": range(26, 31)}      # the notebook's
exp_of = np.array([r["experiment"] for r in rows])
user_of = np.array([r["user"] for r in rows])
splits = {s: np.flatnonzero(np.isin(user_of, list(u))) for s, u in users.items()}
assert sum(len(v) for v in splits.values()) == 61, "the three user ranges must cover all 61 recordings"

frame = pd.DataFrame(rows)
frame.to_csv(CSV, index=False)
np.savez_compressed(OUT, acc=acc, lab=lab, offsets=offsets, exp=exp_of, user=user_of,
                    labels_txt=labels_txt,
                    **{f"split_{s}": v for s, v in splits.items()})

print(f"acc {acc.shape} {acc.dtype} in [{acc.min():.3f}, {acc.max():.3f}] | lab {lab.shape} "
      f"| {offsets[-1]} samples = {offsets[-1] / 50 / 60:.1f} min in 61 recordings")
print(f"labels {int((lab >= 0).sum())} labelled ({(lab >= 0).mean():.1%}), "
      f"{int((lab < 0).sum())} unlabelled (-1) | per class {np.bincount(lab[lab >= 0], minlength=7)}")
for s, idx in splits.items():
    y = np.concatenate([lab[offsets[i]:offsets[i + 1]] for i in idx])
    print(f"  {s:5s} {len(idx)} recordings, {len(y)} samples, users {sorted(set(user_of[idx]))}, "
          f"per class {np.bincount(y[y >= 0], minlength=7)}")
print(f"sanity: the notebook's own print, labels -1,0-6 -> {np.bincount(lab + 1)}")
print(f"cache -> {OUT} | per-recording diagnostics -> {CSV}")
