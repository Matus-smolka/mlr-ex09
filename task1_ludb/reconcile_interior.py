"""Why audit_unlabelled.py and repair_ludb.py counted the unlabelled interior beats differently.

audit_unlabelled.py calls a QRS-like peak "explained" when it falls INSIDE an annotated wave.
repair_ludb.py is more generous: also when it sits within +-50 samples of an annotated QRS peak.
They also ran at different detector sensitivities (0.3 / 0.5 against 0.4). This runs both predicates
at every threshold, so the number that goes in the README is one number and not two.

Usage:  py -I task1_ludb/reconcile_interior.py <cache.npz> [masked_cache.npz]

The second argument is what lets the last block answer the question it claims to answer: whether the
repair left any unlabelled beat asserted to be background.
"""
import sys
from collections import Counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks

FS = 500
d = np.load(sys.argv[1], allow_pickle=True)
x, y, waves, splits = d["x"], d.get("y_raw", d["y"]), d["waves"], {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of = waves[:, 0]
n, T = y.shape
B, A = butter(3, [8 / (FS / 2), 20 / (FS / 2)], btype="band")


def qrs_peaks(s):
    env = gaussian_filter1d(np.abs(filtfilt(B, A, s)), 2.0)
    return find_peaks(env, height=HT * env.max(), distance=int(0.35 * FS))[0]


print(f"{'det':>4s}  {'predicate counting as explained':34s}  interior  records (split)")
for HT in (0.3, 0.4, 0.5):
    peaks = [qrs_peaks(x[r]) for r in range(n)]
    inside = lambda p, w: any(o <= p <= f for _, _, o, _, f in w)
    for name, test in [("inside an annotated wave", lambda p, q, w: inside(p, w)),
                       ("...or within 50 of a QRS peak", lambda p, q, w: inside(p, w)
                                                                        or (len(q) and np.min(np.abs(q - p)) <= 50))]:
        hits = []
        for r in range(n):
            w = waves[rec_of == r]
            q = np.sort(w[w[:, 1] == 2][:, 3])
            lo, hi = w[:, 2].min(), w[:, 4].max()
            for p in peaks[r]:
                if lo <= p <= hi and not test(p, q, w):
                    hits.append(r)
        c = Counter(hits)
        where = ", ".join(f"{r} ({next(s for s in splits if r in splits[s])})" for r in sorted(c))
        print(f"{HT:4.1f}  {name:34s}  {len(hits):8d}  {dict(sorted(c.items()))}  {where}")

# The honest question: after repair_ludb.py ran, is any QRS-like peak still asserted to be background?
# That can only be asked of the REPAIRED labels, so it needs the second argument: a peak sitting on a
# sample the repair turned into -1 is not asserted to be anything. Asked of the unrepaired cache the
# same loop answers a different question - how many unlabelled beats there are in total - which is
# what made this block misleading when it was run on task1_ludb/cache/ludb.npz alone.
if len(sys.argv) > 2:
    repaired = np.load(sys.argv[2], allow_pickle=True)
    ym, BUF = repaired["y"], int(repaired["buffer"])
    assert ym.shape == y.shape, (ym.shape, y.shape)
    print("\nstill asserted to be BACKGROUND after the repair (the repaired label at the peak is 0 and "
          "no annotation comes within +-50 samples):")
else:
    ym, BUF = d["y"], None
    print("\nNOTE: no repaired cache given, so the block below counts the unlabelled beats AS PARSED - "
          "not what the repair left behind. Pass task1_ludb/cache/ludb_masked.npz as a second argument.")
# the span the repair masked against: first annotated onset to last annotated offset, every class
span = np.array([[waves[rec_of == r][:, 2].min(), waves[rec_of == r][:, 4].max()] if (rec_of == r).any()
                 else (0, T - 1) for r in range(n)])
for HT in (0.3, 0.4, 0.5):
    bad = []
    for r in range(n):
        w = waves[rec_of == r]
        for p in qrs_peaks(x[r]):
            if ym[r, p] == 0 and not ((w[:, 2] - 50 <= p) & (p <= w[:, 4] + 50)).any():
                bad.append((r, int(p)))
    c = Counter(r for r, _ in bad)
    inside = sum(span[r, 0] <= p <= span[r, 1] for r, p in bad)
    where = (f", {inside} of them inside the annotated span (repair_ludb.py promises 0) and "
             f"{len(bad) - inside} in the {BUF}-sample buffer or beyond" if BUF is not None else "")
    print(f"  detector {HT}: {len(bad)} peaks in {len(c)} records{where}")
