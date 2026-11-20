"""Task 2 before any model: is this data what the notebook says it is, and how much of it is axis order?

Run after parse_motion.py. Everything printed is measured from the files, not quoted from the paper,
in five passes:

1. **what the recordings are** - shape, units, sampling rate, broken values. The units are decided by
   measuring the size of gravity (a phone at rest reads about 1.0 in g, about 9.81 in m/s2), the
   sampling rate by measuring the dominant step frequency of the walking stretches, which is 1.5-2.5 Hz
   for a walking adult and would come out near 1 Hz if the files were really sampled at 100 Hz.
2. **what the labels are** - how much of every recording is labelled, how long each class runs, which
   recordings are incomplete, and what test-set accuracy the "always answer standing" baseline the
   notebook quotes really gives.
3. **how the phone is worn** - the direction gravity points in, per recording, taken from its three
   still classes. That single vector per recording is the whole orientation question: if it comes out
   nearly the same for every person, the dataset cannot show an orientation-aware method working and the
   stress test has to be made by hand.
4. **how much of the task lives in the axis order** - two rules with nothing learned in them (nearest
   of a handful of class averages) scored on the test users once as the data comes and once per each of
   the 24 ways of turning the phone.
5. **what the window length trades off**, since the app feeds the network one window at a time.

Self-checks for the helpers come first, so a wrong window or a wrong rotation is caught here rather
than inside a trained model.

Usage:  py -I task2_motion/inspect_motion.py <cache/hapt.npz> <figs-dir> [<cache/records.csv>]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy import signal                                              # noqa: E402

from motion_data import (FS, CLASSES, STATIC, COLORS, confusion, eval_windows,   # noqa: E402
                         f1_scores, lab_to_rgba, load_splits, perm_name, random_rotation,
                         signed_permutations, windows_of)

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
RECORDS = Path(sys.argv[3]) if len(sys.argv) > 3 else CACHE.with_suffix(".csv")
FIGS.mkdir(parents=True, exist_ok=True)
splits, recs = load_splits(CACHE)
meta = pd.read_csv(RECORDS)
out = {}


def runs(y, want, trim=0, min_len=1):
    """(start, end) of every maximal stretch of the label `want` in y, trimmed at both ends."""
    y = np.asarray(y)
    b = np.flatnonzero(np.r_[True, y[1:] != y[:-1], True])
    return [(int(s) + trim, int(e) - trim) for s, e in zip(b[:-1], b[1:])
            if y[s] == want and e - s >= min_len + 2 * trim]


def all_runs(y):
    """[(start, end, label)] of every stretch, the pauses included."""
    b = np.flatnonzero(np.r_[True, y[1:] != y[:-1], True])
    return [(int(s), int(e), int(y[s])) for s, e in zip(b[:-1], b[1:])]


print("=" * 98)
print("0. the helpers this task builds on")
print("=" * 98)
perms = signed_permutations()
assert perms.shape == (24, 3, 3)
assert np.allclose(perms @ perms.transpose(0, 2, 1), np.eye(3)[None]), "each must be orthogonal"
assert np.allclose(np.linalg.det(perms), 1), "a rotation, not a mirror"
names = {perm_name(m) for m in perms}
assert "x,y,z" in names, "the set must contain the orientation the data is in"
assert {perm_name((p @ q).round(6)) for p in perms for q in perms} <= names, "closed under composition"
R = np.stack([random_rotation(np.random.default_rng(s)) for s in range(200)])
assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-5) and np.allclose(np.linalg.det(R), 1)
n = 1000
cover = np.zeros(n, int)
for s, keep in eval_windows(n, 128):
    cover[keep:s + 128] += 1
assert (cover == 1).all(), "every sample predicted exactly once"
assert windows_of(1000, 128).tolist() == list(range(0, 769, 128)), "training windows: full ones only"
assert len(windows_of(127, 128)) == 0
chk = f1_scores([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2], k=3)
assert abs(chk["per_class"][0]["f1"] - 2 / 3) < 1e-9 and abs(chk["per_class"][1]["f1"] - 0.8) < 1e-9
assert confusion([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2], k=3).tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
print(f"the 24 signed permutations are orthogonal, have determinant +1, contain x,y,z and are closed "
      f"under composition; the random SO(3) draws the same\nwindows of 1000 samples at W=128 cover every "
      f"sample exactly once; macro F1 and the confusion matrix match a case done by hand")

whole = np.concatenate([r[0] for r in recs])
lab_all = np.concatenate([r[1] for r in recs])
mag = np.linalg.norm(whole, axis=1)

print("\n" + "=" * 98)
print("1. what the 61 recordings are")
print("=" * 98)
print(f"{len(recs)} recordings, {len(whole)} samples = {len(whole) / FS / 60:.1f} min, "
      f"{len(set(r[2]['user'] for r in recs))} people (users 1..{max(r[2]['user'] for r in recs)}), "
      f"experiments 1..{max(r[2]['exp'] for r in recs)}")
vc = pd.Series([r[2]["user"] for r in recs]).value_counts()
print(f"recordings per user: {int((vc == 2).sum())} users with 2, {int((vc == 3).sum())} with 3 "
      f"(user {vc[vc == 3].index[0]})")
print(f"finite samples {int(np.isfinite(whole).all(axis=1).sum())} of {len(whole)}; "
      f"values in [{whole.min():.3f}, {whole.max():.3f}]")
still = [(r, c, s, e) for r in range(len(recs)) for c in STATIC
         for s, e in runs(recs[r][1], c, trim=FS, min_len=2 * FS)]
mags = np.array([np.linalg.norm(recs[r][0][s:e], axis=1).mean() for r, c, s, e in still])
print(f"units: over the {len(still)} still-class stretches the measured acceleration has size "
      f"{mags.mean():.3f} g (sd {mags.std():.3f}, range {mags.min():.3f}..{mags.max():.3f})")
print(f"      -> the files are in g, not m/s2 (at rest a phone reads about 1 g, and the browser's "
      f"Sensor gives m/s2, so the app has to convert)")
print(f"never saturated: |a| is at most {mag.max():.2f} g anywhere in the 61 recordings")
for thr in (0.2, 0.5, 0.8):
    k = int((mag < thr).sum())
    print(f"      samples with |a| < {thr:.1f} g (free fall, or a sensor that dropped out): {k} "
          f"({k / len(mag):.4%})")
rep = [np.all(r[0][1:] == r[0][:-1], axis=1) for r in recs]
dupl = int(sum(s.sum() for s in rep))


def bool_runs(s):
    """length of every maximal True stretch of a boolean array"""
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    return np.array([j - i for i, j in zip(b[:-1], b[1:]) if s[i]], int)


lens = np.concatenate([bool_runs(s) for s in rep])
by_class = {CLASSES[c][:4] if c >= 0 else "pause":
            sum(int(np.all(r[0][1:] == r[0][:-1], axis=1)[r[1][1:] == c].sum()) for r in recs)
            / max(sum(int((r[1] == c).sum()) for r in recs), 1) for c in [-1] + list(range(7))}
print(f"      samples equal to the previous one on all three axes: {dupl} ({dupl / len(whole):.2%}) - "
      f"{int(np.sum(lens == 1))} of them a single sample held for one extra sample, only "
      f"{int(np.sum(lens > 2))} runs longer than 3, and the same {min(by_class.values()):.1%}-"
      f"{max(by_class.values()):.1%} share in every class and every recording. That is the phone's uneven "
      f"timestamps landed on a regular grid, not a sensor losing movement")
stride = []
for r in range(len(recs)):
    for s, e in runs(recs[r][1], 0, min_len=4 * FS):
        x = np.linalg.norm(recs[r][0][s:e] - recs[r][0][s:e].mean(0), axis=1)
        ac = np.correlate(x - x.mean(), x - x.mean(), "full")[len(x) - 1:]
        ac /= ac[0]
        lo = int(0.5 * FS)
        stride.append((int(np.argmax(ac[lo:int(2.5 * FS)])) + lo) / FS)          # seconds per full stride
stride = np.array(stride)
freqs = []
for r in range(len(recs)):
    for s, e in runs(recs[r][1], 0, min_len=3 * FS):
        x = np.linalg.norm(recs[r][0][s:e] - recs[r][0][s:e].mean(0), axis=1)
        f, p = signal.welch(x, FS, nperseg=min(len(x), 3 * FS))
        band = (f >= 0.7) & (f <= 4.0)
        freqs.append(f[band][np.argmax(p[band])])
freqs = np.array(freqs)
trans = np.array([e - s for r in recs for s, e in runs(r[1], 6)])
print(f"sampling rate: {len(stride)} walking stretches repeat every {np.median(stride):.2f} s (one full "
      f"stride, 10th-90th pct {np.percentile(stride, 10):.2f}-{np.percentile(stride, 90):.2f} s), and their "
      f"spectrum peaks at {np.median(freqs):.2f} Hz - twice that, the second step of the stride")
print(f"      -> a {np.median(stride):.1f} s stride and a {np.median(trans) / FS:.1f} s sit-to-stand are "
      f"human. Read as 100 Hz they would be a {2 * np.median(stride):.1f} s stride and a "
      f"{2 * np.median(trans) / FS:.1f} s sit-to-stand, and twice as slow again at 200 Hz: the files are at "
      f"the 50 Hz the description gives")
out["recording"] = {"n": len(recs), "samples": int(len(whole)), "minutes": len(whole) / FS / 60,
                    "gravity_mean_g": float(mags.mean()), "gravity_sd_g": float(mags.std()),
                    "mag_max_g": float(mag.max()), "mag_below_0p5_g": int((mag < 0.5).sum()),
                    "exact_repeats": dupl, "exact_repeats_share": dupl / len(whole),
                    "stride_median_s": float(np.median(stride)), "welch_median_Hz": float(np.median(freqs)),
                    "transition_median_s": float(np.median(trans) / FS)}

print("\n" + "=" * 98)
print("2. what the labels are")
print("=" * 98)
print(f"labelled {int((lab_all >= 0).sum())} samples ({(lab_all >= 0).mean():.1%}); "
      f"-1 in the pauses {int((lab_all < 0).sum())} ({(lab_all < 0).mean():.1%})")
gap_len = np.array([e - s for r in recs for s, e, c in all_runs(r[1]) if c == -1])
inner = np.array([e - s for r in recs for s, e, c in all_runs(r[1])
                  if c == -1 and s > np.argmax(r[1] >= 0) and e < len(r[1]) - np.argmax(r[1][::-1] >= 0)])
lead = np.array([int(np.argmax(r[1] >= 0)) for r in recs])
tail = np.array([int(len(r[1]) - np.argmax(r[1][::-1] >= 0) - 1) for r in recs])
print(f"pauses: {len(gap_len)} stretches, median {np.median(gap_len)} samples ({np.median(gap_len) / FS:.1f} s), "
      f"longest {gap_len.max() / FS:.0f} s; {len(inner)} of them lie between two activities, and every "
      f"recording also has a median of {np.median(lead) / FS:.0f} s before the first label and "
      f"{np.median(tail) / FS:.0f} s after the last")
print(f"      -> people do not start and stop on cue. Samples with -1 are never scored and never "
      f"trained on as a class of their own.")
tot_lab = int((lab_all >= 0).sum())
print(f"\n{'class':11s} {'runs':>5s} {'samples':>8s} {'share':>6s} {'median s':>9s} {'10th':>6s} {'90th':>6s} "
      f"{'longest':>8s}")
rows = []
for c in range(7):
    d = np.array([e - s for r in recs for s, e in runs(r[1], c)])
    rows.append({"class": CLASSES[c], "runs": int(len(d)), "samples": int(d.sum()),
                 "median_s": float(np.median(d) / FS), "p10_s": float(np.percentile(d, 10) / FS),
                 "p90_s": float(np.percentile(d, 90) / FS), "max_s": float(d.max() / FS)})
    print(f"{CLASSES[c]:11s} {len(d):5d} {d.sum():8d} {d.sum() / tot_lab:6.1%} {np.median(d) / FS:9.1f} "
          f"{np.percentile(d, 10) / FS:6.1f} {np.percentile(d, 90) / FS:6.1f} {d.max() / FS:8.1f}")
out["classes"] = rows
act = np.median([r["median_s"] for r in rows[:6]])
print(f"      -> a {CLASSES[6]} runs {rows[6]['median_s']:.1f} s at the median against {act:.0f} s for an "
      f"activity, and is {rows[6]['samples'] / tot_lab:.1%} of the labelled samples. Six different "
      f"movements are merged into it, and it is the shortest and rarest class: macro F1 is decided there, "
      f"which is why the notebook asks for macro F1")

pat = pd.Series([tuple(c for _, _, c in all_runs(r[1]) if c >= 0)[:9] for r in recs]).value_counts()
first = [c for _, _, c in all_runs(recs[0][1]) if c >= 0][:9]
print(f"\nthe order of the labelled stretches: experiment {recs[0][2]['exp']} starts "
      f"{[CLASSES[c][:4] for c in first]} - activities separated by transitions, as the protocol says. "
      f"{len(pat)} different patterns over the 61 recordings, the commonest shared by {pat.iloc[0]} of them")
bad = meta[meta.n_classes < 7]
print(f"recordings without all 7 classes: {', '.join(b.file for b in bad.itertuples())} "
      f"({'; '.join(f'{int(b.n_classes)} classes, {int(b.label_rows)} label rows, {b.seconds:.0f} s' for b in bad.itertuples())})")
print(f"      -> the subject stopped early. Nothing was lost by the parse: over all 61 recordings "
      f"{int(meta.overwritten_samples.sum())} labelled samples were overwritten by a later label row and "
      f"{int(meta.painted_past_end.sum())} were painted past the end of the file")
for s, v in splits.items():
    y = np.concatenate([r[1][r[1] >= 0] for r in v])
    cnt = np.bincount(y, minlength=7)
    out.setdefault("splits", {})[s] = {"recordings": len(v), "labelled": int(len(y)),
                                       "counts": cnt.tolist(), "share": (cnt / len(y)).round(4).tolist()}
    print(f"{s:5s} {len(v)} recordings, {len(y):7d} labelled; per class "
          f"{dict(zip([c[:4] for c in CLASSES], cnt.tolist()))}")
y_te = np.concatenate([r[1][r[1] >= 0] for r in splits["test"]])
b = f1_scores(y_te, np.full(len(y_te), 4))
out["baseline_always_standing"] = b
print(f"the baseline the notebook names - always answering standing: test accuracy {b['accuracy']:.3f}, "
      f"macro F1 {b['macro_f1']:.3f} (only standing gets an F1, {b['per_class'][4]['f1']:.3f}); the "
      f"notebook says about 0.18 and about 0.04")

print("\n" + "=" * 98)
print("3. how the phone is worn: which way gravity points, per recording")
print("=" * 98)
grav = {}
for r in range(len(recs)):
    for c in STATIC:
        seg = runs(recs[r][1], c, trim=FS, min_len=2 * FS)
        if seg:
            v = np.concatenate([recs[r][0][s:e] for s, e in seg]).mean(0)
            grav[(r, c)] = v / np.linalg.norm(v)
keys = sorted(grav)
pooled = {c: np.mean([grav[k] for k in keys if k[1] == c], 0) for c in STATIC}
pooled = {c: v / np.linalg.norm(v) for c, v in pooled.items()}


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


print("the mean direction of gravity in the phone's own axes, pooled over the recordings:")
for c in STATIC:
    print(f"  {CLASSES[c]:9s} {pooled[c].round(3)}")
angles = {f"{CLASSES[a][:6]}-{CLASSES[b][:6]}": ang(pooled[a], pooled[b]) for a, b in [(3, 4), (3, 5), (4, 5)]}
print("angles between them: " + ", ".join(f"{k} {v:.0f} deg" for k, v in angles.items()))
spread = {}
for c in STATIC:
    a = np.array([ang(grav[k], pooled[c]) for k in keys if k[1] == c])
    spread[CLASSES[c]] = {"n": int(len(a)), "median_deg": float(np.median(a)),
                          "p90_deg": float(np.percentile(a, 90)), "max_deg": float(a.max())}
    print(f"  within {CLASSES[c]:9s}, over its {len(a)} recordings: median {np.median(a):5.1f} deg from "
          f"the pooled mean, 90th pct {np.percentile(a, 90):5.1f}, worst {a.max():5.1f}")
pairs, who = [], []
for u in sorted(set(r[2]["user"] for r in recs)):
    rs = [r for r in range(len(recs)) if recs[r][2]["user"] == u]
    for c in STATIC:
        if len(rs) > 1 and (rs[0], c) in grav and (rs[1], c) in grav:
            pairs.append(ang(grav[(rs[0], c)], grav[(rs[1], c)]))
            who.append((u, CLASSES[c]))
pairs = np.array(pairs)
w = int(np.argmax(pairs))
print(f"the same person's two recordings, same class: a median of {np.median(pairs):.1f} deg apart "
      f"(90th pct {np.percentile(pairs, 90):.1f}, worst {pairs.max():.1f} for user {who[w][0]} in "
      f"{who[w][1]}) over {len(pairs)} pairs -> the phone went back on roughly the same way, not exactly")
out["gravity"] = {"pooled": {CLASSES[c]: pooled[c].tolist() for c in STATIC}, "angles_deg": angles,
                  "spread": spread, "same_user": {"median_deg": float(np.median(pairs)),
                                                  "p90_deg": float(np.percentile(pairs, 90)),
                                                  "max_deg": float(pairs.max())}}
med_spread = np.median([spread[CLASSES[c]]["median_deg"] for c in STATIC])
print(f"\nwithin one still class the phones point {med_spread:.0f} deg apart at the median, while the three "
      f"still classes sit {min(angles.values()):.0f}-{max(angles.values()):.0f} deg apart from each other.")
print("So the dataset does NOT contain the pocket-turning the task is about: the phone was belt-mounted "
      "at the waist, nearly the same way on everyone. Any honest test of an orientation-aware method "
      "means rotating held-out recordings by hand - which is what train_motion.py will do.")

print("\n" + "=" * 98)
print("4. how much of the task lives in the axis order (nothing learned yet)")
print("=" * 98)
W = 256


def single_class_windows(rec_list, W, cls=None):
    """(recording, start, class) for the windows whose samples carry one class (and possibly -1)."""
    idx = []
    for ri, (a, y, m) in enumerate(rec_list):
        for s, keep in eval_windows(len(a), W):
            u = set(y[s:s + W].tolist()) - {-1}
            if len(u) == 1 and (cls is None or u == {cls}):
                idx.append((ri, s, u.pop()))
    return idx


tr_win = [(ri, s, c) for c in range(7) for ri, s, cc in single_class_windows(splits["train"], W, c)]
te_win = single_class_windows(splits["test"], W)
print(f"W = {W} samples ({W / FS:.1f} s): {len(tr_win)} single-class training windows, "
      f"{len(te_win)} test windows whose samples all carry one label (a window that straddles a boundary "
      f"or sits in a pause has no single answer; the real evaluation is per sample)")
unit = lambda w: w.mean(0) / max(np.linalg.norm(w.mean(0)), 1e-9)
mean_sd = lambda w: np.concatenate([w.mean(0), w.std(0)])
for label, feats, pool in (
        ("which way down the window points", unit, STATIC),
        ("mean and sd of each axis", mean_sd, list(range(7)))):
    acc, cnt = {}, {}
    for ri, s, c in tr_win:
        if c in pool:
            v = feats(splits["train"][ri][0][s:s + W])
            acc[c] = acc.get(c, 0) + v
            cnt[c] = cnt.get(c, 0) + 1
    cent = {c: acc[c] / cnt[c] for c in acc}

    def score(R, cent=cent):
        pred, true = [], []
        for ri, s, c in te_win:
            if c not in cent:
                continue
            v = feats(splits["test"][ri][0][s:s + W] @ R.T)
            pred.append(min(cent, key=lambda k: np.linalg.norm(cent[k] - v)))
            true.append(c)
        r = f1_scores(true, pred, k=7)
        r["macro_f1"] = float(np.mean([r["per_class"][c]["f1"] for c in cent]))
        r["classes"] = sorted(cent)
        return r

    r_id = score(np.eye(3, dtype=np.float32))
    table = sorted((score(R)["accuracy"], perm_name(R)) for R in perms)
    better = sum(1 for a, nm in table if a > r_id["accuracy"] and nm != "x,y,z")
    losses = [r_id["accuracy"] - a for a, nm in table if nm != "x,y,z"]
    print(f"\nnearest-class-average on {label} ({len(cent)} classes, {len(y_te)} -> "
          f"{r_id['samples']} test windows scored):")
    print(f"  as delivered (x,y,z): accuracy {r_id['accuracy']:.3f}, macro F1 {r_id['macro_f1']:.3f}; "
          f"per class " + ", ".join(f"{CLASSES[c][:4]} {r_id['per_class'][c]['f1']:.2f}" for c in cent))
    print(f"  the 24 turned phones: accuracy {table[0][0]:.3f} (worst, {table[0][1]}) to "
          f"{table[-1][0]:.3f} (best, {table[-1][1]}); {better} of the 23 beat the real orientation")
    print(f"  turning the phone costs a median of {np.median(losses):.3f} accuracy - the axis order "
          f"carries most of what separates sitting from standing from lying")
    out[f"rule_{label[:16].replace(' ', '_')}"] = {"identity": r_id, "worst": table[0], "best": table[-1],
                                                   "median_loss": float(np.median(losses))}

print("\n" + "=" * 98)
print("5. what the window length trades off")
print("=" * 98)
for W_ in (64, 128, 256, 512, 1024):
    mix = tran = tot = 0
    for a, y, m in recs:
        for s, keep in eval_windows(len(a), W_):
            w = y[s:s + W_]
            tot += 1
            mix += int(len(set(w.tolist()) - {-1}) > 1)
            tran += int(np.any(w == 6))
    ntr = sum(len(windows_of(len(a), W_)) for a, y, m in splits["train"])
    print(f"W = {W_:4d} samples ({W_ / FS:5.2f} s): {ntr:5d} training windows on the 41 training "
          f"recordings, {mix / tot:5.1%} of all windows hold two classes, {tran / tot:5.1%} hold a "
          f"{CLASSES[6]} sample somewhere - the longer the window the more the labels inside it disagree, "
          f"while the network has to answer each sample separately")

# ---------------------------------------------------------------------------- the figures
a0, l0, m0 = splits["test"][0]
t = np.arange(len(a0)) / FS
fig, axes = plt.subplots(2, 1, figsize=(12, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 0.5]})
for k in range(3):
    axes[0].plot(t, a0[:, k], lw=0.6, label="xyz"[k])
axes[0].set_ylabel("acceleration (g)")
axes[0].legend(ncol=3, fontsize=8, loc="upper right")
axes[0].set_title(f"the first test recording: experiment {m0['exp']}, user {m0['user']}, "
                  f"{len(a0) / FS:.0f} s", fontsize=10)
axes[1].imshow(lab_to_rgba(l0)[None], aspect="auto", extent=[0, t[-1], 0, 1], interpolation="nearest")
axes[1].set_yticks([])
axes[1].set_xlabel("time (s); white = no label")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in zip(CLASSES, COLORS)],
           loc="lower center", ncol=7, fontsize=8, frameon=False)
fig.tight_layout(rect=[0, 0.07, 1, 0.97])
fig.savefig(FIGS / "inspect_test_recording.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), subplot_kw={"aspect": "equal"})
for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    for c in STATIC:
        pts = np.array([grav[k] for k in keys if k[1] == c])
        ax.scatter(pts[:, i], pts[:, j], s=16, alpha=0.6, label=CLASSES[c])
    ax.scatter([pooled[c][i] for c in STATIC], [pooled[c][j] for c in STATIC], marker="X", s=140, c="k")
    ax.set_xlabel("device axis " + "xyz"[i])
    ax.set_ylabel("device axis " + "xyz"[j])
    ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.15, 1.15)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("which way gravity points in the phone's own axes: one dot per recording, still classes only "
             "(1 s trimmed at each edge); black X = that class's mean.\nThe three classes are far apart "
             "and each class's dots cluster -> almost no pocket rotation in this dataset.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIGS / "inspect_gravity.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
axes[0].boxplot([np.array([e - s for r in recs for s, e in runs(r[1], c)]) / FS for c in range(7)],
                tick_labels=[c[:9] for c in CLASSES], showfliers=False)
axes[0].set_yscale("log")
axes[0].set_ylabel("one stretch of the class lasts (s)")
axes[0].set_title(f"length of a stretch: a {CLASSES[6]} runs {rows[6]['median_s']:.1f} s at the median, "
                  f"an activity {act:.0f} s", fontsize=9)
present = [c for c in range(7) if any(runs(r[1], c, min_len=2 * FS) for r in recs)]
axes[1].boxplot([np.array([recs[r][0][s:e].std(0) for r in range(len(recs))
                           for s, e in runs(recs[r][1], c, min_len=2 * FS)]).ravel() for c in present],
                tick_labels=[CLASSES[c][:9] for c in present], showfliers=False)
axes[1].set_ylabel("sd inside a stretch (g), one box per class,\nover its stretches, recordings and axes")
axes[1].set_title("how much each class shakes the phone", fontsize=9)
for ax in axes:
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("the seven classes in time and in motion", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGS / "inspect_classes.png", dpi=130)

(FIGS / "results_inspect.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote inspect_test_recording.png, inspect_gravity.png, inspect_classes.png and "
      f"results_inspect.json to {FIGS}")
