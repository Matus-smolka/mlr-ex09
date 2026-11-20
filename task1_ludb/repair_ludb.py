"""Repair the LUDB labels for delineation: stop teaching the network that real beats are background.

The cardiologists annotated only the complete beats, so every record starts and ends with about one
beat that carries no label (413 of them over the 200 records). Three more records - 94, 103 and 110,
all in train, all three among the records whose lead II annotation file the parser drops symbols from
- leave 11 further QRS complexes unlabelled inside the annotated span (see reconcile_interior.py for
the exact count and for the one borderline case in test record 73 that is the same beat detected
twice). Left alone, a U-Net that correctly marks those beats is punished for it, and worse, it is
trained to answer "background" at a textbook QRS complex.

The fix masks those samples instead of cutting them away:

  * the signal is kept whole - the network still sees the neighbouring beats, which is what tells
    you which beat a P wave belongs to, and all records keep the same length so nothing about
    batching changes; cropping would also leak the answer into the input, because a record that
    begins a few samples after an annotated onset is itself a hint that a beat boundary is there;
  * everything outside [first annotated onset - buffer, last annotated offset + buffer] gets the
    label -1, which nn.CrossEntropyLoss(ignore_index=-1) skips, exactly like the padding of
    example 7; inside that region the file's own labels stand, background included, so a beat the
    network invents inside the buffer still counts as the error it is;
  * inside the annotated span, a +-150 ms window around each QRS-like peak that carries no label
    gets -1 too - that is a known label error, not a choice about where to look.

The buffer (200 ms) is there because a mask that stops exactly at the annotation makes the score
mean "accuracy where the annotators happened to look". It costs 2 of the 13 unlabelled beats being
counted against us again, which is the point of it.

Three conventions are written to the cache so the same predictions can be scored under each:

  y          span + buffer                    <- ours, the number we report
  y_tight    span, no buffer                  -> shows what the buffer costs
  ig_signal  margin from the signal's own first/last QRS peak, no labels used for the edge at all
  y_raw      nothing masked, as the notebook parses it

Usage:  py -I task1_ludb/repair_ludb.py <in.npz> <out.npz> <figs-dir> [buffer-ms]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks

FS = 500
WAVES = ["background", "P", "QRS", "T"]
HALF_QRS = 75                     # +-75 samples = +-150 ms around an unlabelled QRS peak
SRC, DST, FIGS = sys.argv[1], sys.argv[2], Path(sys.argv[3])
BUFFER_MS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
BUF = int(round(BUFFER_MS / 1000 * FS))
FIGS.mkdir(parents=True, exist_ok=True)
d = np.load(SRC, allow_pickle=True)
x, y, waves, rhythm = d["x"], d["y"], d["waves"], d["rhythm"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of = waves[:, 0]
n, T = y.shape
B, A = butter(3, [8 / (FS / 2), 20 / (FS / 2)], btype="band")

span = np.array([[waves[rec_of == r][:, 2].min(), waves[rec_of == r][:, 4].max()] if (rec_of == r).any()
                 else (0, T - 1) for r in range(n)])


def qrs_peaks(s):
    """Sensitive QRS detector: envelope of the 8-20 Hz band, peaks above 40 % of its own maximum."""
    env = gaussian_filter1d(np.abs(filtfilt(B, A, s)), 2.0)
    return find_peaks(env, height=0.4 * env.max(), distance=int(0.35 * FS))[0]


peaks = [qrs_peaks(s) for s in x]

# --- the three edge rules
def mask_edges(lo, hi):
    """-1 outside [lo, hi], the parsed label inside it."""
    m = y.copy()
    for r in range(n):
        m[r, :max(lo[r] - BUF, 0)] = -1
        m[r, min(hi[r] + BUF + 1, T):] = -1
    return m


lo_span = span[:, 0]
hi_span = span[:, 1]
lo_sig = np.array([max(p.min() - 1, 0) for p in peaks])          # the signal's own first/last beat
hi_sig = np.array([min(p.max() + 1, T - 1) for p in peaks])
y_buf, y_tight, y_sig = mask_edges(lo_span, hi_span), y.copy(), mask_edges(lo_sig, hi_sig)
for r in range(n):                                                # the buffer must contain no annotation
    assert (y[r, max(lo_span[r] - BUF, 0):lo_span[r]] == 0).all()
    assert (y[r, hi_span[r] + 1:min(hi_span[r] + BUF + 1, T)] == 0).all()
for r in range(n):
    y_tight[r, :lo_span[r]] = -1
    y_tight[r, hi_span[r] + 1:] = -1

# --- inside the annotated span: QRS-like peaks the annotations never mention (a label error, all masks)
interior, in_buffer = [], []
for r in range(n):
    w = waves[rec_of == r]
    q = np.sort(w[w[:, 1] == 2][:, 3]) if (w[:, 1] == 2).any() else np.array([], int)
    lo, hi = span[r]
    for p in peaks[r]:
        explained = (len(q) and np.min(np.abs(q - p)) <= 50) or any(o <= p <= f for _, _, o, _, f in w)
        if explained:
            continue
        if lo <= p <= hi:
            interior.append((r, int(p)))
            for arr in (y_buf, y_tight, y_sig):
                arr[r, max(p - HALF_QRS, 0):min(p + HALF_QRS, T - 1) + 1] = -1
        elif max(lo - BUF, 0) <= p < lo or hi < p <= min(hi + BUF, T - 1):
            in_buffer.append((r, int(p)))                         # the price of the buffer: judged as background

for name, arr in (("span", y_tight), ("span + %d ms" % BUFFER_MS, y_buf),
                  ("signal +- %d ms" % BUFFER_MS, y_sig)):
    kept = (arr >= 0).mean()
    print(f"{name:18s} keeps {kept:6.1%} of the samples"
          f" | per split " + ", ".join(f"{s} {(arr[splits[s]] >= 0).mean():.1%}" for s in splits))
print(f"masked inside the annotated span: {len(interior)} unlabelled QRS complexes in "
      f"{len(set(r for r, _ in interior))} records -> {sorted(set(r for r, _ in interior))}, "
      f"splits {[(s, sum(1 for r, _ in interior if r in splits[s])) for s in splits]}")
print(f"unlabelled QRS complexes the buffer lets back into the judged region: {len(in_buffer)} "
      f"in {sorted(set(r for r, _ in in_buffer))} - counted as false positives, on purpose")

# --- verification: is there any QRS-like peak left that is asserted to be background?
inside_left = buffer_left = 0
for r in range(n):
    w = waves[rec_of == r]
    for p in peaks[r]:
        if y_buf[r, p] != 0 or any(o - 50 <= p <= f + 50 for _, _, o, _, f in w):
            continue                                   # labelled, or covered by an annotation: fine
        if span[r, 0] <= p <= span[r, 1]:
            inside_left += 1
        else:
            buffer_left += 1
print(f"after the repair, unlabelled QRS-like peaks left inside the annotated span: {inside_left} (must be 0)")
print(f"                              ...and left inside the {BUFFER_MS} ms buffer: {buffer_left} "
      f"(on purpose - a beat the network invents there is still counted as a false positive)")
assert inside_left == 0 and buffer_left == len(in_buffer)

kept = y_buf >= 0
print("\nclass fractions among the kept samples:")
print("  before:", dict(zip(WAVES, np.round(np.bincount(y.ravel(), minlength=4) / y.size, 3))))
print("  after :", dict(zip(WAVES, np.round(np.bincount(y_buf[kept], minlength=4) / kept.sum(), 3))))
print(f"waves fully kept: {sum(1 for r, c, o, _, f in waves if (y_buf[r, o:f + 1] == c).all())} of {len(waves)}; "
      f"waves touched by the mask: {sum(1 for r, c, o, _, f in waves if (y_buf[r, o:f + 1] != c).any())}")

ig_signal = np.where(y_sig >= 0, 0, -1).astype(np.int8)            # a mask, not a labelling
assert (ig_signal >= 0).mean() == (y_sig >= 0).mean() < 1
extra = {k: d[k] for k in ("x12", "lead_names") if k in d.files}      # the other leads, carried through
np.savez_compressed(DST, x=x, y=y_buf, y_tight=y_tight, y_raw=y, ig_signal=ig_signal, waves=waves,
                    rhythm=rhythm, span=span, buffer=np.int64(BUF), **extra,
                    **{f"split_{s}": v for s, v in splits.items()})
print("wrote", DST)

# --- figures: before and after, on the record with the tallest unlabelled beats and the broken one
for r, tag in ((int(splits["test"][0]), "edge"), (103, "interior")):
    t = np.arange(T) / FS
    w = waves[rec_of == r]
    unl = [p for p in peaks[r] if y[r, p] == 0 and not ((w[:, 2] - 50 <= p) & (p <= w[:, 4] + 50)).any()]
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.8), sharex=True)
    for ax, lab, arr in ((axes[0], "as parsed:\nunlabelled beats\nare 'background'", y[r]),
                         (axes[1], f"after the repair:\nthose samples\nare -1 (ignored)", y_buf[r])):
        ax.fill_between(t, -4, 4, where=(y_buf[r] == -1), color="0.7", alpha=0.8, linewidth=0)
        ax.fill_between(t, -4, 4, where=(y_tight[r] == -1) & (y_buf[r] >= 0), color="0.85", alpha=0.8,
                        linewidth=0, label=f"the {BUFFER_MS} ms buffer, judged")
        ax.plot(t, x[r], "k", linewidth=0.55)
        for c, col in ((1, "tab:green"), (2, "tab:red"), (3, "tab:blue")):
            ax.fill_between(t, -4, 4, where=arr == c, color=col, alpha=0.35, linewidth=0)
        ax.plot(np.asarray(unl) / FS, np.full(len(unl), 3.9), "m^", markersize=6,
                label="QRS-like peak, no annotation")
        ax.axvline(lo_span[r] / FS, color="k", ls=":", linewidth=0.8)
        ax.axvline(hi_span[r] / FS, color="k", ls=":", linewidth=0.8)
        ax.set_ylim(-4.5, 4.8)
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"record {r} ({tag}): dotted lines = the annotated span, dark grey = ignored by the loss "
                 f"and by the metric, light grey = the {BUFFER_MS} ms buffer, judged as background", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / f"repair_{tag}_{r}.png", dpi=130)
    print("wrote", FIGS / f"repair_{tag}_{r}.png")
