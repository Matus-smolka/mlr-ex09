"""Shared pieces of Task 2: the classes, the windows, the 24 ways to wear the phone, the scores.

Nothing here trains anything - it is the small print that the inspection, the baselines and the
training script all have to agree on:

* **the label of a sample, not of a window.** The network is `UNet(1, 3, 7, widths)`: 3 axes in,
  7 logits per sample out, every sample of a window labelled. Samples with the label -1 (the pauses
  between the activities) are carried along so that the windows line up, and dropped from every score.
* **evaluation windows never overlap and never pad.** A recording is cut into full windows of `W`
  samples; the last window is slid back instead of padded, and only the samples it covers that no
  earlier window covered are kept. Every sample of the recording is therefore predicted exactly once,
  always from a full window of real signal - `eval_windows` is the one place that logic lives.
* **turning the phone does not change the label.** `rot` is a (3, 3) signed permutation matrix
  applied to the axes; the person is doing the same thing either way, so `y` is untouched. That is
  what makes the whole orientation experiment honest: the target never moves, only the input does.
"""
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset

FS = 50                                                     # Hz, from the dataset's own description
CLASSES = ["walking", "upstairs", "downstairs", "sitting", "standing", "lying", "transition"]
COLORS = ["#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#8ab17d", "#9b5de5", "#adb5bd"]
STATIC = [3, 4, 5]                                          # sitting, standing, lying: quasi still


def signed_permutations():
    """The 24 proper rotations that map the device's axes onto its own axes, as (24, 3, 3).

    A phone lying in a pocket, turned through a multiple of 90 degrees, reports the same physical
    acceleration with the axes permuted and some of them negated: 3! orders x 2^3 sign choices,
    halved because half of those are mirror images, which no turning of the phone can produce.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3), np.float32)
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if np.linalg.det(m) > 0:                        # a rotation, not a mirror
                mats.append(m)
    return np.stack(mats)


def perm_name(m):
    """'x,-z,y'-style name of a signed permutation, for labelling a table."""
    out = []
    for row in m:
        k = int(np.flatnonzero(row != 0)[0])
        out.append(("-" if row[k] < 0 else "") + "xyz"[k])
    return ",".join(out)


def random_rotation(rng):
    """A uniformly random orientation: a Haar-random element of SO(3), from the Q-R of a Gaussian."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))                    # a unique sign convention, so the map is 1-to-1
    if np.linalg.det(q) < 0:                                # a third of them come out mirrored
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def canonical_rotation(g):
    """The shortest rotation that turns the direction `g` onto +x, the way gravity points in this data.

    Rotating every window so that its own gravity points along +x is the textbook way to make a network
    ignore how the phone is worn - but it costs the one angle that tells sitting from standing (they
    differ by a tilt of only ~19 degrees), and it leaves the roll about gravity free. It also has to be
    done by whatever feeds the network, so a model trained on canonicalized windows cannot be dropped
    into the app, which feeds raw axes.
    """
    g = np.asarray(g, np.float64)
    g = g / max(np.linalg.norm(g), 1e-12)
    e = np.array([1.0, 0.0, 0.0])
    v = np.cross(g, e)                                      # Rodrigues, shortest arc from g onto +x
    s, c = float(np.linalg.norm(v)), float(np.dot(e, g))
    if s < 1e-9:                                            # already along +x, or exactly along -x
        return np.diag([1.0, 1.0, 1.0] if c > 0 else [-1.0, 1.0, -1.0]).astype(np.float32)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)).astype(np.float32)


def windows_of(n, W):
    """Starts of the non-overlapping training windows of a recording of n samples; the tail is dropped."""
    return np.arange(0, max(0, n - W + 1), W)


def eval_windows(n, W):
    """(start, keep_from) pairs covering 0..n exactly once, every window a full W samples long."""
    if n < W:
        return [(0, 0)]                                     # one short recording: caller pads
    out, s = [], 0
    while s + W < n:
        out.append((s, s))
        s += W
    out.append((n - W, s))                                  # slid back over the previous window
    return out


class AccWindows(Dataset):
    """Windows of the recordings of one split: x (3, W) in g, y (W,) with values 0-6 or -1.

    rot        a (3, 3) matrix applied to the axes of every window (the phone worn turned).
    aug        None | "perm" (a random one of the 24 axis-aligned orientations per window)
               | "so3" (a random continuous orientation per window).
    canon      turn every window so that its own gravity points at +x (see canonical_rotation). Applied
               after rot, so a canon model sees the same input whichever way the phone was turned.
    require_label  only used when the index is built here (index=None): leave out a window whose samples
               are all -1. Such a window carries no supervision whatever, and cross-entropy in which
               every sample is ignored returns NaN, not 0 (measured, torch 2.14) - one NaN step of Adam
               would reset every weight in the network. They are the long pauses before the first
               activity and after the last one: 16.3% of the training windows, 13.6% of the validation
               ones. An explicit `index` is taken as given, which is how `run()` keeps every evaluation
               window - predicting a pause is harmless, since only labelled samples are scored.
    """

    def __init__(self, records, W, index=None, rot=None, aug=None, canon=False, seed=0,
                 require_label=True):
        self.records, self.W, self.rot = records, W, None if rot is None else np.asarray(rot, np.float32)
        self.aug, self.canon, self.rng = aug, canon, np.random.default_rng(seed)
        self.index = [] if index is None else list(index)
        if index is None:
            for i in range(len(records)):                  # a record is (acc, lab) or (acc, lab, meta)
                lab = np.asarray(records[i][1])
                for s in windows_of(len(records[i][0]), W):
                    if not require_label or (np.asarray(lab)[s:s + W] >= 0).any():
                        self.index.append((i, int(s)))
        if not self.index:
            raise RuntimeError("no windows: is W longer than the recordings?")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        acc, lab = self.records[i][0], self.records[i][1]
        x = np.asarray(acc[s:s + self.W], np.float32).T                 # (3, W)
        if self.rot is not None:
            x = self.rot @ x
        elif self.aug == "perm":
            x = signed_permutations()[self.rng.integers(24)].astype(np.float32) @ x
        elif self.aug == "so3":
            x = random_rotation(self.rng) @ x
        if self.canon:                                      # turn the window so its gravity points at +x
            x = canonical_rotation(x.mean(axis=1)) @ x
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.asarray(lab[s:s + self.W], np.int64))


def load_splits(cache):
    """(split, list of (acc (T,3) float32, lab (T,) int8, meta dict)) straight from the parse's cache."""
    d = np.load(cache)
    acc, lab, off = d["acc"], d["lab"], d["offsets"]
    recs = [(acc[off[i]:off[i + 1]], lab[off[i]:off[i + 1]], {"exp": int(d["exp"][i]), "user": int(d["user"][i])})
            for i in range(len(off) - 1)]
    return {s: [recs[i] for i in d[f"split_{s}"]] for s in ("train", "valid", "test")}, recs


def f1_scores(y, p, k=7):
    """Per-class precision, recall, F1 and their macro average, from two arrays of class indices."""
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for c in range(k):
        tp = float(np.sum((p == c) & (y == c)))
        n_pred, n_true = float(np.sum(p == c)), float(np.sum(y == c))
        prec, rec = (tp / n_pred if n_pred else 0.0), (tp / n_true if n_true else 0.0)
        out.append({"class": c, "support": int(n_true), "predicted": int(n_pred),
                    "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
    return {"per_class": out, "macro_f1": float(np.mean([o["f1"] for o in out])),
            "accuracy": float(np.mean(p == y)), "samples": int(len(y))}


def confusion(y, p, k=7):
    """confusion[true, predicted]; rows sum to the support of each true class."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    assert y.size and y.min() >= 0 and p.min() >= 0 and y.max() < k and p.max() < k
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def timeline(rec, pred, path=None, title="", truth=True):
    """The true and the predicted label strip of one recording, as in the notebook's own figure."""
    import matplotlib.pyplot as plt
    t = np.arange(len(pred)) / FS
    rows = ([lab_to_rgba(np.where(lab < 0, -1, lab))] if truth else []) + [lab_to_rgba(pred)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.0 * len(rows) + 1.4), sharex=True)
    for ax, row, name in zip(np.atleast_1d(axes), rows, (["true"] if truth else []) + ["predicted"]):
        ax.imshow(row[None], aspect="auto", extent=[0, t[-1], 0, 1], interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in zip(CLASSES, COLORS)],
               loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig


def lab_to_rgba(y):
    """Label strip -> RGB, with -1 (no label) left white so a pause between activities is visible."""
    import matplotlib.colors
    palette = np.array([matplotlib.colors.to_rgb(c) for c in COLORS])
    out = np.ones((len(y), 3))
    ok = np.asarray(y) >= 0
    out[ok] = palette[np.asarray(y)[ok]]
    return out
