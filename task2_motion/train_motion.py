"""Task 2: a plain 1D U-Net as the baseline, and what to do when the phone is worn turned.

Two things are asked for: train a model plainly - that is the baseline - and then deal with the phone
being rotated in a pocket, by trying the axes switched around and keeping a switch only when the window
really does match better. The second part needs a place where the phone *is* turned, and the inspection
of the data showed HAPT barely has any: the belt-mounted phone points about 10 degrees off at the median
while the still classes are 19-93 degrees apart. So the rotation is applied by hand to held-out
recordings, which is also the only way to score a claim of rotation robustness at all.

Four arms, all the same 1D U-Net of Part 1 (`UNet(1, 3, 7, widths)`, 3 axes in, 7 logits per sample):

  A_plain  trained plainly, nothing about orientation in it. The baseline the task asks for.
  B_rot24  trained with every window turned to one of the 24 axis-aligned orientations at random.
  C_rotSO3 trained with every window turned to a random continuous orientation.
  D_gravity_aligned  every window turned so that its own gravity points along +x before the network sees
           it: the textbook way to make a network ignore the orientation, and it needs no search at run
           time. Both B/C and D buy invariance, and invariance is not free here - sitting and standing
           differ by a tilt of about 19 degrees, so what makes a network orientation-blind also blurs
           them. The measurements say how much.

and one thing added on top of the trained networks, which is the idea asked for:

  search   the phone is in some orientation R we do not know, so the network sees R*x. For every
           candidate orientation C of a fixed list (the 24 ways of turning the phone through multiples of
           90 degrees) feed the network C*R*x and read how well it matches; keep the switch away from
           C = identity only if it beats the plain reading by more than a margin. That is exactly "try
           switching the axes around, and keep it only if it matches better". Two ways of pooling the
           evidence - per window, which is what the task describes, and per recording, because a phone
           does not turn itself around while you walk - and the margin and the pooling are chosen on the
           validation users, never on the test users.

Everything is scored sample by sample, with the -1 samples dropped, on the same windows of the same
users. Two regimes are reported side by side because they answer different questions: the test users as
delivered, where a search can only cost something since nothing is turned, and the test users with a
known rotation applied, where it is supposed to pay.

Usage:  py -I task2_motion/train_motion.py <cache/hapt.npz> <figs-dir>
            [--epochs 400] [--batch 32] [--lr 1e-3] [--arms A,B,C,D] [--quick] [--widths 16,32,64,128]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402

import part1                                                          # noqa: E402
from motion_data import (CLASSES, AccWindows, canonical_rotation, eval_windows,       # noqa: E402
                         f1_scores, load_splits, perm_name, random_rotation,
                         signed_permutations, windows_of)

ap = argparse.ArgumentParser()
ap.add_argument("cache", type=Path)
ap.add_argument("figs", type=Path)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--arms", default="A,B,C,D")
ap.add_argument("--widths", default="16,32,64,128")
ap.add_argument("--stress", type=int, default=0, help="use only this many worn orientations (0 = all)")
ap.add_argument("--recordings", type=int, default=0, help="use only the first N recordings of each split")
ap.add_argument("--quick", action="store_true")
A = ap.parse_args()
FIGS, W = A.figs, 256
FIGS.mkdir(parents=True, exist_ok=True)
EPOCHS = 12 if A.quick else A.epochs
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ARMS = {"A": dict(aug=None, canon=False, name="A_plain",
                  note="trained plainly - the baseline the task asks for"),
        "B": dict(aug="perm", canon=False, name="B_rot24",
                  note="every training window turned to one of the 24 axis-aligned orientations"),
        "C": dict(aug="so3", canon=False, name="C_rotSO3",
                  note="every training window turned to a random continuous orientation"),
        "D": dict(aug=None, canon=True, name="D_gravity_aligned",
                  note="every window turned so that its own gravity points at +x before the network sees it")}
WIDTHS = [int(w) for w in A.widths.split(",")]
ARM_KEYS = [k for k in A.arms.split(",") if k in ARMS]
splits, recs = load_splits(A.cache)
if A.recordings:                                              # a short run to check the whole pipeline
    splits = {s: v[:A.recordings] for s, v in splits.items()}
Y = {s: np.concatenate([r[1] for r in splits[s]]) for s in splits}
out = {"window": W, "epochs": EPOCHS, "batch": A.batch, "lr": A.lr, "widths": WIDTHS,
       "arms_tried": ARM_KEYS, "arm_notes": {k: ARMS[k]["note"] for k in ARM_KEYS}}

print("=" * 100)
print("0. the pipeline, checked before anything is trained")
print("=" * 100)
for v in (np.array([1.0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1]), np.array([0.3, -0.6, 0.75])):
    Rc = canonical_rotation(v)
    assert np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-5) and np.linalg.det(Rc) > 0
    assert np.allclose(Rc @ v / np.linalg.norm(v), [1, 0, 0], atol=1e-5)
model0 = part1.UNet(1, 3, 7, WIDTHS)
N_PAR = sum(p.numel() for p in model0.parameters())
assert model0(torch.zeros(2, 3, W)).shape == (2, 7, W)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
assert abs(float(loss_fn(torch.zeros(1, 7, 4), torch.zeros(1, 4, dtype=torch.long))) - np.log(7)) < 1e-5
BLANK = float(loss_fn(torch.zeros(1, 7, 4), torch.full((1, 4), -1)))
assert np.isnan(BLANK), "measured, and depended on: a window in which every sample is ignored costs NaN, not 0"
all_tr, tr_set = AccWindows(splits["train"], W, require_label=False), AccWindows(splits["train"], W)
all_va, va_set = AccWindows(splits["valid"], W, require_label=False), AccWindows(splits["valid"], W)
x0, y0 = tr_set[0]
assert x0.shape == (3, W) and y0.shape == (W,) and set(y0.numpy().tolist()) <= set(range(7)) | {-1}
assert all((splits["train"][i][1][s:s + W] >= 0).any() for i, s in tr_set.index)
assert all((splits["valid"][i][1][s:s + W] >= 0).any() for i, s in va_set.index)
PERMS = signed_permutations()
assert perm_name(PERMS[0]) == "x,y,z", "candidate 0 must be the orientation the data comes in as"
CANDS = list(PERMS)                                             # what the search is allowed to try
IDX_OF = {perm_name(m): i for i, m in enumerate(CANDS)}
# The orientations the phone will be pretend-worn in: the identity (nothing turned), a spread of the
# axis-aligned turns the search can undo exactly, and continuous ones it cannot. Built by composing
# quarter turns about the phone's own axes rather than typed out, because three of the 48 obvious-looking
# names ("y,-x,-z" and friends) are mirror images with determinant -1, which no turning of a phone gives.
def quarter(M, k):
    out = np.eye(3, dtype=np.float32)
    for _ in range(k % 4):
        out = M @ out
    return out


RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)     # 90 deg about the phone's long edge
RY = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)     # ... about its top edge
RZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)     # ... about the axis out of the screen
WORN = [("worn as the lab put it on", np.eye(3, dtype=np.float32)),
        ("a quarter turn about the screen's normal", quarter(RZ, 1)),
        ("a half turn about the screen's normal", quarter(RZ, 2)),
        ("a half turn about the long edge", quarter(RX, 2)),
        ("a quarter turn about the long edge", quarter(RX, 1)),
        ("a quarter turn about the top edge", quarter(RY, 1)),
        ("a quarter turn about each of two edges", quarter(RZ, 1) @ quarter(RX, 1))]
STRESS_NAMES = [perm_name(m) for _, m in WORN]
assert len(set(STRESS_NAMES)) == len(WORN), "the pretend-worn orientations must differ from each other"
assert all(n in IDX_OF for n in STRESS_NAMES), "each must be a proper rotation the candidate set contains"
RANDOM_WORN = [random_rotation(np.random.default_rng(s)) for s in (101, 202, 303)]
STRESS = [m for _, m in WORN] + RANDOM_WORN
if A.stress:                                       # a short run: keep a few exact turns and one random one
    STRESS = STRESS[:max(1, A.stress - 1)] + STRESS[-1:]
N_AXIS = sum(1 for R in STRESS if perm_name(R) in IDX_OF)      # the turns a candidate can undo exactly


def worn_name(i):
    """The i-th pretend wearing by name. A continuous turn has none: perm_name of a dense matrix reads out
    whichever axis happens to be the first non-zero entry of each row, which is not an axis at all."""
    return perm_name(STRESS[i]) if i < N_AXIS else f"turned at random {i - N_AXIS + 1}"
one = [(0, 0)]
d_plain = AccWindows(splits["train"], W, index=one)[0][0]
d_straight = AccWindows(splits["train"], W, index=one, canon=True)[0][0]
d_turned = AccWindows(splits["train"], W, index=one, canon=True, rot=PERMS[7])[0][0]
# What canonicalizing does, checked rather than assumed: gravity lands on +x whichever way the phone was
# worn, so the signal ALONG gravity comes out the same; what it leaves free is the ROLL about gravity, so
# the y and z rows of a turned phone come out rotated about x. Rows 1,2 differing while their length agrees
# is exactly that roll - the textbook fix is only two thirds of a fix, which is worth a line in the report.
assert torch.allclose(d_straight[0], d_turned[0], atol=2e-4), "the along-gravity row must ignore the turn"
n_str = np.linalg.norm(d_straight[1:].numpy(), axis=0)
n_turn = np.linalg.norm(d_turned[1:].numpy(), axis=0)
assert np.allclose(n_str, n_turn, atol=2e-4), "the difference between the two must be a pure roll"
assert not torch.allclose(d_straight[1:], d_turned[1:], atol=1e-3), "... and that roll really is left free"
assert not torch.allclose(d_plain, d_turned, atol=1e-3), "while the plain arm must still see the turn"
out["canon_leaves_roll_free"] = float(np.mean(np.abs(d_straight[1:].numpy() - d_turned[1:].numpy())))
print(f"UNet(1, 3, 7, {WIDTHS}): {N_PAR / 1e3:.0f}k parameters; (batch, 3, {W}) in -> (batch, 7, {W}) "
      f"logits out, so every sample answers for itself\ncross-entropy with ignore_index=-1: a flat guess "
      f"costs log 7 = {np.log(7):.3f}, and a window in which every sample is a pause costs {BLANK} - not 0 "
      f"- so those windows are left out of training and validation, where they could teach nothing and a "
      f"single NaN step would have reset every weight: {len(all_tr) - len(tr_set)} of {len(all_tr)} "
      f"training and {len(all_va) - len(va_set)} of {len(all_va)} validation windows are the long pauses "
      f"before the first activity or after the last one")
print(f"{len(tr_set)} training windows on {len(splits['train'])} recordings, {len(va_set)} validation "
      f"windows, {sum(len(eval_windows(len(a), W)) for a, _, _ in splits['test'])} evaluation windows on "
      f"the test users - no overlap, no padding, the last window of a recording slid back so that every "
      f"sample is answered once, always from a full window of real signal")
print(f"the search chooses from {len(CANDS)} orientations; the phone will be pretend-worn in "
      f"{len(STRESS)} ({N_AXIS} axis-aligned, which a candidate can undo exactly, {len(STRESS) - N_AXIS} "
      f"continuous, which none can), and the same setting is chosen on the validation users only")


def evaluate(pred, y):
    """accuracy and macro / per-class F1 over the labelled samples only"""
    keep = np.asarray(y) >= 0
    return f1_scores(np.asarray(y)[keep], np.asarray(pred)[keep])


def run(model, rec_list, rot=None, canon=False, batch=512):
    """Answer every sample of every recording once, and note how well each window matched.

    Nothing in here reads the labels. Per window it stores the mean top-1 probability, the mean log top-1
    probability and the share of samples agreeing with the window's majority class - the last one being a
    second reading of "does this look like an activity at all", since a wrong orientation flickers.
    """
    model.eval()
    preds, stats, n_win, starts, keeps = [], [], [], [], []
    off = 0                                                             # where this recording starts in `pred`
    with torch.no_grad():
        for a, y, _ in rec_list:
            wins = eval_windows(len(a), W)
            n_win.append(len(wins))
            # window positions are sample positions in the concatenated pred, so they carry the offset too -
            # without it the second recording's chosen windows would overwrite the first recording's samples
            starts += [s + off for s, _ in wins]
            keeps += [k + off for _, k in wins]
            off += len(a)
            p = np.full(len(a), -1, np.int8)
            ds = AccWindows([(a, y)], W, index=[(0, s) for s, _ in wins], rot=rot, canon=canon)
            done = 0
            for xb, _ in DataLoader(ds, batch_size=batch):
                pr = torch.softmax(model(xb.to(part1.device)), 1)
                top, am = pr.max(1)
                major = (am == am.mode(1, keepdim=True).values).float().mean(1)
                negent = -(pr * torch.log(pr + 1e-12)).sum(1).mean(1)     # sum over the 7 classes, mean over W
                for j in range(len(xb)):
                    s, keep = wins[done + j]
                    # a window is responsible for its samples from `keep` on: for every window but the last
                    # keep == s and that is the whole window; for the last, slid back, it is only the tail
                    p[keep:s + W] = am[j].cpu().numpy().astype(np.int8)[keep - s:]
                    stats.append((float(top[j].mean()), float(torch.log(top[j] + 1e-12).mean()),
                                  float(major[j]), float(negent[j])))
                done += len(xb)
            assert done == len(wins) and (p >= 0).all(), "every sample of the recording answered once"
            preds.append(p)
    st = np.array(stats)
    return dict(pred=np.concatenate(preds),
                rec=np.repeat(np.arange(len(rec_list)), n_win),
                start=np.array(starts), keep=np.array(keeps),
                win_mean=st[:, 0], win_logp=st[:, 1], win_major=st[:, 2], win_negent=st[:, 3])


def search_group(got, margin, how="window", agg="prob"):
    """Pick an orientation per window (or per recording) from how well the network matches it.

    `got[i]` is the run where the network was shown CANDS[i] * (what the phone reported), so got[0] is
    the plain reading that trusts the axes as they came. A switch away from it has to beat it by
    `margin`, which is the "only if it matches better" of the task.
    """
    rec, start, keep = got[0]["rec"], got[0]["start"], got[0]["keep"]
    score = {"prob": "win_mean", "logprob": "win_logp", "major": "win_major"}[agg]
    sc = np.stack([g[score] for g in got], 0)                         # (n_candidates, n_windows)
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(rec) != 0, [True]]))
    chosen = np.zeros(len(rec), int)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if how == "recording":                                        # one orientation per recording
            s = sc[:, lo:hi].mean(1)
            win = int(s.argmax())
            chosen[lo:hi] = win if (win and s[win] > s[0] + margin) else 0
        else:                                                         # per window, as the task says
            s = sc[:, lo:hi]
            chosen[lo:hi] = np.where(s.max(0) > s[0] + margin, s.argmax(0), 0)
    pred = np.full(len(got[0]["pred"]), -1, np.int8)
    for j in range(len(got)):
        w = np.flatnonzero(chosen == j)
        if len(w):
            sel = np.concatenate([np.arange(keep[i], start[i] + W) for i in w])
            pred[sel] = got[j]["pred"][sel]
    if not (pred >= 0).all():
        miss = np.flatnonzero(pred < 0)
        raise AssertionError(f"{len(miss)} samples unanswered, first at {miss[:6].tolist()}; "
                             f"{len(rec)} windows over {rec.max() + 1} recordings, chosen candidate histogram "
                             f"{np.bincount(chosen, minlength=len(got)).tolist()}")
    return pred, chosen


def ang_between(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def group(model, rec_list, R, canon):
    """All the readings the search can take when the phone was worn in orientation R."""
    return [run(model, rec_list, rot=C @ R, canon=canon) for C in CANDS]


print("\n" + "=" * 100)
print("1. the baselines, before any network")
print("=" * 100)
b = evaluate(np.full(len(Y["test"]), 4), Y["test"])
print(f"always answering standing: accuracy {b['accuracy']:.3f}, macro F1 {b['macro_f1']:.3f}")


def centroid_rule(train_list, test_list):
    """One class per window: whichever training class-average feature vector is nearest. Nothing learned."""
    feats = lambda w: np.concatenate([w.mean(0), w.std(0)])                # noqa: E731
    acc, cnt = {}, {}
    for a, y, _ in train_list:
        for s in windows_of(len(a), W):
            u = set(y[s:s + W].tolist()) - {-1}
            if len(u) == 1:
                c = u.pop()
                f = feats(a[s:s + W])
                acc[c] = acc.get(c, 0) + f
                cnt[c] = cnt.get(c, 0) + 1
    cent = {c: acc[c] / cnt[c] for c in acc}
    pred = []
    for a, y, _ in test_list:
        p = np.full(len(a), -1, np.int8)
        for s, keep in eval_windows(len(a), W):
            f = feats(a[s:s + W])
            p[keep:s + W] = min(cent, key=lambda k: np.linalg.norm(cent[k] - f))
        pred.append(p)
    return np.concatenate(pred), cent, cnt


cp, cent, cnt = centroid_rule(splits["train"], splits["test"])
c = evaluate(cp, Y["test"])
print(f"nearest class average of (mean, sd) of each axis, one answer per {W}-sample window: accuracy "
      f"{c['accuracy']:.3f}, macro F1 {c['macro_f1']:.3f}; per class "
      + ", ".join(f"{CLASSES[k][:4]} {c['per_class'][k]['f1']:.2f}" for k in range(7)))
tran_runs = [sum(1 for _ in g) for k, g in itertools.groupby(Y["train"]) if k == 6]
mixed = [len(set(y[s:s + W].tolist()) - {-1}) > 1
         for a, y, _ in splits["test"] for s, _ in eval_windows(len(a), W)]
print(f"      the class averages come from " + ", ".join(f"{CLASSES[k][:4]} {cnt.get(k, 0)}" for k in range(7))
      + f" training windows that hold a single class. A transition runs {int(np.median(tran_runs))} samples "
      f"at the median against a {W}-sample window, so its average is built from the fewest windows of the "
      f"seven, and it is the answer a window that straddles a boundary - {np.mean(mixed):.1%} of the test "
      f"windows - drifts towards, because a boundary is the loudest thing in it")
print(f"-> standing is {np.mean(Y['test'][Y['test'] >= 0] == 4):.1%} of the labelled test samples, all the "
      f"first baseline has to offer; the second reads a whole window at once, the way the app will, and can "
      f"only ever give one answer for it - the network answers each sample of the window separately")
out["baselines"] = {"always_standing": b, "nearest_centroid_window": c,
                    "centroid_windows_per_class": {CLASSES[k]: int(cnt.get(k, 0)) for k in range(7)}}

print("\n" + "=" * 100)
print(f"2. the four arms ({EPOCHS} epochs, batch {A.batch}, lr {A.lr}, widths {WIDTHS})")
print("=" * 100)
results, models = {}, {}
for k in ARM_KEYS:
    cfg, tag = ARMS[k], ARMS[k]["name"]
    torch.manual_seed(0)
    model = part1.UNet(1, 3, 7, WIDTHS)
    t0 = time.perf_counter()
    model = part1.train(model, AccWindows(splits["train"], W, aug=cfg["aug"], canon=cfg["canon"]),
                        AccWindows(splits["valid"], W, canon=cfg["canon"]), loss_fn, EPOCHS,
                        lr=A.lr, batch_size=A.batch)
    secs = time.perf_counter() - t0
    plt.gcf().savefig(FIGS / f"curve_{tag}.png", dpi=110)
    plt.close("all")
    gv = run(model, splits["valid"], canon=cfg["canon"])
    rv = evaluate(gv["pred"], Y["valid"])
    results[k] = {"note": cfg["note"], "seconds": secs, "parameters": N_PAR, "validation": rv}
    models[k] = model
    torch.save({"state": model.state_dict(), "widths": WIDTHS, "window": W, "arm": k, "aug": cfg["aug"],
                "canon": cfg["canon"], "classes": CLASSES, "units": "g", "sample_rate": 50,
                "axes": ["x", "y", "z"]}, FIGS / f"model_{tag}.pt")
    print(f"  {tag:18s} validation accuracy {rv['accuracy']:.3f}, macro F1 {rv['macro_f1']:.3f}   per "
          f"class " + " ".join(f"{CLASSES[j][:4]} {rv['per_class'][j]['f1']:.2f}" for j in range(7))
          + f"   in {secs:.0f} s")
out["arms"] = results

print("\n" + "=" * 100)
print("3. what the search would choose - every arm read in all 24 orientations, scored on validation")
print("=" * 100)
VAL_STRESS = sorted({0, 1, min(4, len(STRESS) - 1), len(STRESS) - 2, len(STRESS) - 1})
MARGINS = {"prob": (0.0, 0.02, 0.05, 0.10), "logprob": (0.0, 0.2, 0.7, 1.5)}
print("the setting is chosen on the validation users, with the phone worn as: "
      + ", ".join(worn_name(i) for i in VAL_STRESS) + "  (the first is nothing turned)")
grid, plain_val, t0 = [], {}, time.perf_counter()
for k in ARM_KEYS:
    acc, plain = {}, []
    for i in VAL_STRESS:
        got = group(models[k], splits["valid"], STRESS[i], ARMS[k]["canon"])
        plain.append(evaluate(got[0]["pred"], Y["valid"])["macro_f1"])
        for how in ("window", "recording"):
            for agg in MARGINS:
                for margin in MARGINS[agg]:
                    p, _ = search_group(got, margin, how, agg)
                    acc.setdefault((how, agg, margin), []).append(evaluate(p, Y["valid"])["macro_f1"])
        del got                                             # one group at a time: 24 runs is enough in RAM
    grid += [{"arm": k, "how": h, "agg": a, "margin": m, "valid_macro_f1": float(np.mean(v)),
              "valid_macro_f1_by_rotation": [float(x) for x in v]} for (h, a, m), v in acc.items()]
    plain_val[ARMS[k]["name"]] = float(np.mean(plain))
    print(f"  {ARMS[k]['name']:18s} {len(acc)} settings; without any search {np.mean(plain):.3f}, best "
          f"with one {max(r['valid_macro_f1'] for r in grid if r['arm'] == k):.3f} macro F1")
print(f"{len(grid)} settings scored in {time.perf_counter() - t0:.0f} s, and not one test sample touched")
best = max(grid, key=lambda r: r["valid_macro_f1"])
out["search_grid"], out["search_chosen"] = grid, best
out["plain_validation_macro_f1"] = plain_val
print("the best few, all of them chosen without looking at the test users:")
for r in sorted(grid, key=lambda r: -r["valid_macro_f1"])[:6]:
    print(f"   {ARMS[r['arm']]['name']:18s} per {r['how']:9s} score {r['agg']:8s} margin {r['margin']:4.2f}"
          f" -> validation macro F1 {r['valid_macro_f1']:.3f}")
CA, HW, AG, MG = best["arm"], best["how"], best["agg"], best["margin"]

print("\n" + "=" * 100)
print("4. the test users, scored once")
print("=" * 100)
i_show = 1 if len(STRESS) > 1 else 0            # the first turned wearing of the phone, shown in full later
i_rand = len(STRESS) - 1                      # a continuous turn: no candidate can represent it
out["saved_rotations"] = {"as_it_comes": 0, "axis_aligned": i_show, "other": i_rand}
print(f"{'arm':18s} {'as it comes':>12s} {'macro F1':>9s} | {'worn as ' + worn_name(i_show):>15s} "
      f"{'mean over the worn set':>22s}")
final, robust, per_rot = {}, {}, {}
for k in ARM_KEYS:
    accs, f1s = [], []
    for i in range(len(STRESS)):
        # only the arm the search was chosen on needs all 24 readings here; the others are scored as they
        # come, which is the number the comparison in the table above is about
        got = (group(models[k], splits["test"], STRESS[i], ARMS[k]["canon"]) if k == CA else
               [run(models[k], splits["test"], rot=STRESS[i], canon=ARMS[k]["canon"])])
        r0 = evaluate(got[0]["pred"], Y["test"])
        accs.append(r0["accuracy"])
        f1s.append(r0["macro_f1"])
        if k == CA:
            p, cw = search_group(got, MG, HW, AG)
            s1 = evaluate(p, Y["test"])
            want = STRESS[i].T                               # the orientation that undoes the turn
            off = float(np.mean([ang_between(CANDS[c], want) for c in cw]))
            nm = worn_name(i)
            per_rot[nm] = {"as_is": r0["accuracy"], "as_is_macro_f1": r0["macro_f1"],
                           "searched": s1["accuracy"], "searched_macro_f1": s1["macro_f1"],
                           "gain": s1["accuracy"] - r0["accuracy"], "angle_left_off_deg": off,
                           "undone_to_under_20deg": float(np.mean([ang_between(CANDS[c], want) < 20 for c in cw])),
                           "switched_away_from_real": float(np.mean(cw != 0))}
            if i in (i_show, i_rand):
                np.savez_compressed(FIGS / f"preds_turned_{i}_test.npz", pred=got[0]["pred"], searched=p,
                                    chosen=cw, rot=STRESS[i], name=nm, rec=got[0]["rec"],
                                    start=got[0]["start"], keep=got[0]["keep"])
        if i == 0:                                    # every arm's own predictions, so the report can draw a
            first = r0                                # confusion matrix and a timeline for each of them
            np.savez_compressed(FIGS / f"preds_{ARMS[k]['name']}_test.npz", pred=got[0]["pred"],
                                rec=got[0]["rec"], start=got[0]["start"], keep=got[0]["keep"],
                                win_mean=got[0]["win_mean"], win_major=got[0]["win_major"],
                                **({} if k != CA else {"searched": p, "chosen": cw}))
        del got
    final[k] = {"accuracy": accs[0], "macro_f1": f1s[0], "by_worn_orientation": accs,
                "by_worn_macro_f1": f1s, "test": first}
    robust[k] = accs
    print(f"{ARMS[k]['name']:18s} {accs[0]:12.3f} {f1s[0]:9.3f} | {accs[i_show]:15.3f} {np.mean(accs):22.3f}")
out["final"] = final
out["robustness_as_is"] = {ARMS[k]["name"]: v for k, v in robust.items()}
out["per_rotation"] = per_rot

print(f"\nthe search as chosen ({ARMS[CA]['name']}, per {HW}, score {AG}, margin {MG}), on the test users:")
print(f"{'worn as':22s} {'plain':>8s} {'searched':>9s} {'gain':>7s} | {'orientation left off':>21s} "
      f"{'switched away':>13s}")
for nm, v in per_rot.items():
    print(f"{nm:22s} {v['as_is']:8.3f} {v['searched']:9.3f} {v['gain']:+7.3f} | "
          f"{v['angle_left_off_deg']:19.0f} deg {v['switched_away_from_real']:13.1%}")
ax_r = [v for nm, v in per_rot.items() if not nm.startswith("turned")]
rd_r = [v for nm, v in per_rot.items() if nm.startswith("turned")]
print(f"mean over the {len(ax_r)} axis-aligned wearings: {np.mean([v['as_is'] for v in ax_r]):.3f} plain, "
      f"{np.mean([v['searched'] for v in ax_r]):.3f} searched")
print(f"mean over the {len(rd_r)} continuous wearings no candidate can represent: "
      f"{np.mean([v['as_is'] for v in rd_r]):.3f} plain, {np.mean([v['searched'] for v in rd_r]):.3f} searched")
print(f"on the test users as they come, the search switched away from the real orientation in "
      f"{per_rot['x,y,z']['switched_away_from_real']:.1%} of the windows and moved accuracy "
      f"{per_rot['x,y,z']['gain']:+.3f} - that is the price of looking when nothing is turned")
out["chosen_arm"] = CA
(FIGS / "results_net.json").write_text(json.dumps(out, indent=1, default=float))
print(f"\nwrote model_*.pt, curve_*.png, preds_*_test.npz and results_net.json to {FIGS}")
print("for the phone, a model has to read the raw axes the app feeds it: "
      + ", ".join(f"{ARMS[k]['name']} {'does' if not ARMS[k]['canon'] else 'does NOT'}" for k in ARM_KEYS)
      + f" - the search was chosen on {ARMS[CA]['name']}, so whatever goes in the .onnx is a different "
      f"choice, made on validation among the arms that do")
