"""Task 1: a 1D U-Net that delineates P, QRS and T, trained on the repaired LUDB labels.

In the order the notebook asks for the work:

  1. the networks and train/predict of Part 1, copied WITHOUT any change (they are the ones the
     task says to reuse, so the comparison with every classmate's run and with the reference
     numbers means something);
  2. the Dataset: one record -> a (1, 5000) signal and (5000,) labels, where the samples
     repair_ludb.py marked -1 are skipped by both the loss and the metric;
  3. the choices (depth of the U-Net, augmentation, learning rate) - all made on the VALIDATION
     split, by the metric, not by the loss;
  4. the test split, evaluated once, next to the classical baseline from classical.py;
  5. "What to show": the Dice of every wave against the baseline's, one test record with the true
     and the predicted waves one above the other, and the QRS duration in ms of one test patient.

Every number is also written to figs/results.json and every figure to figs/, so the hand-in
reproduces from a fresh clone.

Usage:  py task1_ludb/train_ludb.py <masked_cache.npz> <figs-dir> [--quick]
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))           # our own modules, however this is run

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from classical import FS, CHOSEN, labels as baseline_labels
from ludb_metric import CLASSES, WAVES, evaluate, fmt, per_record

CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
EPOCHS = 6 if QUICK else 120        # train() keeps the weights of the best validation epoch
MS = 1000 / FS                      # one sample of a 500 Hz record is 2 ms
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")   # train() calls plt.show()

# -------------------------------------------------------------------------------------------
# 1. the networks and train/predict, imported verbatim from part1.py (a copy of Part 1 of
#    the notebook) - UNet, train and predict are used exactly as the task asks them to be.
# -------------------------------------------------------------------------------------------
from part1 import UNet, device, predict, train


# ---------------------------------------------------------------------------------------------
# 2. the data
# ---------------------------------------------------------------------------------------------
d = np.load(CACHE, allow_pickle=True)
x, waves, y_raw = d["x"], d["waves"], d["y_raw"]
y, y_tight, ig_signal = d["y"], d["y_tight"], d["ig_signal"]      # y carries -1 where nothing is annotated
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of, T = waves[:, 0], x.shape[1]
pos_in_test = {int(rr): i for i, rr in enumerate(splits["test"])}
# the four ways of deciding which samples count, shared with baseline_ludb.py (repair_ludb.py made them)
CONVENTIONS = {"span + buffer": y, "span, no buffer": y_tight, "signal edges": ig_signal, "as parsed": None}
PRIMARY = "span + buffer"


class LudbDataset(Dataset):
    """One record -> (1, 5000) signal and (5000,) class labels. Augmentation lives here, not in train.

    Augmented: circularly shifted by a random amount (the labels with it, so the seam is the
    record's own end meeting its own start), rescaled and given a little noise. LUDB is stored scaled
    to 1 mV peak-to-peak, so a gain change is the only amplitude variation the data can be asked
    about - and it stops the network latching on to absolute amplitudes.
    """

    def __init__(self, records, augment=False):
        self.x, self.y, self.augment = x[records], y[records], augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig, lab = self.x[i].astype(np.float32), self.y[i].astype(np.int64)
        if self.augment:
            shift = int(np.random.randint(len(sig)))
            sig, lab = np.roll(sig, shift), np.roll(lab, shift)
            sig = (sig * np.random.uniform(0.8, 1.25) + np.random.normal(0, 0.1, len(sig))).astype(np.float32)
        return torch.from_numpy(sig[None]), torch.from_numpy(lab)


def score_split(pred, which, conv=PRIMARY):
    """The metric of our specification table: macro Dice per wave, judged under one convention."""
    idx, m = splits[which], CONVENTIONS[conv]
    return evaluate(y_raw[idx], pred, None if m is None else m[idx])


def macro(res):
    return float(np.mean([res[WAVES[c]]["macro"] for c in CLASSES]))


def mean_record_dice(y_true, pred, idx):
    """Mean of the per-record Dice over the three waves, for ranking records."""
    out = []
    for i in range(len(idx)):
        vals = [v for c in CLASSES for v in [per_record(y_true[i:i + 1], pred[i:i + 1], c, y[idx][i:i + 1])[0]]
                if v is not None]
        out.append(np.mean(vals) if vals else 1.0)
    return np.array(out)


torch.manual_seed(0)
np.random.seed(0)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)      # the -1 of repair_ludb.py, exactly like example 7
train_set, valid_set = LudbDataset(splits["train"]), LudbDataset(splits["valid"])
sig0, lab0 = train_set[0]
print(f"device {device} | torch {torch.__version__}")
print(f"one training sample: signal {tuple(sig0.shape)} {sig0.dtype}, labels {tuple(lab0.shape)} "
      f"{lab0.dtype}, classes present {sorted(set(lab0.numpy().tolist()))}")
print(f"{len(train_set)} training records, {len(train_set) // 16 + 1} batches per epoch; samples the loss "
      f"skips: {(y[splits['train']] == -1).mean():.1%}")

# ---------------------------------------------------------------------------------------------
# 3. the choices, all of them on validation
# ---------------------------------------------------------------------------------------------
# Depth is the interesting knob: a block halves the resolution and its two 3-wide convolutions add
# +-2^level samples to what one output position can see, so 4 levels give a receptive field of about
# 60 samples, 6 levels about 250 and 7 levels about 500 - a whole R-R interval, which is what telling
# a P wave from the tail of the previous T wave needs.
WIDTHS = {"4 levels": [16, 32, 64, 128], "5 levels": [16, 32, 64, 128, 256],
          "6 levels": [16, 32, 64, 128, 256, 512], "7 levels": [8, 16, 32, 64, 128, 256, 512],
          "7 wide": [16, 32, 64, 128, 256, 512, 1024]}

base_valid, _, _ = baseline_labels(x[splits["valid"]], waves, rec_of, splits["train"], CHOSEN)
base_val_res = score_split(base_valid, "valid")
print(f"\nthe bar to clear on validation: classical baseline {fmt(base_val_res)} = {macro(base_val_res):.3f}")

sweep = []
for name, widths in WIDTHS.items():
    for augment in ([False] if QUICK else [False, True]):
        torch.manual_seed(0)
        np.random.seed(0)
        net = UNet(1, 1, 4, widths)
        print(f"\n{name}, width {widths}, augment {augment}, "
              f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters")
        net = train(net, LudbDataset(splits["train"], augment), valid_set, loss_fn, EPOCHS)
        plt.savefig(FIGS / f"curve_{name.replace(' ', '')}_aug{int(augment)}.png", dpi=110)
        plt.close("all")
        res = score_split(predict(net, valid_set)[0].argmax(1).numpy(), "valid")
        sweep.append((macro(res), name, widths, augment, 1e-3, net, res))

sweep.sort(key=lambda t: -t[0])
print("\nvalidation sweep, judged under the reported convention (the winner is the model we keep):")
print(f"  {'depth':10s} {'augment':8s} {'lr':7s} {'parameters':12s} validation macro Dice P/QRS/T   mean")
for m, name, widths, augment, lr, net, res in sweep:
    print(f"  {name:10s} {str(augment):8s} {lr:<7.0e} "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:8.1f} M   {fmt(res)}   {m:.3f}")

best_macro, _, best_widths, best_aug, _, best_net, best_valid = sweep[0]
lr_results = [(best_macro, 1e-3, best_valid, best_net)]
if not QUICK:
    print(f"\nlearning rate for the winning depth ({best_widths}), still on validation:")
    for lr in (3e-4, 3e-3, 1e-2):
        torch.manual_seed(0)
        np.random.seed(0)
        net = UNet(1, 1, 4, best_widths)
        net = train(net, LudbDataset(splits["train"], best_aug), valid_set, loss_fn, EPOCHS, lr=lr)
        plt.close("all")
        res = score_split(predict(net, valid_set)[0].argmax(1).numpy(), "valid")
        lr_results.append((macro(res), lr, res, net))
        print(f"  lr {lr:<7.0e} {fmt(res)}   {macro(res):.3f}")
    lr_results.sort(key=lambda t: -t[0])
best_lr, best_valid, model = lr_results[0][1], lr_results[0][2], lr_results[0][3]
print(f"\nchosen: UNet(1, 1, 4, {best_widths}), augment={best_aug}, lr={best_lr:.0e}, {EPOCHS} epochs with the "
      f"best validation epoch kept -> validation {fmt(best_valid)} = {macro(best_valid):.3f}, "
      f"baseline {macro(base_val_res):.3f}")

# ---------------------------------------------------------------------------------------------
# 4. the test split, once
# ---------------------------------------------------------------------------------------------
net_pred = predict(model, LudbDataset(splits["test"]))[0].argmax(1).numpy()
base_pred, _, _ = baseline_labels(x[splits["test"]], waves, rec_of, splits["train"], CHOSEN)
print(f"\nTEST split ({len(splits['test'])} patients): macro Dice of every wave, network next to baseline")
table = {}
for conv in CONVENTIONS:
    net_res, base_res = score_split(net_pred, "test", conv), score_split(base_pred, "test", conv)
    table[conv] = {"network": net_res, "baseline": base_res}
    print(f"  judged as '{conv}'" + ("   <- the convention we report" if conv == PRIMARY else ""))
    for c in CLASSES:
        n, b = net_res[WAVES[c]], base_res[WAVES[c]]
        print(f"    {WAVES[c]:4s} network {n['macro']:.3f} (micro {n['micro']:.3f}, {n['n_records']} records)   "
              f"baseline {b['macro']:.3f} (micro {b['micro']:.3f})   {n['macro'] - b['macro']:+.3f}")
    print(f"    mean  network {macro(net_res):.3f} vs baseline {macro(base_res):.3f}")
print(f"  sanity, a network that answers background everywhere: {fmt(score_split(np.zeros_like(net_pred), 'test'))}")

# ---------------------------------------------------------------------------------------------
# 5. what to show
# ---------------------------------------------------------------------------------------------
COLORS = {1: "tab:green", 2: "tab:red", 3: "tab:blue"}


def wave_panel(ax, arr, sig, ylabel):
    """Signal with a class-per-sample answer filled in; grey where nothing is annotated."""
    t = np.arange(T) / FS
    ax.plot(t, sig, "k", linewidth=0.5)
    for c in CLASSES:
        ax.fill_between(t, -4.5, 4.5, where=arr == c, color=COLORS[c], alpha=0.4, linewidth=0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_ylabel(ylabel, fontsize=9)


def record_dice(rr, pred_row):
    return ", ".join(f"{WAVES[c]} " + (f"{v:.2f}" if (v := per_record(y_raw[[rr]], pred_row, c, y[[rr]])[0])
                                       is not None else "n/a") for c in CLASSES)


r = int(splits["test"][0])
i = pos_in_test[r]
fig, axes = plt.subplots(3, 1, figsize=(13, 5.6), sharex=True)
for ax, arr, lab in ((axes[0], y_raw[r], "annotated"), (axes[1], base_pred[i], "baseline"),
                     (axes[2], net_pred[i], "network")):
    ax.fill_between(np.arange(T) / FS, -4.5, 4.5, where=y[r] == -1, color="0.8", linewidth=0)
    wave_panel(ax, arr, x[r], lab)
axes[2].set_xlabel("time (s)")
fig.suptitle(f"test record {r}: annotated (top), classical baseline (middle), U-Net (bottom); "
             f"green P, red QRS, blue T, grey ignored by the metric | this record's Dice "
             f"{record_dice(r, net_pred[i:i + 1])}", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "one_test_record.png", dpi=130)


def runs_of_qrs(row):
    out, start = [], None
    for j, v in enumerate(list(row) + [0]):
        if v == 2 and start is None:
            start = j
        elif v != 2 and start is not None:
            out.append((start, j - 1))
            start = None
    return out


def durations(rr, pred_row):
    """(annotated ms, predicted ms) for every annotated QRS of record rr that the answer found."""
    ann = [(int(o), int(f)) for _, c, o, _, f in waves[(rec_of == rr) & (waves[:, 1] == 2)]]
    pairs = []
    for a, b in ann:
        over = [(p, q) for p, q in runs_of_qrs(pred_row) if p <= b and q >= a]
        if over:
            p, q = max(over, key=lambda pq: min(b, pq[1]) - max(a, pq[0]))
            pairs.append(((b - a + 1) * MS, (q - p + 1) * MS))
    return pairs


pairs = durations(r, net_pred[i])
all_pairs = [pr for j, rr in enumerate(splits["test"]) for pr in durations(int(rr), net_pred[j])]
a_all, p_all = np.array(all_pairs).T
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
axes[0].plot(range(1, len(pairs) + 1), [a for a, _ in pairs], "o-", label="annotated")
axes[0].plot(range(1, len(pairs) + 1), [q for _, q in pairs], "s-", label="predicted")
axes[0].set_xlabel("beat")
axes[0].set_ylabel("QRS duration (ms)")
axes[0].set_title(f"test record {r}: {len(pairs)} beats matched", fontsize=9)
axes[0].legend(fontsize=8)
lo, hi = min(a_all.min(), p_all.min()) - 5, max(a_all.max(), p_all.max()) + 5
axes[1].scatter(a_all, p_all, s=12, alpha=0.5)
axes[1].plot([lo, hi], [lo, hi], "k", linewidth=0.8)
axes[1].set_xlim(lo, hi)
axes[1].set_ylim(lo, hi)
axes[1].set_xlabel("annotated QRS duration (ms)")
axes[1].set_ylabel("predicted (ms)")
axes[1].set_title(f"every test patient, {len(a_all)} beats: mean {a_all.mean():.0f} vs {p_all.mean():.0f} ms, "
                  f"mean error {np.abs(a_all - p_all).mean():.0f} ms", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "qrs_duration.png", dpi=130)

worst = np.argsort(mean_record_dice(y_raw[splits["test"]], net_pred, splits["test"]))[:3]
fig, axes = plt.subplots(len(worst), 1, figsize=(13, 2.8 * len(worst)))
for ax, j in zip(np.atleast_1d(axes), worst):
    rr = int(splits["test"][j])
    ax.fill_between(np.arange(T) / FS, -4.5, 4.5, where=y[rr] == -1, color="0.8", linewidth=0)
    t = np.arange(T) / FS
    ax.plot(t, x[rr], "k", linewidth=0.5, label="signal")
    for c in CLASSES:
        ax.fill_between(t, -4.5, 4.5, where=y_raw[rr] == c, color=COLORS[c], alpha=0.18, linewidth=0)
        ax.fill_between(t, -4.5, 4.5, where=net_pred[j] == c, color=COLORS[c], alpha=0.45, linewidth=0,
                        step="mid")
    ax.set_ylim(-4.5, 4.5)
    ax.set_ylabel(f"record {rr}\nDice {mean_record_dice(y_raw[[rr]], net_pred[j:j + 1], [rr])[0]:.2f}",
                  fontsize=8)
fig.suptitle("the three test records the network delineates worst: pale fill = annotated, solid = predicted",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "worst_records.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, c in zip(axes, CLASSES):
    bv = [v for v in per_record(y_raw[splits["test"]], base_pred, c, y[splits["test"]]) if v is not None]
    nv = [v for v in per_record(y_raw[splits["test"]], net_pred, c, y[splits["test"]]) if v is not None]
    ax.boxplot([bv, nv], tick_labels=["baseline", "network"], showfliers=False)
    ax.set_ylabel(f"{WAVES[c]} Dice, one value per test record")
    ax.set_title(f"median {np.median(bv):.2f} -> {np.median(nv):.2f}", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "dice_per_record.png", dpi=130)

torch.save(model.state_dict(), FIGS.parent / "model_ludb.pt")
results = {"chosen": {"widths": best_widths, "augment": bool(best_aug), "lr": best_lr, "epochs": EPOCHS,
                      "parameters": int(sum(p.numel() for p in model.parameters()))},
           "validation": {"network": {WAVES[c]: best_valid[WAVES[c]]["macro"] for c in CLASSES},
                          "baseline": {WAVES[c]: base_val_res[WAVES[c]]["macro"] for c in CLASSES}},
           "test": {conv: {who: {WAVES[c]: table[conv][who][WAVES[c]] for c in CLASSES}
                           for who in ("network", "baseline")} for conv in CONVENTIONS},
           "qrs_duration_ms": {"annotated_mean": float(a_all.mean()), "predicted_mean": float(p_all.mean()),
                               "mean_absolute_error": float(np.abs(a_all - p_all).mean()),
                               "beats_matched": int(len(a_all))},
           "baseline_config": {"kind": str(CHOSEN["kind"]), "frac": CHOSEN["frac"],
                               "distance": CHOSEN["distance"], "sigma": CHOSEN["sigma"]},
           "worst_records": [int(splits["test"][j]) for j in worst]}
(FIGS / "results.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {FIGS / 'results.json'}, one_test_record.png, qrs_duration.png, worst_records.png, "
      f"dice_per_record.png and model_ludb.pt")
