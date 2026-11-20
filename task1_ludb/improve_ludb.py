"""Four attempts at beating the lead-II U-Net that train_ludb.py chose, all decided on validation.

The architecture, width, learning rate, augmentation and the 120-epoch schedule of train_ludb.py are
kept exactly as they were: they were already chosen on these same 30 validation records, and choosing
them a second time would be tuning twice on one small split. What varies here is four things that
were never varied at all:

  * which leads the network sees. The cardiologists annotated lead II, which is why the notebook
    hands the network lead II - but a P wave is often largest in V1 or in V5, and one lead has to
    throw that information away. Lead II stays the reference result because that is the task as set;
    the multi-lead numbers are reported next to it, not instead of it.
  * the loss. CrossEntropy weights every sample alike while the metric is Dice per wave and the P
    class is a tenth of the kept samples. Two fixes, tried separately and combined: class weights
    (1/sqrt of the class frequency) and a soft-Dice term added to the CE, which is the metric itself
    made differentiable.
  * the seed. Two runs of the identical recipe disagreed by 0.03 of validation P on the depth sweep
    (logs/train_full.log against logs/train_confirm.log), which is more than most of the arms below
    are apart by. cuDNN is put in deterministic mode and the arms that survive round 2 are run with
    three seeds and compared by their mean.
  * what happens at inference: the softmax averaged over five circular shifts of the record
    (test-time augmentation), the votes of three independently trained seeds, and predicted runs
    shorter than a couple of samples merged into their longer neighbour. None of these trains
    anything, so all nine combinations are tried on validation for free and only the winner is
    carried to the test split.

Rounds: 1 - leads x loss, seed 0. 2 - class weights added to the two best. 3 - seeds 1 and 2 for the
two best, so that the final choice is made between two arms that each have three seeds (comparing a
three-seed mean against a one-seed result would not be a comparison). Then inference, then test once.

Usage:  py task1_ludb/improve_ludb.py <masked_cache.npz> <figs-dir> [--quick]
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from classical import FS, CHOSEN, labels as baseline_labels
from ludb_metric import CLASSES, WAVES, evaluate, fmt, per_record, qrs_durations, runs_of

CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
EPOCHS = 6 if QUICK else 120
SEEDS = (0,) if QUICK else (0, 1, 2)
MS = 1000 / FS
TTA_SHIFTS = (0, 90, 180, 270, 360)      # a fifth of the median R-R interval (450 samples) apart
TIE = 0.005                              # closer than this on 30 records is a tie, see choose()
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# the two runs of the same recipe that disagreed by 0.03 were not reproducible run to run; this is
# the part of that which cuDNN is responsible for. It does not remove the seed-to-seed spread, it
# stops the same seed from moving, so the three seeds measure the spread instead of adding to it.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from part1 import UNet, device, predict, train      # noqa: E402  (after matplotlib is configured)

# ---------------------------------------------------------------------------------------------
# the data, and the recipe inherited from train_ludb.py
# ---------------------------------------------------------------------------------------------
d = np.load(CACHE, allow_pickle=True)
x, x12, waves, y_raw, y = d["x"], d["x12"], d["waves"], d["y_raw"], d["y"]
y_tight, ig_signal = d["y_tight"], d["ig_signal"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of, T = waves[:, 0], x.shape[1]
pos_in_test = {int(rr): i for i, rr in enumerate(splits["test"])}
CONVENTIONS = {"span + buffer": y, "span, no buffer": y_tight, "signal edges": ig_signal, "as parsed": None}
PRIMARY = "span + buffer"
LEADS = [str(s) for s in d["lead_names"]]
LEAD_SETS = {"lead II": [LEADS.index("II")],
             "II+V1+V5": [LEADS.index(n) for n in ("II", "V1", "V5")],
             "all 12": list(range(len(LEADS)))}

chosen = json.loads((FIGS / "results.json").read_text())["chosen"]
WIDTHS, LR, AUG = chosen["widths"], chosen["lr"], chosen["augment"]


def signals(records, lead_cols):
    """(n, n_leads, 5000) float32, contiguous, for feeding outside a Dataset (test-time shifts)."""
    return np.ascontiguousarray(x12[np.ix_(np.asarray(records), lead_cols, np.arange(T))], dtype=np.float32)


class Records(Dataset):
    """The same records and the same augmentation as train_ludb.py, with the leads picked by index.

    One difference on purpose: the gain and the noise are drawn per lead instead of once per record,
    because the leads are standardized one by one and so each has its own scale. With a single lead
    the two are the same draw, which keeps the lead II arm comparable to train_ludb.py.
    """

    def __init__(self, records, lead_cols, augment):
        self.x, self.y = signals(records, lead_cols), y[np.asarray(records)]
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        sig, lab = self.x[i], self.y[i].astype(np.int64)
        if self.augment:
            shift = int(np.random.randint(sig.shape[1]))
            sig, lab = np.roll(sig, shift, axis=1), np.roll(lab, shift)
            sig = (sig * np.random.uniform(0.8, 1.25, size=(sig.shape[0], 1))
                   + np.random.normal(0, 0.1, sig.shape)).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(sig, dtype=np.float32)), torch.from_numpy(lab)


def score(pred, which, conv=PRIMARY):
    idx, m = splits[which], CONVENTIONS[conv]
    return evaluate(y_raw[idx], pred, None if m is None else m[idx])


def macro(res):
    return float(np.mean([res[WAVES[c]]["macro"] for c in CLASSES]))


def soft_dice_loss(logits, target, eps=1e-6):
    """1 - mean soft Dice over P, QRS and T: the metric we are graded on, differentiated.

    Samples labelled -1 leave both the numerator and the denominator, so an ignored beat cannot
    contribute to the score either way. Background is left out of the mean: it is over half of the
    signal and including it would let the network buy Dice by predicting nothing.
    """
    keep = (target >= 0).float()[:, None]
    prob = torch.softmax(logits, 1) * keep
    onehot = F.one_hot(target.clamp(min=0), 4).permute(0, 2, 1).float() * keep
    inter = (prob * onehot).sum((0, 2))
    dice = (2 * inter + eps) / (prob.sum((0, 2)) + onehot.sum((0, 2)) + eps)
    return 1 - dice[1:].mean()


kept_train = int((y[splits["train"]] >= 0).sum())                    # the samples the loss sees at all
freq = np.array([(y[splits["train"]] == c).sum() for c in range(4)], float) / kept_train
w = 1 / np.sqrt(freq)
WEIGHT = torch.tensor(w / w.mean(), dtype=torch.float32)              # mean 1, so the loss scale is kept


def make_loss(name):
    """CrossEntropy, optionally weighted, optionally with a soft-Dice term. The name says which."""
    weight = WEIGHT if "weights" in name else None
    return Loss(weight, 1.0 if "dice" in name else 0.0)


class Loss(nn.Module):
    def __init__(self, weight=None, dice=0.0):
        super().__init__()
        self.register_buffer("w", torch.ones(4) if weight is None else weight)
        self.dice = dice

    def forward(self, logits, target):
        loss = F.cross_entropy(logits, target, weight=self.w.to(logits.device), ignore_index=-1)
        return loss + self.dice * soft_dice_loss(logits, target) if self.dice else loss


ARMS = {}


def train_arm(lead_name, loss_name, seed):
    """One arm of the study: the inherited recipe, these leads, this loss, this seed."""
    leads = LEAD_SETS[lead_name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = train(UNet(1, len(leads), 4, WIDTHS), Records(splits["train"], leads, AUG),
                Records(splits["valid"], leads, False), make_loss(loss_name), EPOCHS, lr=LR)
    plt.close("all")
    res = score(predict(net, Records(splits["valid"], leads, False))[0].argmax(1).numpy(), "valid")
    ARMS.setdefault((lead_name, loss_name), {"nets": [], "res": []})
    ARMS[(lead_name, loss_name)]["nets"].append(net)
    ARMS[(lead_name, loss_name)]["res"].append(res)
    print(f"  {lead_name:9s} {loss_name:20s} seed {seed}   {fmt(res)}   {macro(res):.3f}")
    return net, res


print(f"device {device} | torch {torch.__version__} | cudnn deterministic "
      f"{torch.backends.cudnn.deterministic}")
print(f"inherited recipe: UNet(1, n_leads, 4, {WIDTHS}), lr {LR:.0e}, augment {AUG}, {EPOCHS} epochs, "
      f"seeds {SEEDS}")
print(f"leads available {LEADS}")
print(f"  lead sets tried: " + ", ".join(f"{k} {[LEADS[i] for i in v]}" for k, v in LEAD_SETS.items()))
print(f"class fractions among the {kept_train} trained-on samples {np.round(freq, 3)} -> loss weights "
      f"{np.round(WEIGHT.numpy(), 3)}")

base_valid, _, _ = baseline_labels(x[splits["valid"]], waves, rec_of, splits["train"], CHOSEN)
base_val_res = score(base_valid, "valid")
# the same baseline train_ludb.py measured on these records; if this drifts, the two scripts are no
# longer describing the same data or the same detector, and every comparison below would be hollow
recorded_val = json.loads((FIGS / "results.json").read_text())["validation"]["baseline"]
assert all(abs(base_val_res[WAVES[c]]["macro"] - recorded_val[WAVES[c]]) < 2e-3 for c in CLASSES), \
    (fmt(base_val_res), recorded_val)
print(f"the bar to clear on validation: classical baseline {fmt(base_val_res)} = {macro(base_val_res):.3f}")

# ---------------------------------------------------------------------------------------------
# round 1: leads x loss
# ---------------------------------------------------------------------------------------------
print("\nround 1, which leads and which loss, seed 0:")
for lead_name in LEAD_SETS:
    for loss_name in ("CE", "CE + dice"):
        train_arm(lead_name, loss_name, 0)


def ranked_arms():
    return sorted(ARMS, key=lambda k: -np.mean([macro(r) for r in ARMS[k]["res"]]))


def table(title, keys):
    print(f"\n{title}")
    print(f"  {'leads':9s} {'loss':20s} {'seeds':6s} {'mean':6s}  sd      validation P/QRS/T of its best seed")
    for k in keys:
        m = np.array([macro(r) for r in ARMS[k]["res"]])
        print(f"  {k[0]:9s} {k[1]:20s} {len(m):<6d} {m.mean():.3f}   {m.std():.3f}   "
              f"{fmt(ARMS[k]['res'][int(np.argmax(m))])}")


table("round 1 ranked:", ranked_arms()[:4])

# ---------------------------------------------------------------------------------------------
# round 2: add the class weights to the two best arms
# ---------------------------------------------------------------------------------------------
print("\nround 2, class weights added to the two best arms, seed 0:")
for k in ranked_arms()[:2]:
    with_w = (k[0], k[1] + " + weights")
    if with_w not in ARMS:
        train_arm(*with_w, 0)
table("round 2 ranked:", ranked_arms()[:4])

# ---------------------------------------------------------------------------------------------
# round 3: three seeds for the two arms still in contention
# ---------------------------------------------------------------------------------------------
if len(SEEDS) > 1:
    print(f"\nround 3, seeds {SEEDS[1:]} for the two best arms - this is the noise floor of the whole "
          f"comparison:")
    for k in ranked_arms()[:2]:
        for seed in SEEDS[len(ARMS[k]['res']):]:
            train_arm(*k, seed)

complete = [k for k in ranked_arms() if len(ARMS[k]["res"]) == len(SEEDS)] or ranked_arms()[:2]
table(f"the two arms with all {len(SEEDS)} seeds - the final choice is made between these:", complete)


def choose(keys):
    """Best mean over the seeds; inside TIE, the arm that deviates least from the task as set.

    'deviates least' means fewer leads (the notebook defines the input as lead II) and, failing that,
    plain cross-entropy. A 0.005 gap between two 30-record validation means is not worth buying an
    input the task did not ask for.
    """
    best = max(np.mean([macro(r) for r in ARMS[k]["res"]]) for k in keys)
    inside = [k for k in keys if best - np.mean([macro(r) for r in ARMS[k]["res"]]) <= TIE]
    return min(inside, key=lambda k: (len(LEAD_SETS[k[0]]), "dice" in k[1], "weights" in k[1]))


best_cfg = choose(complete)
best_nets, best_res = ARMS[best_cfg]["nets"], ARMS[best_cfg]["res"]
seed_macros = [macro(r) for r in best_res]
print(f"\nchosen arm: {best_cfg[0]} [{', '.join(LEADS[i] for i in LEAD_SETS[best_cfg[0]])}], "
      f"loss {best_cfg[1]} -> validation {fmt(ARMS[best_cfg]['res'][int(np.argmax(seed_macros))])}, "
      f"mean over seeds {np.mean(seed_macros):.3f} +- {np.std(seed_macros):.3f} "
      f"(baseline {macro(base_val_res):.3f})")

# ---------------------------------------------------------------------------------------------
# inference-time options, tried on validation without training anything
# ---------------------------------------------------------------------------------------------
def drop_short_runs(pred, min_len):
    """Predicted runs shorter than min_len samples take the label of their longer neighbour.

    A one-sample island of P in the middle of the T wave costs Dice twice - it is a false positive
    where it is and it does not help the run it should have joined. Decided on validation, applied to
    the predicted labels only, never to the probabilities' order.
    """
    out = pred.copy()
    for row, new in zip(pred, out):
        cut = np.flatnonzero(np.diff(row)) + 1                        # where the label changes
        bounds = [(int(a), int(b) - 1) for a, b in zip([0, *cut], [*cut, len(row)])]
        for i, (a, b) in enumerate(bounds):
            if b - a + 1 >= min_len:
                continue
            left, right = (bounds[i - 1] if i else None), (bounds[i + 1] if i + 1 < len(bounds) else None)
            if left and right:
                pick = row[left[0]] if left[1] - left[0] >= right[1] - right[0] else row[right[0]]
            else:
                pick = row[left[0]] if left else (row[right[0]] if right else 0)
            new[a:b + 1] = pick
    return out


@torch.no_grad()
def predict_shifted(models, sig, shifts):
    """Mean softmax of a record seen at several circular shifts, each rolled back before averaging.

    part1's `predict` is used unchanged everywhere else; this exists only because test-time
    augmentation has to show the model the same record at several offsets. The shifts are circular,
    the same kind the training augmentation already uses, so the seam where the end of the record
    meets its own start is not a new situation for the network. Averaging the probabilities rather
    than the argmax votes is what lets a shift that is unsure simply not vote.
    """
    B, _, T_ = sig.shape
    prob = torch.zeros(B, 4, T_, device=device)
    for model in models:
        model.eval()
        for s in shifts:
            for i in range(0, B, 32):
                blk = torch.from_numpy(sig[i:i + 32]).to(device)
                p = torch.softmax(model(torch.roll(blk, s, dims=2).float()), dim=1)
                prob[i:i + 32] += torch.roll(p, -s, dims=2)
    return (prob / (len(models) * len(shifts))).argmax(1).cpu().numpy()


print("\ninference-time options, validation, nothing retrained:")
sig_valid = signals(splits["valid"], LEAD_SETS[best_cfg[0]])
grid = []
for n_models in dict.fromkeys((1, len(best_nets))):          # dict.fromkeys: one row per distinct choice
    for shifts in ((0,), TTA_SHIFTS):
        for k in (0, 5, 10):
            pred = predict_shifted(best_nets[:n_models], sig_valid, shifts)
            if k:
                pred = drop_short_runs(pred, k)
            res = score(pred, "valid")
            grid.append((macro(res), n_models, shifts, k, res))
            print(f"  {n_models} model(s), {len(shifts)} shift(s), merge runs < {k:2d} samples "
                  f"({k * MS:.0f} ms): {fmt(res)}   {macro(res):.3f}")
grid.sort(key=lambda t: -t[0])
_, n_models, shifts, merge_k, tta_val_res = grid[0]
print(f"  chosen at inference: {n_models} model(s), {len(shifts)} shift(s), merge below {merge_k} samples")

# ---------------------------------------------------------------------------------------------
# the test split, once
# ---------------------------------------------------------------------------------------------
idx_test = splits["test"]
sig_test = signals(idx_test, LEAD_SETS[best_cfg[0]])


def final_pred(models, shifts=(0,), k=0):
    p = predict_shifted(models, sig_test, shifts)
    return drop_short_runs(p, k) if k else p


pred_final = final_pred(best_nets[:n_models], shifts, merge_k)
pred_single = final_pred([best_nets[0]])                      # directly comparable to train_ludb.py
per_seed = [macro(score(p, "test")) for p in
            (final_pred([net]) for net in best_nets)]
base_pred, _, _ = baseline_labels(x[idx_test], waves, rec_of, splits["train"], CHOSEN)

# the model train_ludb.py saved, run again here so that the two figures can be drawn on the same
# record and so that the recorded primary result is checked rather than trusted
primary = UNet(1, 1, 4, WIDTHS).to(device)
primary.load_state_dict(torch.load(FIGS.parent / "model_ludb.pt", map_location=device))
pred_primary = predict(primary, Records(idx_test, LEAD_SETS["lead II"], False))[0].argmax(1).numpy()
recorded = json.loads((FIGS / "results.json").read_text())["test"][PRIMARY]["network"]
got = {WAVES[c]: score(pred_primary, "test")[WAVES[c]]["macro"] for c in CLASSES}
print(f"\nthe saved lead-II model reproduces figs/results.json: "
      + ", ".join(f"{WAVES[c]} {got[WAVES[c]]:.3f} vs {recorded[WAVES[c]]['macro']:.3f}" for c in CLASSES))
assert all(abs(got[WAVES[c]] - recorded[WAVES[c]]["macro"]) < 2e-3 for c in CLASSES), got

print(f"\nTEST split ({len(idx_test)} patients), macro Dice, network next to the classical baseline")
table_test = {}
for conv in CONVENTIONS:
    net_res, base_res = score(pred_final, "test", conv), score(base_pred, "test", conv)
    table_test[conv] = {"network": net_res, "baseline": base_res}
    print(f"  judged as '{conv}'" + ("   <- the convention we report" if conv == PRIMARY else ""))
    for c in CLASSES:
        n, b = net_res[WAVES[c]], base_res[WAVES[c]]
        print(f"    {WAVES[c]:4s} network {n['macro']:.3f} (micro {n['micro']:.3f}, {n['n_records']} records)"
              f"   baseline {b['macro']:.3f} (micro {b['micro']:.3f})   {n['macro'] - b['macro']:+.3f}")
    print(f"    mean  network {macro(net_res):.3f} vs baseline {macro(base_res):.3f}")

print(f"\n  under '{PRIMARY}', for context, on the same test records:")
for label, p in (("  lead II, the primary model of train_ludb.py", pred_primary),
                 ("  this arm, one seed, no test-time tricks", pred_single)):
    r = score(p, "test")
    print(f"  {label}: {fmt(r)}   mean {macro(r):.3f}")
print(f"  this arm, the {len(best_nets)} seeds one at a time: mean {np.mean(per_seed):.3f} "
      f"+- {np.std(per_seed):.3f}, individually " + ", ".join(f"{m:.3f}" for m in per_seed))
print(f"  this arm, final ({n_models} model(s), {len(shifts)} shift(s), merge < {merge_k} samples): "
      f"{fmt(score(pred_final, 'test'))}   mean {macro(score(pred_final, 'test')):.3f}")
print(f"  sanity, a network that answers background everywhere: "
      f"{fmt(score(np.zeros_like(pred_final), 'test'))}")

# ---------------------------------------------------------------------------------------------
# what to show, for the final model
# ---------------------------------------------------------------------------------------------
COLORS = {1: "tab:green", 2: "tab:red", 3: "tab:blue"}
r = int(idx_test[0])
i = pos_in_test[r]


def wave_panel(ax, arr, sig_row, ylabel):
    t = np.arange(T) / FS
    ax.plot(t, sig_row, "k", linewidth=0.5)
    for c in CLASSES:
        ax.fill_between(t, -4.5, 4.5, where=arr == c, color=COLORS[c], alpha=0.4, linewidth=0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_ylabel(ylabel, fontsize=9)


fig, axes = plt.subplots(4, 1, figsize=(13, 7.2), sharex=True)
for ax, arr, lab in ((axes[0], y_raw[r], "annotated"), (axes[1], base_pred[i], "baseline"),
                     (axes[2], pred_primary[i], "U-Net, lead II"), (axes[3], pred_final[i], "U-Net, final")):
    ax.fill_between(np.arange(T) / FS, -4.5, 4.5, where=y[r] == -1, color="0.8", linewidth=0)
    wave_panel(ax, arr, x[r], lab)
axes[3].set_xlabel("time (s)")
fig.suptitle(f"test record {r}: annotated, classical baseline, the lead-II network, the final network "
             f"({best_cfg[0]}, {best_cfg[1]}) | green P, red QRS, blue T, grey ignored by the metric",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "one_test_record_final.png", dpi=130)

ann = [(int(o), int(f)) for _, c, o, _, f in waves[(rec_of == r) & (waves[:, 1] == 2)]]
pairs = qrs_durations(ann, pred_final[i], MS)
all_pairs = [pr for j, rr in enumerate(idx_test)
             for pr in qrs_durations([(int(o), int(f)) for _, c, o, _, f
                                      in waves[(rec_of == rr) & (waves[:, 1] == 2)]], pred_final[j], MS)]
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
fig.savefig(FIGS / "qrs_duration_final.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, c in zip(axes, CLASSES):
    bv = [v for v in per_record(y_raw[idx_test], base_pred, c, y[idx_test]) if v is not None]
    pv = [v for v in per_record(y_raw[idx_test], pred_primary, c, y[idx_test]) if v is not None]
    fv = [v for v in per_record(y_raw[idx_test], pred_final, c, y[idx_test]) if v is not None]
    ax.boxplot([bv, pv, fv], tick_labels=["baseline", "lead II", "final"], showfliers=False)
    ax.set_ylabel(f"{WAVES[c]} Dice, one value per test record")
    ax.set_title(f"median {np.median(bv):.2f} -> {np.median(pv):.2f} -> {np.median(fv):.2f}", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "dice_per_record_final.png", dpi=130)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
keys = ranked_arms()
ys = np.arange(len(keys))
means = [np.mean([macro(rr) for rr in ARMS[k]["res"]]) for k in keys]
sds = [np.std([macro(rr) for rr in ARMS[k]["res"]]) for k in keys]
ax.barh(ys, means, xerr=sds, color=["tab:blue" if k == best_cfg else "0.7" for k in keys])
ax.axvline(macro(base_val_res), color="tab:red", ls="--", linewidth=1, label="classical baseline")
ax.set_yticks(ys, [f"{k[0]} | {k[1]} ({len(ARMS[k]['res'])} seeds)" for k in keys], fontsize=7)
ax.set_xlabel(f"validation macro Dice (mean +- sd over seeds)")
ax.set_xlim(min(means) - 0.03, 1)
ax.invert_yaxis()
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "improve_arms.png", dpi=130)

for net, n in zip(best_nets, SEEDS):
    torch.save(net.state_dict(), FIGS.parent / f"model_final_seed{n}.pt")   # model_ludb.pt is left as is
results = {
    "inherited": {"widths": WIDTHS, "lr": LR, "augment": AUG, "epochs": EPOCHS},
    "arms": {f"{k[0]} | {k[1]}": {"validation_mean": float(np.mean([macro(r) for r in ARMS[k]["res"]])),
                                  "validation_sd": float(np.std([macro(r) for r in ARMS[k]["res"]])),
                                  "seeds": [macro(r) for r in ARMS[k]["res"]],
                                  "waves": {WAVES[c]: float(np.mean([r[WAVES[c]]["macro"] for r in
                                                                     ARMS[k]["res"]])) for c in CLASSES}}
             for k in ranked_arms()},
    "chosen": {"leads": best_cfg[0], "lead_names": [LEADS[i] for i in LEAD_SETS[best_cfg[0]]],
               "loss": best_cfg[1], "loss_weights": WEIGHT.tolist(),
               "inference": {"models": int(n_models), "shifts": list(map(int, shifts)),
                             "merge_below_samples": int(merge_k)},
               "validation_mean": float(np.mean(seed_macros)), "validation_sd": float(np.std(seed_macros))},
    "test": {conv: {who: {WAVES[c]: table_test[conv][who][WAVES[c]] for c in CLASSES}
                    for who in ("network", "baseline")} for conv in CONVENTIONS},
    "test_leadII_primary": {WAVES[c]: score(pred_primary, "test")[WAVES[c]]["macro"] for c in CLASSES},
    "test_one_seed_no_tta": {WAVES[c]: score(pred_single, "test")[WAVES[c]]["macro"] for c in CLASSES},
    "test_seed_spread": {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed)),
                         "each": per_seed},
    "qrs_duration_ms": {"annotated_mean": float(a_all.mean()), "predicted_mean": float(p_all.mean()),
                        "mean_absolute_error": float(np.abs(a_all - p_all).mean()),
                        "beats_matched": int(len(a_all))},
    "baseline_config": {"kind": str(CHOSEN["kind"]), "frac": CHOSEN["frac"], "distance": CHOSEN["distance"],
                        "sigma": CHOSEN["sigma"]},
}
(FIGS / "results_final.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {FIGS / 'results_final.json'}, one_test_record_final.png, qrs_duration_final.png, "
      f"dice_per_record_final.png, improve_arms.png and model_final_seed*.pt")
