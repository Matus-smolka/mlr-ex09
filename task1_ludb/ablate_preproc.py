"""Does classical ECG preprocessing buy anything, now that the labels and the network are settled?

train_ludb.py and improve_ludb.py both feed the network the per-record standardized signal exactly as
the notebook's data cell produces it. That is defensible - LUDB stores every record scaled to exactly
1 mV peak-to-peak and then standardizes it, so the network never sees an absolute amplitude - but the
audit found the consequence: standardizing divides by a std that, in the low-amplitude records, is
mostly baseline wander. Record 125 ends up spanning -25.1 to +3.7 and 147 of the 200 records reach
|x| > 6 somewhere. A BatchNorm network can live with that, but it is a legitimate question whether
taking the wander out first helps.

Four arms, the winning arm of improve_ludb.py with nothing changed but the input transform:

  none      the standardized signal, as handed over;
  clip      saturated at +-5, which leaves every wave in the linear range (the widest QRS in the
            training set reaches well under 5) and only touches the drift spikes;
  wander    the 300 ms running median subtracted, then the record standardized again by its own std -
            the textbook baseline-wander removal of ECG preprocessing;
  both.

Two seeds each, because round 3 of improve_ludb.py measured what a difference has to beat before it
is a difference. Selection is on validation; the test split is evaluated once at the end for the
chosen transform and for `none`, so the report can say what the transform is worth on unseen data
instead of only on the split that picked it.

Usage:  py task1_ludb/ablate_preproc.py <masked_cache.npz> <figs-dir> [--quick]
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
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

from classical import FS, CHOSEN, labels as baseline_labels
from ludb_metric import CLASSES, WAVES, evaluate, fmt

CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
QUICK = "--quick" in sys.argv
EPOCHS = 6 if QUICK else 120
SEEDS = (0,) if QUICK else (0, 1)
MS = 1000 / FS
CLIP = 5.0
WANDER_MS = 300
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from part1 import UNet, device, predict, train      # noqa: E402

d = np.load(CACHE, allow_pickle=True)
x, x12, waves, y_raw, y = d["x"], d["x12"], d["waves"], d["y_raw"], d["y"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of, T = waves[:, 0], x.shape[1]
LEADS = [str(s) for s in d["lead_names"]]
final = json.loads((FIGS / "results_final.json").read_text())
WIDTHS, LR, AUG = final["inherited"]["widths"], final["inherited"]["lr"], final["inherited"]["augment"]
LEAD_SET = {"lead II": [LEADS.index("II")],
            "II+V1+V5": [LEADS.index(n) for n in ("II", "V1", "V5")],
            "all 12": list(range(len(LEADS)))}[final["chosen"]["leads"]]
WEIGHT = torch.tensor(final["chosen"]["loss_weights"], dtype=torch.float32) \
    if "weights" in final["chosen"]["loss"] else None
DICE = 1.0 if "dice" in final["chosen"]["loss"] else 0.0


def transform(sig, wander=False, clip=False):
    """Per record, per lead: subtract the running median, saturate. Nothing else, and nothing fitted."""
    out = sig.astype(np.float32)
    if wander:
        k = int(round(WANDER_MS / 1000 * FS)) | 1                     # odd window, 300 ms -> 151
        # size is per axis, so (1, 1, k) slides along time only - across neither record nor lead.
        # (`axes=` would say the same thing but is newer than the SciPy this needs.)
        out = out - median_filter(out, size=(1, 1, k), mode="nearest")
        out = (out - out.mean(axis=2, keepdims=True)) / (out.std(axis=2, keepdims=True) + 1e-6)
    return np.clip(out, -CLIP, CLIP) if clip else out


ARMS = {"none": (False, False), "clip": (False, True), "wander": (True, False), "both": (True, True)}
TRANSFORMED = {}


def signals(records, arm):
    """(n, n_leads, T) for one arm and one set of records, computed once and shared by its seeds.

    The cache key includes the records. Keying it on the arm alone made the statistics print below -
    which transforms all 200 records - poison the entry the 140 training records then read back, and
    the dataset answered len() 200 while holding 140 label rows.
    """
    records = np.asarray(records)
    key = (arm, records.tobytes())
    if key not in TRANSFORMED:
        raw = np.ascontiguousarray(x12[np.ix_(records, LEAD_SET, np.arange(T))], dtype=np.float32)
        TRANSFORMED[key] = transform(raw, *ARMS[arm])
    return TRANSFORMED[key]


class Records(Dataset):
    def __init__(self, records, arm, augment):
        self.x, self.y, self.augment = signals(records, arm), y[np.asarray(records)], augment

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


def soft_dice_loss(logits, target, eps=1e-6):
    keep = (target >= 0).float()[:, None]
    prob = torch.softmax(logits, 1) * keep
    onehot = F.one_hot(target.clamp(min=0), 4).permute(0, 2, 1).float() * keep
    inter = (prob * onehot).sum((0, 2))
    return 1 - ((2 * inter + eps) / (prob.sum((0, 2)) + onehot.sum((0, 2)) + eps))[1:].mean()


class Loss(nn.Module):
    def __init__(self, weight=None, dice=0.0):
        super().__init__()
        self.register_buffer("w", torch.ones(4) if weight is None else weight)
        self.dice = dice

    def forward(self, logits, target):
        loss = F.cross_entropy(logits, target, weight=self.w.to(logits.device), ignore_index=-1)
        return loss + self.dice * soft_dice_loss(logits, target) if self.dice else loss


def score(pred, which):
    return evaluate(y_raw[splits[which]], pred, y[splits[which]])


def macro(res):
    return float(np.mean([res[WAVES[c]]["macro"] for c in CLASSES]))


print(f"device {device} | arm under test: {final['chosen']['leads']} {[LEADS[i] for i in LEAD_SET]}, "
      f"loss {final['chosen']['loss']}, recipe {WIDTHS} lr {LR:.0e} augment {AUG}, {EPOCHS} epochs, "
      f"seeds {SEEDS}")
print(f"input transform arms: none | clip +-{CLIP:.0f} | {WANDER_MS} ms median subtracted | both")
for arm in ARMS:
    s = signals(np.arange(len(x)), arm)
    print(f"  {arm:7s} |x| > {CLIP:.0f} in {(np.abs(s) > CLIP).mean():7.4%} of samples, "
          f"range {s.min():+.1f} .. {s.max():+.1f}")

results = {}
for arm in ARMS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = train(UNet(1, len(LEAD_SET), 4, WIDTHS), Records(splits["train"], arm, AUG),
                    Records(splits["valid"], arm, False), Loss(WEIGHT, DICE), EPOCHS, lr=LR)
        plt.close("all")
        res = score(predict(net, Records(splits["valid"], arm, False))[0].argmax(1).numpy(), "valid")
        results.setdefault(arm, {"nets": [], "val": []})
        results[arm]["nets"].append(net)
        results[arm]["val"].append(macro(res))
        print(f"  {arm:7s} seed {seed}   validation {fmt(res)}   {macro(res):.3f}")

means = {a: float(np.mean(v["val"])) for a, v in results.items()}
print("\nvalidation, mean over seeds:")
for a in sorted(means, key=lambda a: -means[a]):
    print(f"  {a:7s} {means[a]:.3f}   (sd {np.std(results[a]['val']):.3f}, "
          f"{means[a] - means['none']:+.3f} against no preprocessing)")
winner = max(means, key=means.get)

# the metric an ECG engineer would also ask for, and the test split, once, for the winner and for none
@torch.no_grad()
def predict_plain(models, records, arm):
    sig = signals(records, arm)
    prob = torch.zeros(len(sig), 4, T, device=device)
    for model in models:
        model.eval()
        for i in range(0, len(sig), 32):
            blk = torch.from_numpy(sig[i:i + 32]).to(device)
            prob[i:i + 32] += torch.softmax(model(blk.float()), 1)
    return (prob / len(models)).argmax(1).cpu().numpy()


test_out = {}
for arm in dict.fromkeys((winner, "none")):
    pred = predict_plain(results[arm]["nets"], splits["test"], arm)
    res = score(pred, "test")
    test_out[arm] = {WAVES[c]: res[WAVES[c]]["macro"] for c in CLASSES}
    print(f"\ntest, transform '{arm}': {fmt(res)}   mean {macro(res):.3f}   "
          f"({len(results[arm]['nets'])} seed(s) averaged)")

base_valid, _, _ = baseline_labels(x[splits["valid"]], waves, rec_of, splits["train"], CHOSEN)
print(f"\nfor scale, the classical baseline on validation is {fmt(score(base_valid, 'valid'))}")
print(f"and the spread between seeds of one arm, which is what a difference has to beat, is "
      f"{max(np.std(v['val']) for v in results.values()):.3f} here")
(FIGS / "ablate_preproc.json").write_text(json.dumps(
    {"validation": means, "validation_seeds": {a: v["val"] for a, v in results.items()},
     "test": test_out, "chosen_by_validation": winner,
     "clip": CLIP, "wander_ms": WANDER_MS, "seeds": list(SEEDS)}, indent=1))
print(f"wrote {FIGS / 'ablate_preproc.json'}")
