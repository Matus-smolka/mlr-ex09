"""Does the label repair change the network, or only the score?

Two arms, same architecture, same seed, same number of epochs, same Adam schedule:

  A  trained on the labels exactly as the notebook parses them - the QRS complexes nobody annotated
     are then labelled "background", so the network is taught to answer background at a real beat;
  B  trained on the repaired labels of repair_ludb.py, where those samples are -1 and the loss skips
     them.

Each arm is then scored under BOTH conventions, "as parsed" (every sample counts, the unannotated
beat is a false positive) and "span + buffer" (it is ignored). That separation is the point: arm A
can only ever look better where the labels themselves are wrong, and the behavioural column - how
often each arm marks a beat the cardiologists left unlabelled - shows what it had to learn to do in
order to look better there.

The architecture and the learning rate are read from figs/results.json, so the ablation always runs
the model that train_ludb.py actually chose. The augmentation is deliberately left off in both arms:
the two arms may differ in the labels and in nothing else, and augmentation would be a third
variable moving between them.

Usage:  py task1_ludb/ablate_repair.py <masked_cache.npz> <figs-dir>
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
from torch.utils.data import Dataset

from classical import CHOSEN, detect
from ludb_metric import CLASSES, WAVES, evaluate, fmt

CACHE, FIGS = sys.argv[1], Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
EPOCHS = 120

d = np.load(CACHE, allow_pickle=True)
x, waves, y_raw, y = d["x"], d["waves"], d["y_raw"], d["y"]
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
rec_of, span, T = waves[:, 0], d["span"], x.shape[1]
chosen = json.loads((FIGS / "results.json").read_text())["chosen"] if (FIGS / "results.json").exists() else {}
WIDTHS, LR = chosen.get("widths", [16, 32, 64, 128, 256, 512]), chosen.get("lr", 1e-3)   # no augmentation

from part1 import UNet, predict, train      # noqa: E402  (after matplotlib is configured)


class Pairs(Dataset):
    """The same records with a choice of labels: repaired (with -1) or as parsed."""

    def __init__(self, records, labels):
        self.x, self.y = x[records], labels[records]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i].astype(np.float32)[None]), torch.from_numpy(self.y[i].astype(np.int64))


def qrs_runs(row):
    out, start = [], None
    for j, v in enumerate(list(row) + [0]):
        if v == 2 and start is None:
            start = j
        elif v != 2 and start is not None:
            out.append((start, j - 1))
            start = None
    return out


# the beats the cardiologists did not annotate, at the edges, found from the signal alone
edge_peaks, annotated_peaks = {}, {}
for r in splits["test"]:
    r = int(r)
    q = np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3])
    peaks = detect(x[r:r + 1], ("bandpass", (8, 20)), 0.4, 150, 0.0)[0]
    edge_peaks[r] = [int(p) for p in peaks if not (span[r, 0] <= p <= span[r, 1]) and
                     (len(q) == 0 or np.min(np.abs(q - p)) > 50)]
    annotated_peaks[r] = [int(p) for p in q]

results = {}
print(f"architecture UNet(1, 1, 4, {WIDTHS}), lr {LR}, {EPOCHS} epochs, seeds fixed, no augmentation")
for arm, labels in [("A  as parsed (unannotated beats taught as background)", y_raw),
                    ("B  repaired (those samples are -1)", y)]:
    torch.manual_seed(0)
    np.random.seed(0)
    net = train(UNet(1, 1, 4, WIDTHS), Pairs(splits["train"], labels), Pairs(splits["valid"], labels),
                nn.CrossEntropyLoss(ignore_index=-1), EPOCHS, lr=LR)
    plt.close("all")
    pred = predict(net, Pairs(splits["test"], y))[0].argmax(1).numpy()
    print(f"\n{arm}")
    for conv, m in [("as parsed", None), ("span + buffer", y)]:
        idx = splits["test"]
        res = evaluate(y_raw[idx], pred, None if m is None else m[idx])
        print(f"  judged {conv:14s} {fmt(res)}   mean {np.mean([res[WAVES[c]]['macro'] for c in CLASSES]):.3f}")
        results.setdefault(arm, {})[conv] = {WAVES[c]: res[WAVES[c]]["macro"] for c in CLASSES}
    # behaviour, independent of any convention: does it mark a beat at all?
    on_ann = on_edge = 0
    for j, r in enumerate(splits["test"]):
        runs = qrs_runs(pred[j])
        on_ann += sum(any(p - 50 <= q <= p + 50 for p, q in runs) for q in annotated_peaks[int(r)])
        on_edge += sum(any(p - 50 <= q <= p + 50 for p, q in runs) for q in edge_peaks[int(r)])
    n_ann = sum(len(v) for v in annotated_peaks.values())
    n_edge = sum(len(v) for v in edge_peaks.values())
    print(f"  marks {on_ann}/{n_ann} annotated beats ({on_ann / n_ann:.1%}) and "
          f"{on_edge}/{n_edge} unannotated edge beats ({on_edge / n_edge:.1%})")
    results[arm]["beats"] = {"annotated_found": on_ann, "annotated_total": n_ann,
                             "edge_marked": on_edge, "edge_total": n_edge}

(FIGS / "ablate_repair.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {FIGS / 'ablate_repair.json'}")
