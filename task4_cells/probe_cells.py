"""Is the network counting cells, or is it weighing the picture?

The integral of an image here correlates 0.991 with its cell count (inspect_cells.py), and dividing that
integral by one fitted number already scores a test MAE of 10.7 (classical_cells.py). A U-Net trained on
raw images with MSE can satisfy its loss largely through that door. This script measures how much of a
given arm's answer comes in through it, by turning the lamp up and down:

    gain g    every test pixel multiplied by g. The number of cells does not change, so a counter that
              counts cells keeps its answer and a counter that weighs brightness scales its answer by g.
    offset o  every pixel shifted by o, the same trick against a network that leans on the floor level.

Three arms of the chosen shape are retrained here for the comparison - `raw` (what the notebook's data
cell hands over), `raw + brightness jitter` (the same lesson taught by the data) and `zscore` (per-image
standardization, which makes a global gain or offset invisible by construction). Everything else -
the bumps' width, the depth, the loss, the learning rate - is taken from figs/results_net.json, the arm
the sweep chose, so the three lines of this figure differ only in what the network is allowed to see.

Usage:  py -I task4_cells/probe_cells.py <cache.npz> <figs-dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402
import warnings                                          # noqa: E402

from cells_data import CellsDataset, count_of, density, heads, make_loss    # noqa: E402
from cells_metric import evaluate                        # noqa: E402
from part1 import UNet, device, predict, train           # noqa: E402

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

d = np.load(CACHE, allow_pickle=True)
x, dots, count = d["x"], d["dots"].astype(np.float32), d["count"].astype(np.float32)
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
tr, va, te = splits["train"], splits["valid"], splits["test"]
ARM = json.loads((FIGS / "results_net.json").read_text())["arm"]
EPOCHS = json.loads((FIGS / "results_net.json").read_text())["epochs"]
print(f"probing the chosen arm {json.dumps(ARM)} at {EPOCHS} epochs; the only thing varied below is the input")

GAINS = (0.6, 0.8, 1.0, 1.25, 1.6)
OFFSETS = (-0.05, 0.0, 0.05)
ARMS = {"raw": dict(input="raw", augment=False),
        "raw + jitter": dict(input="raw", augment=True),
        "zscore": dict(input="zscore", augment=False)}

out = {}
for name, cfg in ARMS.items():
    torch.manual_seed(0)
    np.random.seed(0)
    net = nn.Sequential(UNet(2, 1, 1, ARM["widths"]), heads(ARM["head"])).to(device)
    maps = density(dots, ARM["sigma"])
    net = train(net, CellsDataset(x, maps, tr, cfg["input"], cfg["augment"]),
                CellsDataset(x, maps, va, cfg["input"], False), make_loss(ARM["loss"]), EPOCHS,
                lr=ARM["lr"])
    plt.close("all")

    def score(gain=1.0, offset=0.0):
        pred = predict(net, CellsDataset(x, maps, te, cfg["input"], False, gain, offset))[0]
        pred = pred.cpu().numpy()[:, 0]
        return pred, evaluate(count[te], count_of(pred), pred, dots[te])

    _, plain = score()
    integral = np.array([np.maximum(x[i] - 0.20, 0).sum() for i in te])
    pred0, _ = score()
    print(f"\n{name}: test MAE {plain['mae']:.2f} (bias {plain['bias']:+.1f}, recall {plain['recall']:.3f}); "
          f"its answer against the true count r = "
          f"{np.corrcoef(count_of(pred0), count[te])[0, 1]:+.3f}, against the brightness integral r = "
          f"{np.corrcoef(count_of(pred0), integral)[0, 1]:+.3f}")
    rows = {"clean": plain, "gain": {}, "offset": {},
            "corr_count": float(np.corrcoef(count_of(pred0), count[te])[0, 1]),
            "corr_integral": float(np.corrcoef(count_of(pred0), integral)[0, 1])}
    print(f"  {'lamp':>12s} {'mean count':>11s} {'ratio to g=1':>13s} {'MAE':>7s}   "
          f"{'what a cell-counter would do':>28s}")
    base = float(count_of(pred0).mean())
    for g in GAINS:
        p, r = score(gain=g)
        rows["gain"][str(g)] = {"mae": r["mae"], "mean": float(count_of(p).mean())}
        print(f"  gain {g:6.2f} {count_of(p).mean():11.1f} {count_of(p).mean() / base:13.3f} {r['mae']:7.2f}"
              f"   {'(answer unchanged)' if g == 1.0 else f'a weigher would say x{g:.2f} = {base * g:.0f}'}")
    for o in OFFSETS:
        p, r = score(offset=o)
        rows["offset"][str(o)] = {"mae": r["mae"], "mean": float(count_of(p).mean())}
        print(f"  offset {o:+.2f} {count_of(p).mean():11.1f} {count_of(p).mean() / base:13.3f} {r['mae']:7.2f}")
    out[name] = rows

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name, rows in out.items():
    g = np.array(list(rows["gain"]), float)
    y = np.array([rows["gain"][str(v)]["mean"] for v in g])
    axes[0].plot(g, y, "o-", label=name)
    axes[1].plot(g, [rows["gain"][str(v)]["mae"] for v in g], "o-", label=name)
axes[0].axhline(float(count[te].mean()), color="k", ls=":", lw=1, label="the true mean count")
flat = float(count[te].mean())
axes[0].plot(GAINS, [flat * g for g in GAINS], "--", color="0.6", lw=1, label="a brightness weigher")
axes[0].set_xlabel("gain applied to every test pixel")
axes[0].set_ylabel("mean count the network answers")
axes[0].set_title("turning the lamp up should not add cells", fontsize=9)
axes[0].legend(fontsize=7)
axes[1].set_xlabel("gain applied to every test pixel")
axes[1].set_ylabel("count MAE on the test split")
axes[1].set_title("... and the error should not grow with it", fontsize=9)
axes[1].legend(fontsize=7)
fig.suptitle("how much of the count comes from the cells and how much from the exposure: the same arm, "
             "three ways of showing it the picture", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "probe_leak.png", dpi=130)

out["_arms"] = {k: {**v, **ARM} for k, v in ARMS.items()}
(FIGS / "results_probe.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {FIGS / 'results_probe.json'} and probe_leak.png")
