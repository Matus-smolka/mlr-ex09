"""Task 4: a 2D U-Net paints a density map, and the count is the sum of its output.

In the order the notebook asks for the work:

  1. `UNet`, `train`, `predict` of Part 1, imported from part1.py **unchanged** - `UNet(2, ...)` is the
     same code with 2D convolutions instead of 1D, which is what the task means by "the same U-Net";
  2. the Dataset: one image -> (1, 256, 256) blue channel and a (256, 256) density target, a Gaussian
     on every annotated centre, so the target sums to the count (checked in section 1 below);
  3. the choices - width of the map's bumps, depth of the U-Net, loss, last layer, input
     normalization, augmentation, learning rate - every one of them made on the 20 validation images
     by the count MAE, and reported next to a localization score, because on this data a count MAE can
     be won from the total brightness of the image without finding a single cell (classical_cells.py
     section 4: MAE 10.7 from one fitted number);
  4. the winning arm three times over (seeds 0, 1, 2), the test split evaluated **once**, next to the
     two baselines the task names;
  5. the numbers, to figs/results_net.json, and the maps themselves to figs/preds_test.npz, which is
     what report_cells.py draws the required figures from.

Usage:  py -I task4_cells/train_cells.py <cache.npz> <figs-dir> [--quick]
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))           # our own modules, however this is run

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402
from scipy import ndimage                                # noqa: E402
from cells_data import CellsDataset, count_of, density, heads, make_loss    # noqa: E402
from cells_metric import evaluate, localize              # noqa: E402
from part1 import UNet, device, predict, train           # noqa: E402

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
FIGS.mkdir(parents=True, exist_ok=True)
QUICK = "--quick" in sys.argv
EPOCHS = 6 if QUICK else 300            # train() keeps the weights of the best validation epoch
if "--epochs" in sys.argv:               # ... unless the run says otherwise (the epoch budget is a knob)
    EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1])
FINAL_ONLY = "--final-only" in sys.argv  # skip the sweep, reuse the arm in figs/results_net.json
LR_OVERRIDDEN = "--lr" in sys.argv
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")   # train() calls plt.show()
torch.backends.cudnn.deterministic = True                # so a "seed" means a reproducible run
torch.backends.cudnn.benchmark = False

d = np.load(CACHE, allow_pickle=True)
x, dots, count = d["x"], d["dots"].astype(np.float32), d["count"].astype(np.float32)
splits = {s: d[f"split_{s}"] for s in ("train", "valid", "test")}
tr, va, te = splits["train"], splits["valid"], splits["test"]

# ---------------------------------------------------------------------------------------------
# 1. the target: a Gaussian on every annotated centre, so that sum(map) == count
# ---------------------------------------------------------------------------------------------
SIGMAS = [1.5] if QUICK else [1.0, 1.5, 2.0, 3.0]
PICK = [0, 1, int(np.argmax(count[tr]))]          # the notebook's three: first, second, the densest

print(f"device {device} | torch {torch.__version__} | {len(tr)} train / {len(va)} valid / {len(te)} test images")
print("=" * 96)
print("1. THE TARGET SUMS TO THE COUNT (the notebook's check, on a few training images)")
print("=" * 96)
print(f"{'sigma':>6s} {'image':>7s} {'count':>6s} {'sum(target)':>12s} {'ratio':>8s} {'brightest px':>13s}")
for sigma in SIGMAS:
    m = density(dots, sigma)
    ratio = count_of(m) / count
    for i in PICK:
        print(f"{sigma:6.1f} {i + 1:7d} {count[i]:6.0f} {m[i].sum():12.3f} {ratio[i]:8.5f} {m[i].max():13.4f}")
    cut = np.stack([ndimage.gaussian_filter(v.astype(np.float64), sigma, mode="constant")
                    for v in dots]).reshape(len(dots), -1).sum(1) / count
    print(f"{sigma:6.1f} {'all 200':>7s} {'':6s} ratio {ratio.min():.5f} .. {ratio.max():.5f}   "
          f"if the borders were zero-padded instead: {cut.min():.4f} .. {cut.max():.4f} "
          f"({100 * (1 - cut.min()):.2f}% lost on the worst image)")


# ---------------------------------------------------------------------------------------------
# 2. the data - CellsDataset, the losses and the heads live in cells_data.py, which probe_cells.py
#    imports too so that "the same arm" is the same code in both scripts.
# ---------------------------------------------------------------------------------------------
def cells_set(idx, sigma, input="raw", augment=False):
    return CellsDataset(x, density(dots, sigma), idx, input, augment)


def run(arm, epochs=EPOCHS, seeds=(0,), keep=False):
    """Train one arm, score it on validation (or the given split), return its record.

    Scored by count MAE - the metric of the specification - and by localization on the same maps, and
    by the two things a sum-of-the-map readout can get wrong: a negative patch (allowed by a linear
    head) and a peak count that disagrees with the sum.
    """
    out = {"arm": arm, "seeds": []}
    maps_va = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = nn.Sequential(UNet(2, 1, 1, arm["widths"]), heads(arm["head"])).to(device)
        set_va = cells_set(va, arm["sigma"], arm["input"], False)
        t0 = time.perf_counter()
        net = train(net, cells_set(tr, arm["sigma"], arm["input"], arm["augment"]), set_va,
                    make_loss(arm["loss"]), epochs, lr=arm["lr"])
        plt.savefig(FIGS / f"curve_{arm['tag']}_s{seed}.png", dpi=110)
        plt.close("all")
        pred = predict(net, set_va)[0].cpu().numpy()[:, 0]
        res = evaluate(count[va], count_of(pred), pred, dots[va])
        out["seeds"].append({"seed": seed, "net": net, "valid": res, "valid_maps": pred,
                             "seconds": round(time.perf_counter() - t0, 1)})
        maps_va.append(pred)
    if len(seeds) > 1:                                    # the ensemble of the seeds, same validation maps
        ens = np.mean(maps_va, 0)
        out["valid_ensemble"] = evaluate(count[va], count_of(ens), ens, dots[va])
        out["valid_maps"] = ens
    v = out["seeds"][0]["valid"] if len(seeds) == 1 else out["valid_ensemble"]
    out["valid"] = v
    print(f"  {arm['tag']:34s} valid MAE {v['mae']:6.2f}  bias {v['bias']:+7.2f}  "
          f"recall {v.get('recall', float('nan')):.3f}  precision {v.get('precision', float('nan')):.3f}  "
          f"negative pixels {100 * np.mean(out['seeds'][0]['valid_maps'] < 0):.2f}%")
    if keep:
        for s in out["seeds"]:
            torch.save(s["net"].state_dict(), FIGS.parent / f"model_{arm['tag']}_seed{s['seed']}.pt")
    return out


def best_of(records):
    return min(records, key=lambda r: r["valid"]["mae"])


# ---------------------------------------------------------------------------------------------
# 3. the choices, every one of them on the 20 validation images
# ---------------------------------------------------------------------------------------------
BASE = dict(sigma=2.0, widths=[16, 32, 64, 128], head="none", loss="mse", input="raw",
            augment=False, lr=1e-3)
# Widths of the U-Net: a block's two 3x3 convolutions see +-2 px at their own resolution and pooling
# doubles that per level, so 4 levels reach +-30 px and 5 levels +-62 px. A cell is ~13 px across and
# the median nearest-neighbour centre distance is 8.5 px, so the finest path already sees several
# cells; the deep path is what tells a crowd from a blob.
sweep_records, sweep_previous = [], {}
if FINAL_ONLY:            # the arm was chosen by an earlier run; here only the epoch budget changes
    previous = json.loads((FIGS / "results_net.json").read_text())
    BASE = {**previous["arm"], "tag": "reused"}
    if "--lr" in sys.argv:
        BASE["lr"] = float(sys.argv[sys.argv.index("--lr") + 1])
    sweep_records, sweep_previous = [], previous["validation_sweep"]
    ARMS_TRIED = previous["arms_tried_on_validation"]
    print(f"\nreusing the arm an earlier run chose on validation: {json.dumps(previous['arm'])}. "
          f"Its validation MAE there was {min(v['valid_mae'] for v in sweep_previous.values()):.2f} at "
          f"{previous['epochs']} epochs; this run gives it {EPOCHS} epochs and compares on the same 20 images.")
else:
    ARMS = []
    if QUICK:
        ARMS.append({**BASE, "sigma": 1.5, "tag": "smoke_sig1.5_4lv"})
    else:
        # A 5-level net was measured first, at sigma 1.0: 24.41 against 15.09 on validation, and 3.5x
        # the wall clock (logs/train_partial_5lv.log) - the extra pooling throws away detail at exactly
        # the scale the cells live at. 4 levels, and a wider 4 levels in stage D, is what is swept.
        for s in (1.0, 1.5, 2.0, 3.0):
            ARMS.append({**BASE, "sigma": s, "tag": f"A_sig{s}"})
    print()
    print("=" * 96)
    print(f"2. STAGE A - width of the bumps ({len(ARMS)} runs, seed 0, {EPOCHS} epochs, 4 levels, MSE, "
          f"raw input, no augmentation)")
    print("=" * 96)
    stage_a = [run(a) for a in ARMS]
    win = best_of(stage_a)
    BASE = {**win["arm"], "tag": "win"}
    print(f"stage A keeps sigma {BASE['sigma']}, widths {BASE['widths']} "
          f"(valid MAE {win['valid']['mae']:.2f}); the others: "
          + ", ".join(f"{r['arm']['tag']} {r['valid']['mae']:.2f}"
                      for r in sorted(stage_a, key=lambda r: r["valid"]["mae"])))

if not (QUICK or FINAL_ONLY):
    print()
    print("=" * 96)
    print("3. STAGE B - the loss and the last layer, on the stage-A winner")
    print("=" * 96)
    stage_b = []
    for loss in ("mse", "l1", "mse+100", "mse+1000"):
        for head in (["none", "softplus"] if loss == "mse" else ["none"]):
            stage_b.append(run({**BASE, "loss": loss, "head": head, "tag": f"B_{loss}_{head}"}))
    win = best_of(stage_b)
    BASE = {**win["arm"], "tag": "win"}
    print(f"stage B keeps loss '{BASE['loss']}' and last layer '{BASE['head']}' (valid MAE "
          f"{win['valid']['mae']:.2f}); " + ", ".join(f"{r['arm']['tag']} {r['valid']['mae']:.2f}"
                                                      for r in sorted(stage_b, key=lambda r: r["valid"]["mae"])))

    print()
    print("=" * 96)
    print("4. STAGE C - what the network is allowed to see, and how the images are varied")
    print("=" * 96)
    print("   'raw' leaves the total brightness in the image (the leak); 'zscore' removes it, so the")
    print("   count can only come from where the light is. 'jitter' is the same lesson taught by data.")
    stage_c = []
    for inp, aug in (("raw", False), ("raw", True), ("zscore", True)):
        stage_c.append(run({**BASE, "input": inp, "augment": aug, "tag": f"C_{inp}{'+jit' if aug else ''}"}))
    win_score = best_of(stage_c)                       # by the task's metric: count MAE
    win_local = min(stage_c, key=lambda r: -r["valid"].get("f1", 0))
    BASE = {**win_score["arm"], "tag": "win"}
    print(f"stage C keeps input '{BASE['input']}', augment {BASE['augment']} on the count MAE "
          f"(valid MAE {win_score['valid']['mae']:.2f}); the best localization in the same stage was "
          f"'{win_local['arm']['input']}' at F1 {win_local['valid']['f1']:.3f} / MAE "
          f"{win_local['valid']['mae']:.2f} - both arms are carried to the test split.")

    print()
    print("=" * 96)
    print("5. STAGE D - learning rate, and a wider net at the winner's shape")
    print("=" * 96)
    stage_d = [run({**BASE, "lr": lr, "tag": f"D_lr{lr:.0e}"}) for lr in (3e-4, 3e-3)]
    stage_d.append(run({**BASE, "widths": [32, 64, 128, 256], "tag": "D_wide"}))   # same 4 levels, twice the channels
    win = best_of(stage_d)
    if win["valid"]["mae"] < min(r["valid"]["mae"] for r in stage_c):
        BASE = {**win["arm"], "tag": "final"}
    print(f"chosen arm: {json.dumps({k: v for k, v in BASE.items() if k != 'tag'})} "
          f"-> valid MAE {min(r['valid']['mae'] for r in stage_c + stage_d):.2f}")

sweep_records = sweep_records if FINAL_ONLY else (
    [r for r in stage_a] if QUICK else stage_a + stage_b + stage_c + stage_d)
if not FINAL_ONLY:
    ARMS_TRIED = len(sweep_records)

# ---------------------------------------------------------------------------------------------
# 6. the chosen arm, three seeds, and the test split once
# ---------------------------------------------------------------------------------------------
print()
print("=" * 96)
print(f"6. THE CHOSEN ARM x3 SEEDS, then the TEST split evaluated once ({ARMS_TRIED} arms were tried on "
      f"validation to get here)")
print("=" * 96)
final_arm = {**BASE, "tag": f"cells_e{EPOCHS}" + (f"_lr{BASE['lr']:.0e}" if LR_OVERRIDDEN else "")}
final = run(final_arm, seeds=(0,) if QUICK else (0, 1, 2), keep=True)
for s in final["seeds"]:
    print(f"   seed {s['seed']}: validation MAE {s['valid']['mae']:.2f}, best epochs {s['seconds']:.0f} s, "
          f"{sum(p.numel() for p in s['net'].parameters()) / 1e6:.1f} M parameters")

test_sets = {who: cells_set(splits[who], final_arm["sigma"], final_arm["input"], False)
             for who in ("train", "valid", "test")}
table = {}
for s in final["seeds"]:
    for who in ("train", "valid", "test"):
        pred = predict(s["net"], test_sets[who])[0].cpu().numpy()[:, 0]
        idx = splits[who]
        s.setdefault("maps", {})[who] = pred
        s.setdefault("res", {})[who] = evaluate(count[idx], count_of(pred), pred, dots[idx])
print(f"\n{'seed':>6s} {'train MAE':>10s} {'valid MAE':>10s} {'TEST MAE':>9s} {'TEST RMSE':>10s} "
      f"{'bias':>7s} {'R2':>7s} {'worst':>6s} {'recall':>7s} {'prec':>6s} {'F1':>6s}")
for s in final["seeds"]:
    r = s["res"]["test"]
    print(f"{s['seed']:>6d} {s['res']['train']['mae']:10.2f} {s['res']['valid']['mae']:10.2f} "
          f"{r['mae']:9.2f} {r['rmse']:10.2f} {r['bias']:+7.2f} {r['r2']:7.3f} {r['worst']:6.0f} "
          f"{r['recall']:7.3f} {r['precision']:6.3f} {r['f1']:6.3f}")
maes = [s["res"]["test"]["mae"] for s in final["seeds"]]
print(f"seed spread on test: {min(maes):.2f} .. {max(maes):.2f} (mean {np.mean(maes):.2f}) - this is the "
      f"noise floor of the arm itself, read every difference smaller than it with care")

ens_maps = np.mean([s["maps"]["test"] for s in final["seeds"]], 0)
ens = evaluate(count[te], count_of(ens_maps), ens_maps, dots[te])
print(f"{'ensemble':>6s} {'':10s} {'':10s} {ens['mae']:9.2f} {ens['rmse']:10.2f} {ens['bias']:+7.2f} "
      f"{ens['r2']:7.3f} {ens['worst']:6.0f} {ens['recall']:7.3f} {ens['precision']:6.3f} {ens['f1']:6.3f}")

np.savez_compressed(FIGS / "preds_test.npz",
                    sigma=final_arm["sigma"], input_mode=final_arm["input"],
                    widths=np.array(final_arm["widths"]),
                    **{f"maps_seed{s['seed']}_{who}": s["maps"][who] for s in final["seeds"]
                       for who in ("train", "valid", "test")},
                    **{f"count_{who}": count[splits[who]] for who in ("train", "valid", "test")})
results = {"arm": {k: v for k, v in final_arm.items() if k != "tag"}, "arms_tried_on_validation": ARMS_TRIED,
           "epochs": EPOCHS, "parameters": int(sum(p.numel() for p in final["seeds"][0]["net"].parameters())),
           "seed_seconds": [s["seconds"] for s in final["seeds"]],
           "validation_sweep": {**sweep_previous,
                                **{r["arm"]["tag"]: {"arm": {k: v for k, v in r["arm"].items() if k != "tag"},
                                                     "valid_mae": r["valid"]["mae"],
                                                     "valid_f1": r["valid"].get("f1")}
                                   for r in sweep_records},
                                final_arm["tag"]: {"arm": {k: v for k, v in final_arm.items() if k != "tag"},
                                                   "valid_mae": final["valid"]["mae"],
                                                   "valid_f1": final["valid"].get("f1")}},
           "seeds": {str(s["seed"]): s["res"] for s in final["seeds"]}, "ensemble_test": ens,
           "ceiling_target_map": {str(sig): localize(density(dots, sig)[te], dots[te])
                                  for sig in ([1.5] if QUICK else (1.0, 1.5, 2.0, 3.0))}}
(FIGS / "results_net.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {FIGS / 'results_net.json'}, preds_test.npz and "
      f"model_{final_arm['tag']}_seed*.pt")
