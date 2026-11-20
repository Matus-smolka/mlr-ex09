"""Turn one trained arm into the .onnx the course app loads, and check that it still says the same thing.

The app (https://tomasvicar.github.io/MLR-public/apps/activity/) feeds the network exactly one tensor -
`acc`, 1 x 3 x window, the raw axes in the order the phone reports them - and reads `logits`,
1 x 7 x window, one class per sample. Everything else it needs is in the `model_info` metadata string:
the class names in the order the logits come out, the units the network was trained on (the browser's
Sensor gives m/s2, so with "g" here the app divides by 9.80665), the sampling rate, the window length and
the axis order. A network trained on canonicalized windows (arm D) is therefore not exportable: something
would have to rotate every window before the network sees it, and that something is not in the app.

Which arm goes in? The one with the best VALIDATION macro F1 among those that read the raw axes. Picking
it on the test users instead would spend the test set twice - once to choose, once to measure - and the
number would mean less for it.

Usage:  py -I export_onnx.py <figs-dir-with-model_*.pt> <out.onnx> [--arm A] [--cache hapt.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part1                                                    # noqa: E402
from motion_data import CLASSES, eval_windows, load_splits      # noqa: E402

for s in (sys.stdout, sys.stderr):                              # the exporter prints a check mark; a cp1250
    if hasattr(s, "reconfigure"):                                # Windows console would stop the script over it
        s.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("figs", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--arm", default="", help="which arm to export (A, B, C...); default: the best exportable "
                                          "one on the validation users")
ap.add_argument("cache", type=Path, nargs="?", default=None)
A = ap.parse_args()

ckpt_of = {"A": "model_A_plain.pt", "B": "model_B_rot24.pt", "C": "model_C_rotSO3.pt",
           "D": "model_D_gravity_aligned.pt"}
results = json.loads((A.figs / "results_net.json").read_text()) if (A.figs / "results_net.json").exists() else {}
arms = {k: v for k, v in results.get("arms", {}).items() if not ckpt_of.get(k, "").startswith("model_D")}
if A.arm:
    key = A.arm
    assert key in ckpt_of, f"--arm must be one of {sorted(ckpt_of)}"
else:
    assert arms, "no results_net.json with the arms' validation scores; pass --arm explicitly"
    key = max(arms, key=lambda k: arms[k]["validation"]["macro_f1"])
    print(f"exportable arms (a network that reads the raw axes the app feeds it): "
          f"{', '.join(sorted(arms))}; chosen {key} on validation macro F1 "
          + ", ".join(f"{k} {arms[k]['validation']['macro_f1']:.3f}" for k in sorted(arms)))
    print(f"      chosen on validation, not on test, so the test number below is still worth what it says")

path = A.figs / ckpt_of[key]
ck = torch.load(path, map_location="cpu", weights_only=False)
model = part1.UNet(1, 3, 7, ck["widths"])
model.load_state_dict(ck["state"])
model.eval()
W = ck["window"]
assert not ck["canon"], "this arm needs its windows rotated before the network sees them; the app cannot do that"
print(f"{path.name}: UNet(1, 3, 7, {ck['widths']}), window {W}, aug {ck['aug']}, trained on "
      f"{ck['units']} at {ck['sample_rate']} Hz; "
      f"{sum(p.numel() for p in model.parameters())} parameters")

with torch.no_grad():
    ref = model(torch.zeros(1, 3, W))
assert tuple(ref.shape) == (1, 7, W), f"the app expects 1 x 7 x window, this nets {tuple(ref.shape)}"

torch.onnx.export(model, (torch.zeros(1, 3, W),), str(A.out), input_names=["acc"],
                  output_names=["logits"], external_data=False)
import onnx                                                     # noqa: E402
m = onnx.load(str(A.out))
info = {"classes": list(ck["classes"]), "units": ck["units"], "sample_rate": int(ck["sample_rate"]),
        "window": int(W), "axes": list(ck["axes"])}
assert info["classes"] == CLASSES, "the class order in the file must be the order the app colours by"
m.metadata_props.add(key="model_info", value=json.dumps(info))
onnx.checker.check_model(m)
onnx.save(m, str(A.out))
print(f"wrote {A.out} ({A.out.stat().st_size / 1e6:.2f} MB), opset {m.opset_import[0].version}, "
      f"model_info {json.dumps(info)}")
print(f"      in  {[('%s %s' % (i.name, [d if isinstance(d, int) else d.dim_value for d in i.type.tensor_type.shape.dim])) for i in m.graph.input]}")
print(f"      out {[('%s %s' % (o.name, [d if isinstance(d, int) else d.dim_value for d in o.type.tensor_type.shape.dim])) for o in m.graph.output]}")
print("      the batch axis is fixed at 1 on purpose - the app sends one window at a time")

import onnxruntime as ort                                       # noqa: E402
sess = ort.InferenceSession(str(A.out), providers=["CPUExecutionProvider"])
got = sess.run(None, {"acc": np.zeros((1, 3, W), np.float32)})[0]
print(f"onnxruntime on a window of zeros: {got.shape}, max |logit - torch logit| "
      f"{np.abs(got - ref.numpy()).max():.2e}")

if A.cache:                                                     # the real check: real windows, real decisions
    splits, _ = load_splits(A.cache)
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for a, y, _ in splits["test"]:
        wins = np.array([s for s, _ in eval_windows(len(a), W)])
        for s in rng.choice(wins, size=min(20, len(wins)), replace=False):
            xs.append(np.asarray(a[s:s + W], np.float32).T)
            ys.append(y[s:s + W])
    x = np.stack(xs)
    y = np.stack(ys)
    with torch.no_grad():
        tp = model(torch.from_numpy(x)).numpy()
    # one window per call, because that is all the app ever sends: the exported graph is fixed at batch 1
    op = np.concatenate([sess.run(None, {"acc": x[i:i + 1]})[0] for i in range(len(x))])
    lab = y >= 0
    print(f"{len(x)} real test windows (users 26-30): max |logit| difference {np.abs(op - tp).max():.2e} g, "
          f"same class on {100 * np.mean((op.argmax(1) == tp.argmax(1))[lab]):.4f}% of the {lab.sum()} "
          f"labelled samples; top-1 probability agrees to "
          f"{np.abs(op.max(1) - tp.max(1)).max():.1e}")
    print(f"      in g as trained, so the app divides the browser's m/s2 by 9.80665 before feeding this; "
          f"a window of {W} samples is {W / ck['sample_rate']:.1f} s at {ck['sample_rate']} Hz")
    other = 128 if W % 128 == 0 else 64
    with torch.no_grad():
        o2 = model(torch.zeros(1, 3, other)).shape[2]
    print(f"      the net itself is not tied to {W}: a window of {other} samples gives {o2} answers back "
          f"(it needs a length divisible by {2 ** (len(ck['widths']) - 1)}); the app uses the {W} in the "
          f"metadata")
