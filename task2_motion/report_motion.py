"""Everything the notebook asks to show for Task 2, drawn from what train_motion.py left behind.

Nothing here trains or searches: it reads results_net.json, the model_*.pt files and the saved test
predictions, checks that they agree with each other, and prints the numbers and draws the figures the
specification asks for -

* the specification table (cell 53 of the notebook),
* macro F1 and per-class F1 on the test users, and the confusion matrix, next to the baseline,
* the true and the predicted timeline of one test recording,
* and - for the orientation part the task adds - what happens when the phone is worn turned, both as a
  table over the wearings and as a timeline of a turned recording read plain and read after the search.

Usage:  py -I report_motion.py <cache/hapt.npz> <figs-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt                                 # noqa: E402
import part1                                                    # noqa: E402
from motion_data import (CLASSES, COLORS, FS, AccWindows, eval_windows, f1_scores,  # noqa: E402
                         lab_to_rgba, load_splits, perm_name, signed_permutations)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE, FIGS = Path(sys.argv[1]), Path(sys.argv[2])
net = json.loads((FIGS / "results_net.json").read_text())
W = net["window"]
splits, all_recs = load_splits(CACHE)
CKPT = {}                                                 # arm letter -> (name written on the file, checkpoint)
for p in sorted(FIGS.glob("model_*.pt")):
    c = torch.load(p, map_location="cpu", weights_only=False)
    CKPT[c["arm"]] = (p.stem[len("model_"):], c)
names = {k: CKPT[k][0] for k in CKPT}
ARM_KEYS = [k for k in net["arms"] if k in CKPT]
assert set(ARM_KEYS) == set(net["arms"]), f"trained {sorted(net['arms'])} on disk, found {sorted(CKPT)}"
CA = net["chosen_arm"]
NAME_OF = {v: k for k, v in names.items()}
NAMES = sorted(FIGS.glob("preds_*_test.npz"), key=lambda p: p.name)
PRED = {p.stem[len("preds_"):-len("_test")]: np.load(p) for p in NAMES if "turned" not in p.stem}
TURNED = {int(p.stem.split("_")[2]): np.load(p) for p in sorted(FIGS.glob("preds_turned_*_test.npz"))}
# the artifacts say how many test recordings they cover - a short run covers fewer than the split - so
# they, not the cache, decide the scope of everything below
N_REC = max(int(z["rec"].max()) + 1 for z in list(PRED.values()) + list(TURNED.values()))
TEST = splits["test"][:N_REC]
Yte = np.concatenate([r[1] for r in TEST])
BOUNDS = np.concatenate([[0], np.cumsum([len(r[0]) for r in TEST])])
PERMS = signed_permutations()


def strip(pred, r):
    """The samples of one test recording out of a prediction over the concatenation of all of them."""
    return np.asarray(pred)[BOUNDS[r]:BOUNDS[r + 1]]


print("=" * 98)
print("0. do the artifacts agree with each other?")
print("=" * 98)
for nm, z in PRED.items():
    assert len(z["pred"]) == len(Yte), f"{nm}: {len(z['pred'])} predictions for {len(Y['test'])} samples"
    assert (z["pred"] >= 0).all(), f"{nm}: {int((z['pred'] < 0).sum())} samples were never answered"
    again = f1_scores(Yte[Yte >= 0], z["pred"][Yte >= 0])
    k = NAME_OF[nm]
    assert abs(again["accuracy"] - net["final"][k]["accuracy"]) < 1e-9, \
        f"{nm}: {again['accuracy']:.6f} from the npz against {net['final'][k]['accuracy']:.6f} in the json"
    assert abs(again["macro_f1"] - net["final"][k]["macro_f1"]) < 1e-9, f"{nm}: macro F1 disagrees"
for i, z in TURNED.items():
    assert abs(np.linalg.det(z["rot"]) - 1) < 1e-5 and np.allclose(z["rot"] @ z["rot"].T, np.eye(3), atol=1e-5)
    assert (z["searched"] >= 0).all(), f"turned {i}: the searched timeline has holes"
grid = {(r["arm"], r["how"], r["agg"], r["margin"]) for r in net["search_grid"]}
chosen = (net["search_chosen"]["arm"], net["search_chosen"]["how"], net["search_chosen"]["agg"],
          net["search_chosen"]["margin"])
assert chosen in grid, "the chosen setting was not one of the settings scored"
perms_of = {i: z["name"] for i, z in TURNED.items()}
print(f"{len(PRED)} prediction files ({', '.join(PRED)}) over {len(Yte)} test samples in "
      f"{N_REC} recordings; every sample answered once, and re-scoring each file from disk "
      f"reproduces the accuracy and macro F1 in results_net.json to 1e-9")
print(f"the setting the search runs with ({names[CA]}, per {chosen[1]}, score {chosen[2]}, margin "
      f"{chosen[3]}) is one of the {len(grid)} scored on the validation users; the test users were scored "
      f"once, after that choice")
print(f"turned cases saved: {', '.join(f'{i}: {perms_of[i]}' for i in TURNED)}")


print("\n" + "=" * 98)
print("1. the specification table (notebook cell 53), filled in from what was actually run")
print("=" * 98)
ck = CKPT[CA][1]
spec = [("network input (shape)", f"(batch, 3, {W}) - the three accelerometer axes in g at {FS} Hz, "
                                 f"{W / FS:.1f} s per window; at evaluation every sample of a recording is "
                                 f"answered once, from a full window, never from padding"),
        ("network output (shape)", f"(batch, 7, {W}) - seven logits per sample, one answer for every "
                                   f"sample of the window, not one for the window"),
        ("last layer", f"nn.Conv1d({ck['widths'][0]}, 7, 1), a 1 x 1 convolution; no softmax - softmax only "
                       f"for reading the probabilities, argmax for the answer"),
        ("training loss", "CrossEntropyLoss(ignore_index=-1) over the 7 classes - the samples between "
                          "activities (label -1) cost nothing, and a window in which EVERY sample is -1 is "
                          "left out because torch returns NaN for it, not 0"),
        ("evaluation metric", "macro F1 over the 7 classes on the test users (mean of the per-class F1s, so "
                              "the rare 3.6 s transition counts as much as standing), with accuracy "
                              "alongside; -1 samples excluded"),
        ("baseline", f"always answering standing: test accuracy {net['baselines']['always_standing']['accuracy']:.3f}, "
                     f"macro F1 {net['baselines']['always_standing']['macro_f1']:.3f} (the notebook says about 0.18 and 0.04); "
                     f"a second, stronger baseline answers one class per window by nearest class average "
                     f"(below)"),
        ("network", f"UNet(1, 3, 7, {ck['widths']}) from Part 1, {net['arms'][CA]['parameters'] / 1e3:.0f}k "
                    f"parameters, {net['epochs']} epochs, batch {net['batch']}, lr {net['lr']}, Adam, best "
                    f"validation weights")]
for k, v in spec:
    print(f"| {k:22s} | {v} |")
print(f"\n(paste-ready) network {names[CA]} is the arm the search was chosen on; the exported file is the "
      f"same network without the search, since the app runs one forward pass per window")


print("\n" + "=" * 98)
print("2. the numbers on the test users, next to the baselines")
print("=" * 98)
rows = [("baseline: always standing", net["baselines"]["always_standing"], None),
        ("baseline: nearest class average, one answer per window", net["baselines"]["nearest_centroid_window"], None)]
rows += [(f"{names[k]} (as it comes)", net["final"][k]["test"], k) for k in ARM_KEYS]
zs = PRED.get(names[CA])
searched = net["per_rotation"]["x,y,z"]
if zs is not None and "searched" in zs.files:               # the searched timeline of the same run, from disk
    m = Yte >= 0
    sr = f1_scores(Yte[m], zs["searched"][m])
    assert abs(sr["accuracy"] - searched["searched"]) < 1e-9 and abs(sr["macro_f1"] - searched["searched_macro_f1"]) < 1e-9, \
        "the searched predictions on disk disagree with the searched numbers in the json"
else:
    sr = {"accuracy": searched["searched"], "macro_f1": searched["searched_macro_f1"]}
rows.append((f"{names[CA]} + the orientation search (nothing turned)", sr, None))
print(f"{'model':58s} {'accuracy':>9s} {'macro F1':>9s} | " + " ".join(f"{c[:4]:>5s}" for c in CLASSES))
for label, r, k in rows:
    per = r.get("per_class")
    cells = " ".join(f"{per[j]['f1']:5.2f}" for j in range(7)) if per else " " * 47
    print(f"{label:58s} {r['accuracy']:9.3f} {r['macro_f1']:9.3f} | {cells}")
best_net = max(ARM_KEYS, key=lambda k: net["final"][k]["test"]["macro_f1"])
bs = net["baselines"]["always_standing"]["macro_f1"]
print(f"the notebook's target is a macro F1 around 0.9 on the test users. Best arm {names[best_net]}: "
      f"{net['final'][best_net]['test']['macro_f1']:.3f} against {bs:.3f} for always-standing "
      f"({net['final'][best_net]['test']['accuracy']:.3f} accuracy against "
      f"{net['baselines']['always_standing']['accuracy']:.3f})")
print(f"support on the test users: " + ", ".join(
    f"{CLASSES[j][:4]} {net['baselines']['always_standing']['per_class'][j]['support']}" for j in range(7)))
print(f"the class-average baseline is built from " + ", ".join(
    f"{CLASSES[j][:4]} {n}" for j, n in enumerate(net["baselines"]["centroid_windows_per_class"].values())))

fig, ax = plt.subplots(figsize=(9.5, 3.6))
labels, vals, cols = [], [], []
labels.append("always\nstanding")
vals.append(bs)
cols.append("#adb5bd")
labels.append("class average\nper window")
vals.append(net["baselines"]["nearest_centroid_window"]["macro_f1"])
cols.append("#6c757d")
for k in ARM_KEYS:
    labels.append(names[k].replace("_", "\n", 1))
    vals.append(net["final"][k]["test"]["macro_f1"])
    cols.append("#457b9d" if k != best_net else "#2a9d8f")
labels.append(f"{names[CA].split('_', 1)[0]}\n+ search")
vals.append(searched["as_is_macro_f1"])
cols.append("#e9c46a")
bars = ax.bar(range(len(vals)), vals, color=cols)
ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=1)
ax.axhline(0.9, color="k", ls=":", lw=1)
ax.text(len(vals) - 0.4, 0.905, "the notebook's target ~0.9", fontsize=8, ha="right")
ax.set_xticks(range(len(vals)), labels, fontsize=8)
ax.set_ylim(0, 1.02)
ax.set_ylabel("macro F1, test users")
ax.set_title("Task 2, test users (26-30): the baselines, the four arms, and the arm with the search", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "fig_results.png", dpi=130)
plt.close(fig)


print("\n" + "=" * 98)
print("3. where the mistakes are: the confusion matrix on the test users")
print("=" * 98)
show = list(dict.fromkeys([NAME_OF["A_plain"], CA]))
fig, axes = plt.subplots(1, len(show), figsize=(5.0 * len(show), 4.5))
for ax, k in zip(np.atleast_1d(axes), show):
    p, y = PRED[names[k]]["pred"], Yte
    m = y >= 0
    cm = np.zeros((7, 7), int)
    for t, pr in zip(y[m], p[m]):
        cm[t, pr] += 1
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(share, cmap="viridis_r", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j] / 1000:.0f}k\n{share[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=6, color="w" if share[i, j] > 0.45 else "k")
    ax.set_xticks(range(7), labels=[c[:4] for c in CLASSES], fontsize=7, rotation=90)
    ax.set_yticks(range(7), labels=[c[:4] for c in CLASSES], fontsize=7)
    ax.set_xlabel("predicted", fontsize=8)
    ax.set_ylabel("true", fontsize=8)
    ax.set_title(f"{names[k]}: {cm.sum() / 1000:.0f}k labelled samples", fontsize=9)
np.atleast_1d(axes)[0].set_title(np.atleast_1d(axes)[0].get_title() + "\n(counts, and share of the row)", fontsize=8)
fig.suptitle("Confusion on the test users - rows are the true class", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGS / "fig_confusion.png", dpi=130)
plt.close(fig)
for k in show:
    p, y = PRED[names[k]]["pred"], Yte
    m = y >= 0
    cm = np.zeros((7, 7), int)
    for t, pr in zip(y[m], p[m]):
        cm[t, pr] += 1
    off = cm.sum() - np.trace(cm)
    top = sorted(((cm[i, j], i, j) for i in range(7) for j in range(7) if i != j), reverse=True)[:4]
    print(f"  {names[k]:20s} {off} of {cm.sum()} labelled samples wrong ({off / cm.sum():.1%}); most of it: "
          + ", ".join(f"{CLASSES[i][:4]}->{CLASSES[j][:4]} {n}" for n, i, j in top))
print(f"      the matrix is drawn in fig_confusion.png; the same picture for the always-standing baseline "
      f"is a single filled column under standing")


print("\n" + "=" * 98)
print("4. one test recording: what it was, what the network said")
print("=" * 98)
r_show = 0
exp, user = TEST[r_show][2]["exp"], TEST[r_show][2]["user"]
acc = TEST[r_show][0]
truth = strip(Yte, r_show)
TRACE = ["#264653", "#2a9d8f", "#e76f51"]
fig, axes = plt.subplots(len(PRED) + 4, 1, figsize=(13, 1.15 * (len(PRED) + 4) + 1.6), sharex=True)
t = np.arange(len(truth)) / FS
for j, kk in enumerate("xyz"):
    axes[j].plot(t, acc[:, j], linewidth=0.5, color=TRACE[j])
    axes[j].set_ylabel(f"{kk} (g)", fontsize=7)
axes[3].imshow(lab_to_rgba(truth)[None], aspect="auto", extent=[0, t[-1], 0, 1], interpolation="nearest")
axes[3].set_yticks([])
axes[3].set_ylabel("true", fontsize=7)
for ax, nm in zip(axes[4:], PRED):
    ax.imshow(lab_to_rgba(strip(PRED[nm]["pred"], r_show))[None], aspect="auto",
              extent=[0, t[-1], 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel(nm.replace("_", " ", 1), fontsize=7)
axes[-1].set_xlabel("time (s)", fontsize=8)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in zip(CLASSES, COLORS)],
           loc="lower center", ncol=7, fontsize=8, frameon=False)
fig.suptitle(f"test recording exp{exp:02d} user {user} ({len(truth) / FS:.0f} s), shown first because it is "
             f"first - not chosen for its result; white = no label (the pauses between activities)",
             fontsize=10)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig(FIGS / "fig_timeline.png", dpi=130)
plt.close(fig)
for nm in PRED:
    p = strip(PRED[nm]["pred"], r_show)
    m = truth >= 0
    print(f"  {nm:20s} accuracy on this recording {np.mean(p[m] == truth[m]):.3f}, macro F1 "
          f"{f1_scores(truth[m], p[m])['macro_f1']:.3f}")
print(f"      drawn in fig_timeline.png ({N_REC} test recordings covered by the artifacts, this is number 1 "
      f"of them)")


print("\n" + "=" * 98)
print("5. the phone turned round in the pocket")
print("=" * 98)
print(f"{'worn as':22s} {'acc plain':>10s} {'acc searched':>13s} {'gain':>7s} | {'macro F1':>9s} "
      f"{'searched':>9s} | {'deg left off':>12s} {'<20 deg':>8s} {'switched':>9s}")
for nm, v in net["per_rotation"].items():
    print(f"{nm:22s} {v['as_is']:10.3f} {v['searched']:13.3f} {v['gain']:+7.3f} | "
          f"{v['as_is_macro_f1']:9.3f} {v['searched_macro_f1']:9.3f} | "
          f"{v['angle_left_off_deg']:12.0f} {v['undone_to_under_20deg']:7.0%} "
          f"{v['switched_away_from_real']:9.1%}")
axis_keys = [nm for nm in net["per_rotation"] if not nm.startswith("turned")]
rand_keys = [nm for nm in net["per_rotation"] if nm.startswith("turned")]
axv = [net["per_rotation"][nm] for nm in axis_keys]
rdv = [net["per_rotation"][nm] for nm in rand_keys]
print(f"mean over the {len(axv)} wearings a candidate can represent exactly: accuracy "
      f"{np.mean([v['as_is'] for v in axv]):.3f} plain, {np.mean([v['searched'] for v in axv]):.3f} searched")
print(f"mean over the {len(rdv)} continuous wearings none can represent: accuracy "
      f"{np.mean([v['as_is'] for v in rdv]):.3f} plain, {np.mean([v['searched'] for v in rdv]):.3f} searched")
print(f"nothing turned (the data as delivered): the search switched away from the plain reading in "
      f"{net['per_rotation']['x,y,z']['switched_away_from_real']:.1%} of the windows and moved accuracy "
      f"{net['per_rotation']['x,y,z']['gain']:+.3f}")
print("robustness without any search, accuracy as the phone is worn in each of the "
      f"{len(net['robustness_as_is'][names[ARM_KEYS[0]]])} stress wearings:")
for k in ARM_KEYS:
    v = net["robustness_as_is"][names[k]]
    print(f"  {names[k]:20s} as it comes {v[0]:.3f}; over the wearings min {np.min(v):.3f}, "
          f"mean {np.mean(v):.3f}, and {np.mean(np.array(v) < 0.5):.0%} of them below 0.5")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
xs = np.arange(len(net["robustness_as_is"][names[ARM_KEYS[0]]]))
for k in ARM_KEYS:
    axes[0].plot(xs, net["robustness_as_is"][names[k]], marker="o", ms=4, lw=1.2,
                 label=names[k] + ("" if k != CA else " (search arm)"))
sr = [net["per_rotation"][nm]["searched"] for nm in axis_keys + rand_keys]
axes[0].plot(xs, sr, marker="s", ms=5, lw=1.6, color="k", ls="--", label=names[CA] + " + search")
axes[0].set_xticks(xs, labels=[str(i) for i in xs], fontsize=8)
axes[0].set_xlabel("worn orientation (0 = as delivered, then axis-aligned turns, then random turns)", fontsize=8)
axes[0].set_ylabel("accuracy, test users", fontsize=8)
axes[0].legend(fontsize=7)
axes[0].set_title("how each model copes with a turned phone", fontsize=9)
v0 = net["per_rotation"]["x,y,z"]
axes[1].bar([0, 1], [v0["as_is"], v0["searched"]], color=["#457b9d", "#e9c46a"])
axes[1].set_xticks([0, 1], labels=["as it comes", "+ search\n(nothing turned)"], fontsize=8)
axes[1].set_ylim(0, 1.0)
axes[1].set_title("... and what looking costs when nothing is turned", fontsize=9)
for i, v in enumerate([v0["as_is"], v0["searched"]]):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "fig_rotations.png", dpi=130)
plt.close(fig)

for i, z in sorted(TURNED.items()):
    r_t = 0
    acc_t = TEST[r_t][0] @ np.asarray(z["rot"]).T                    # the axes as the turned phone reports them
    tr = strip(Yte, r_t)
    pl, se = strip(z["pred"], r_t), strip(z["searched"], r_t)
    cw = z["chosen"][z["rec"] == r_t]
    counts = sorted(((int(np.sum(cw == c)), c) for c in np.unique(cw)), reverse=True)[:5]
    fig, axes = plt.subplots(5, 1, figsize=(13, 5.0), sharex=True)
    for j in range(3):
        axes[0].plot(np.arange(len(acc_t)) / FS, acc_t[:, j], linewidth=0.5, color=TRACE[j])
    axes[0].set_ylabel("reported (g)", fontsize=7)
    for row, lab_v, nm in ((1, lab_to_rgba(tr), "true"),
                           (2, lab_to_rgba(np.where(pl < 0, -1, pl)), "read plain"),
                           (3, lab_to_rgba(np.where(se < 0, -1, se)), "read after the search")):
        axes[row].imshow(lab_v[None], aspect="auto", extent=[0, len(tr) / FS, 0, 1], interpolation="nearest")
        axes[row].set_yticks([])
        axes[row].set_ylabel(nm, fontsize=7)
    axes[4].step(np.asarray(z["start"])[z["rec"] == r_t] / FS, cw, where="post", lw=0.8, color="#264653")
    axes[4].set_yticks([c for _, c in counts], labels=[perm_name(PERMS[c]) for _, c in counts], fontsize=6)
    axes[4].set_ylabel("candidate kept", fontsize=7)
    axes[4].set_xlabel("time (s)", fontsize=8)
    m = tr >= 0
    fig.suptitle(f"the same recording worn as {z['name']}: plain {np.mean(pl[m] == tr[m]):.3f}, after the "
                 f"search {np.mean(se[m] == tr[m]):.3f} accuracy; the search settled on "
                 + ", ".join(f"{perm_name(PERMS[c])} ({n})" for n, c in counts[:3]), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / f"fig_turned_{i}.png", dpi=130)
    plt.close(fig)
    print(f"  turned case {i} ({z['name']}), recording 1 of the test users: accuracy "
          f"{np.mean(pl[m] == tr[m]):.3f} plain, {np.mean(se[m] == tr[m]):.3f} searched -> fig_turned_{i}.png")

print("\n" + "=" * 98)
print("6. what goes on the phone")
print("=" * 98)
raw = [k for k in ARM_KEYS if not CKPT[k][1]["canon"]]       # the arms the app can be fed
pick = max(raw, key=lambda k: net["arms"][k]["validation"]["macro_f1"])
robust = max(raw, key=lambda k: np.mean(net["robustness_as_is"][names[k]]))
print(f"the .onnx can only carry an arm that reads the raw axes the app feeds it: "
      f"{', '.join(names[k] for k in raw)}. {names[CA]} is the one the search was chosen on and it is NOT "
      f"one of them - canonicalizing a window is work the app does not do.")
print(f"export the best of those on validation - {names[pick]} ({net['arms'][pick]['validation']['macro_f1']:.3f} "
      f"validation macro F1) - with export_onnx.py: input acc 1 x 3 x {W}, output logits 1 x 7 x {W}, "
      f"model_info with the class names, units g, {FS} Hz, window {W}, axes x,y,z. The app divides the "
      f"browser's m/s2 by 9.80665 because the file says g.")
print(f"the search is NOT in the .onnx either - the app does one forward pass per window - so what runs on "
      f"the phone is one network with no second look. If the pocket is not the belt mount this data was "
      f"collected with, the honest alternative is {names[robust]}, which is robust by training: "
      f"{net['final'][robust]['accuracy']:.3f} accuracy as it comes against "
      f"{net['final'][pick]['accuracy']:.3f} for {names[pick]}, but "
      f"{np.mean(net['robustness_as_is'][names[robust]]):.3f} against "
      f"{np.mean(net['robustness_as_is'][names[pick]]):.3f} averaged over the turned wearings.")
print("then on the phone: load the .onnx, Start, phone on the waist, and the script the notebook suggests - "
      "stand 20 s, walk 30 s, sit 20 s, take the stairs, lie down on a bench - Stop, and download the CSV.")
