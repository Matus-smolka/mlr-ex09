# Task 1 — ECG delineation: where are P, QRS and T?

Per-sample classification of lead II of the Lobachevsky University ECG database (LUDB 1.0.1):
200 patients, 12 leads, 500 Hz, 10 s, one of four classes per sample
(0 background, 1 P, 2 QRS, 3 T). 140 / 30 / 30 patients for train / validation / test, the split the
notebook's data cell defines (seed 0), reproduced exactly by `parse_ludb.py` so the numbers stay
comparable with the reference in the notebook.

Part 1's `conv_block`, `Encoder`, `UNet`, `train` and `predict` are reused **unchanged**, in
`part1.py`, which every training script imports.

## Dependencies

Python 3.12, `numpy`, `scipy`, `matplotlib`, `wfdb` (4.3.1 or newer — 4.1.x breaks against NumPy 2
with `OverflowError: Python integer 256 out of bounds for uint8`), `torch` (any CUDA build; the run
used torch 2.14.1+cu126 on an RTX 4060 Laptop, and the whole sweep takes a couple of minutes).

## How to run

From the repository root, in this order. Every script takes its paths as arguments and writes its
figures and numbers under `task1_ludb/`, so nothing needs editing.

```
LUD=data/ludb/lobachevsky-university-electrocardiography-database-1.0.1
py -I task1_ludb/parse_ludb.py         "$LUD" task1_ludb/cache/ludb.npz
py -I task1_ludb/inspect_ludb.py       task1_ludb/cache/ludb.npz task1_ludb/cache/ludb_records.csv
py -I task1_ludb/audit_ludb.py         "$LUD" task1_ludb/cache/ludb.npz
py -I task1_ludb/audit_unlabelled.py   task1_ludb/cache/ludb.npz task1_ludb/figs
py -I task1_ludb/repair_ludb.py        task1_ludb/cache/ludb.npz task1_ludb/cache/ludb_masked.npz task1_ludb/figs 200
py -I task1_ludb/reconcile_interior.py task1_ludb/cache/ludb.npz task1_ludb/cache/ludb_masked.npz
py -I task1_ludb/baseline_ludb.py      task1_ludb/cache/ludb_masked.npz task1_ludb/figs
py      task1_ludb/train_ludb.py       task1_ludb/cache/ludb_masked.npz task1_ludb/figs
py      task1_ludb/improve_ludb.py     task1_ludb/cache/ludb_masked.npz task1_ludb/figs
py      task1_ludb/ablate_repair.py    task1_ludb/cache/ludb_masked.npz task1_ludb/figs
py      task1_ludb/ablate_preproc.py   task1_ludb/cache/ludb_masked.npz task1_ludb/figs
```

`reconcile_interior.py` runs after `repair_ludb.py` because its last check - is any unlabelled beat
still asserted to be background? - can only be asked of the repaired labels. `baseline_ludb.py` needs
no training but does need the repaired cache; the last four need `figs/results.json` and
`model_ludb.pt` from `train_ludb.py` (and `improve_ludb.py`, for the preprocessing ablation).

`parse_ludb.py` needs the LUDB archive extracted (PhysioNet, `wget -c -r -np -R "index.html*"
https://physionet.org/files/ludb/1.0.1/`); the folder it lands in is the `$LUD` above, one level
inside `data/ludb/`. `--quick` runs the two training sweeps with 6 epochs and one seed, for a fast
check that everything works — its numbers mean nothing, its exit code does.

`train_ludb.py` must run before `improve_ludb.py` and `ablate_repair.py`: both read the recipe and
the reference numbers out of `figs/results.json` and the weights out of `model_ludb.pt`, and
`improve_ludb.py` asserts that what it recomputes agrees with what is recorded there.

Everything the training scripts measure is in `task1_ludb/figs/results.json` (the lead-II model) and
`figs/results_final.json` (the final model after the improvement study); their console output is in
`task1_ludb/logs/`; the weights are `model_ludb.pt` and `model_final_seed*.pt`. `TASK1.md` walks
through the numbers, how each was obtained and why each choice was made.

## What the scripts are

| file | what it does |
| --- | --- |
| `parse_ludb.py` | reads LUDB exactly as the notebook's data cell does, caches `ludb.npz` + `ludb_records.csv` |
| `inspect_ludb.py` | validity report: structure, signal quality, annotation completeness, wave geometry, label/annotation agreement, class balance, rhythms, split integrity |
| `audit_ludb.py` | the anomalies: 1 mV storage, dropped annotations, wave-width and R-R outliers |
| `audit_unlabelled.py` | finds QRS complexes with no annotation, splits them into edge and interior, and measures what a crop with a buffer would let back in |
| `reconcile_interior.py` | the exact count of interior unlabelled beats under both predicates and three detector thresholds, and — given the repaired cache — how many are still asserted to be background (0 inside the span, 2 in the buffer) |
| `repair_ludb.py` | the label repair: unannotated beats get label `-1`, ignored by the loss and the metric |
| `ludb_metric.py` | per-wave Dice (macro and micro) with the `-1` ignore mask, plus the QRS run and duration measures — one scorer for baseline and network |
| `classical.py` | the baseline detector: R peaks, then the waves at their median distances |
| `baseline_ludb.py` | searches the baseline on validation, scores it on test under all four conventions |
| `part1.py` | Part 1 of the notebook verbatim: `conv_block`, `Encoder`, `UNet`, `train`, `predict` |
| `train_ludb.py` | the 1D U-Net on lead II: dataset, sweep of the choices on validation, test split once, all four figures |
| `improve_ludb.py` | the improvement study: which leads, which loss, three seeds against the run-to-run spread, and the inference-time options; test split once, at the end |
| `ablate_repair.py` | trains once on the raw labels and once on the repaired ones, and shows what each does with the beats nobody annotated |
| `ablate_preproc.py` | whether classical ECG preprocessing (saturating the extremes, removing the 300 ms baseline wander) helps the chosen arm. It does not; the wander filter costs the most |

## The label repair, in one paragraph

LUDB annotates only the complete beats, so 413 QRS complexes at the record edges and 11 more inside
three records (94, 103, 110 — all train, all with lead II annotation files the parser drops symbols
from) carry no label. Taken literally, the labels say "background" at a textbook QRS complex, so a
network that finds those beats is trained and scored to be wrong about them. `repair_ludb.py` gives
those samples the label `-1`, which `nn.CrossEntropyLoss(ignore_index=-1)` skips and the metric
skips, exactly like the padding of Part 1's protein example. Everything outside the annotated span
plus a 200 ms buffer is ignored; inside the buffer the file's own labels stand, so an invented beat
is still a false positive. Cropping the record instead would leak the answer into the input (a
record that starts 30 ms after an annotated onset is a hint that a beat boundary is there), give
every record a different length, and any buffer wider than ~300 ms lets the unlabelled beat back in
anyway — measured, not assumed.

## Which samples count — four conventions, reported together

`span + buffer` is what we report; the others are printed next to it so the choice is visible and the
comparison cannot be accused of depending on it. The detector sweep picks the same baseline
configuration under all four, and the network beats it under all four.

| convention | what is judged | keeps |
| --- | --- | --- |
| `span + buffer` | the annotated span plus 200 ms at each end | 76.9 % |
| `span, no buffer` | exactly the annotated span | 72.9 % |
| `signal edges` | between the signal's own first and last QRS peak, +200 ms — no label decides the edge | 93.6 % |
| `as parsed` | every sample, as the notebook's reference numbers do | 100 % |
