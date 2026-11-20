# Task 1 — ECG delineation: where are P, QRS and T?

Everything below was measured by the scripts in this folder; every number points at the log or the
JSON that printed it, so nothing here has to be taken on trust. Console output of every run is in
`logs/`, machine-readable results in `figs/results.json` (the lead-II model) and
`figs/results_final.json` (the final model).

**The reported convention.** A sample is judged if it lies inside the cardiologists' annotated span
plus a 200 ms buffer, and is otherwise ignored by both the loss and the metric. The same
predictions are also scored under three other conventions in every table, because the choice is ours
and it should not be allowed to carry the result.

## 1. Final values

Test split, 30 patients never used for any decision, macro Dice per wave (mean of the per-record
Dice), convention *span + buffer* (`figs/results_final.json`, `logs/improve.log`):

| wave | baseline | network, lead II, one model | **final network** | final − baseline |
| --- | --- | --- | --- | --- |
| P | 0.603 | 0.717 | **0.782** | **+0.179** |
| QRS | 0.858 | 0.939 | **0.942** | **+0.084** |
| T | 0.813 | 0.906 | **0.904** | **+0.090** |
| mean | 0.758 | 0.854 | **0.876** | +0.118 |

The network beats the baseline on all three waves — which is what the task asks for — under all four
conventions, not only the reported one (the `keeps` column, and its per-split breakdown — the test
split keeps 79.1 % under the reported convention — are in `logs/repair.log`):

| convention | what is judged | keeps | baseline P/QRS/T | final network P/QRS/T |
| --- | --- | --- | --- | --- |
| **span + buffer** | annotated span + 200 ms each end | 76.9 % | 0.603 / 0.858 / 0.813 | **0.782 / 0.942 / 0.904** |
| span, no buffer | exactly the annotated span | 72.9 % | 0.637 / 0.862 / 0.836 | 0.825 / 0.945 / 0.928 |
| signal edges | between the signal's own first and last QRS peak, +200 ms; no label decides the edge | 93.6 % | 0.552 / 0.780 / 0.745 | 0.725 / 0.850 / 0.819 |
| as parsed | every sample, as the notebook's reference numbers do | 100 % | 0.544 / 0.773 / 0.717 | 0.713 / 0.849 / 0.789 |

Sanity check that the metric is not generous: a predictor that answers *background* everywhere scores
**0.000 on every wave** (`logs/improve.log`, under the reported convention — and it follows for the
other three, since Dice has no overlap left to give credit for).

The QRS duration the notebook asks for: over all 288 matched beats of the test split the annotated
complexes are **100.2 ms** wide on average and the final network's are **99.5 ms**, a mean absolute
error of **9.2 ms** — fewer than five samples. (The single lead-II model alone: 99.3 ms and 9.95 ms,
`figs/results.json` — the same answer either way.) The network is not just placing the complex, it is
drawing it about as wide as the cardiologists did.

## 2. The task, as set

One class per sample — 0 background, 1 P, 2 QRS, 3 T — for lead II of LUDB 1.0.1 (200 patients,
10 s of a 12-lead ECG at 500 Hz, so one sample is 2 ms, every wave outlined by cardiologists). The
input is `(batch, 1, 5000)`, the output `(batch, 4, 5000)`, the loss cross-entropy with
`ignore_index=-1`, the metric per-wave Dice, and Part 1's `UNet`, `train` and `predict` are to be
reused unchanged. They are, in `part1.py`, imported rather than copied, so there is one copy of them
and it is the notebook's.

The notebook hands over a classical baseline said to reach *"about 0.74 for P, 0.89 for QRS and 0.85
for T"* and says the U-Net should beat it on all three. Section 5 explains why our reconstruction of
that baseline is considerably better than those numbers suggest, and lands below them.

The data cell ends with an instruction we took literally: *"Look at the plot closely, also at the
start and the end of the record."* At the start and the end of the record there are QRS complexes
with no label on them. Section 3 is what we did about that, and it is the single decision that moves
the numbers most.

## 3. Is the data what it claims to be?

From `logs/inspect.log` and `logs/audit.log` (`inspect_ludb.py`, `audit_ludb.py`).

**Sound.** 200 × 5000 signals and labels, 500 Hz, no NaN anywhere, labels exactly {0,1,2,3}. Every
wave's peak lies inside its own [onset, offset] (200/200 records), waves are listed in temporal
order, none runs past the record. The label array is a faithful encoding of the annotation triplets:
1401 P / 1817 QRS / 1642 T annotated waves produce exactly 1401 / 1817 / 1642 contiguous blocks in
`y`, with **0 fragments and 0 samples overwritten** by a later wave. Wave widths are physiologically
sane (median P 110 ms, QRS 92 ms, T 180 ms). The split is disjoint, covers all 200 records, and
leaves the class fractions nearly identical across splits (train .686/.077/.089/.149, valid
.682/.083/.089/.147, test .661/.084/.096/.159). No two records share a signal fingerprint.

**Needs knowing.**

* **Absolute amplitude is not in the data.** Every one of the 200 records spans exactly 1.000 mV
  peak-to-peak (the `.hea` gains differ, 1069–1950, so the storage was rescaled to full scale). A
  network cannot learn "the R wave is 1.2 mV" here, and none should have to.
* **Standardization exaggerates the low-amplitude records.** The notebook divides by each record's
  own std; in the flattest records that std is mostly drift, so record 125 ends up spanning −25.1 to
  +3.7 and 147 of the 200 records reach |x| > 6 somewhere (`logs/audit.log`); 0.87 % of all samples
  exceed |x| = 5 (`logs/ablate_preproc.log`). We measured
  whether taking it back out helps — saturating the extremes, or subtracting the slow component
  first. Both make the network slightly *worse* (§8), so the final model does no preprocessing at all
  beyond the notebook's own standardization.
* **14 wave symbols are silently dropped by the notebook's parser.** A wave must arrive as the
  triplet `( N )`; in records 94, 103 and 110 that triplet is malformed — the `N` is there, but the
  bracket that should open the wave is missing or has drifted to the far side of it. Those are 14
  real QRS complexes that never reach `y` at all, and they are the same beats as the interior ones in
  §4.
* **Record 103 has no annotated QRS in lead II at all** — 8 T waves and nothing else — while leads
  I, V1 and V6 of the same patient each have 9 complete QRS triplets. It is in the training split,
  so no score depends on it, but it is why one record can never teach you what a QRS looks like.
* **24 records contain no P wave**, 15 of them atrial fibrillation. A per-wave mean must skip the
  records that have nothing to say, or those 24 records are counted as failures of a wave that does
  not exist. `ludb_metric.py` does that and reports how many records actually voted (28 of 30 for P
  on the test split, 30 for QRS and T).
* The corpus is brady-brained: median heart rate 54–56 bpm, 25 of 200 patients sinus bradycardia.
  Only 77.5 % of QRS complexes are preceded by a P wave within 1.2 s, and the P-peak-to-R-peak
  interval has p5–p95 of 113–794 ms. "The P wave is always ~150 ms before the R" is a bad prior —
  which is exactly why a baseline that places waves at fixed distances from the R peak loses.

## 4. The label problem, and the repair

Taken literally, the labels say *background* at real QRS complexes.

* **413 of them at the record edges** — QRS-like peaks outside the annotated span, typically two per
  record (one before the first annotated onset, one after the last offset), because the
  cardiologists annotated only the complete beats. (399 at the stricter 50 % detector threshold, 413
  at 30 %; `logs/audit_unlabelled.log`.)
* **11 more inside the span**, in records 94 (3), 103 (7) and 110 (1) — all three in the *training*
  split, all three the records whose lead II annotation files the parser drops symbols from. Under a
  stricter predicate the count is 12, the extra one in test record 73, and `reconcile_interior.py`
  shows that beat is the same beat detected twice. `logs/reconcile_interior.log` has both counts at
  three detector thresholds.

Left alone, a U-Net that correctly marks those beats is punished for it twice — once in the loss,
which actively teaches it to answer background at a textbook QRS, and once in the metric.

**What we did:** give those samples the label `-1`, which `nn.CrossEntropyLoss(ignore_index=-1)`
skips and which the metric skips — exactly the padding convention of Part 1's protein example.
Everything outside [first onset − 200 ms, last offset + 200 ms] becomes `-1`, and inside the span a
±150 ms window around each unlabelled QRS-like peak becomes `-1` (that one is a known label error,
not a decision about where to look). The signal itself is never cut.

**Why masking and not cropping** — three reasons, all measured (`logs/audit_unlabelled.log`):

1. Cropping to the annotated span keeps 3640 of 5000 samples on average (min 2318, max 4411) — it
   throws away **26.9 % of the signal**, and gives every record a different length, so batching, the
   circular-shift augmentation and the whole `(200, 5000)` geometry change.
2. A crop **leaks the answer into the input**: a record that begins 30 ms after an annotated onset is
   itself a hint that a beat boundary is near. Masking keeps the neighbouring beats visible, which is
   also what tells the network which beat a P wave belongs to.
3. The buffer cliff. Trimming and putting a buffer back re-admits the unlabelled beats anyway:
   **0 ms → 11 beats, 200 ms → 13, 500 ms → 29, 1000 ms → 334.** A crop cannot be both wide enough to
   be safe and narrow enough to be useful.

**Why 200 ms even though 0 ms scores better.** A mask that stops exactly at the annotation makes the
score mean "accuracy where the annotators happened to look", and it is the one width that can be
chosen with an eye on the result. The 200 ms buffer costs us the two unlabelled beats that fall
inside it — records 94 and 103 — which are then judged as background and counted as false positives
**on purpose**. `repair_ludb.py` asserts this all holds: 0 unexplained beats left inside the span, and
the 2 buffer beats (records 94 and 103) left outside it, confirmed independently by
`reconcile_interior.py`'s last block — which finds those 2 at the 30 % and 40 % detector and 1 of them
at 50 %, the only way in which the two scripts' counts are not the same number.

**Does the repair change the model, or only the score?** `ablate_repair.py` answers that by training
the same architecture, seed and schedule twice — once on the raw labels, once on the repaired ones —
and then looking at what each does with the beats nobody annotated (test split, 288 annotated beats
and 60 unannotated edge beats):

| trained on | as parsed P/QRS/T | span + buffer P/QRS/T | marks annotated beats | marks unannotated edge beats |
| --- | --- | --- | --- | --- |
| raw labels (A) | 0.741 / 0.901 / 0.831 | 0.763 / 0.919 / 0.869 | **262/288 (91.0 %)** | 56/60 (93.3 %) |
| repaired labels (B) | 0.707 / 0.843 / 0.836 | 0.741 / **0.935 / 0.896** | **288/288 (100 %)** | 60/60 (100 %) |

Arm A only "wins" under the convention that treats the annotation error as the truth. Under our
reported convention B is better on all three waves, and behaviourally B finds **every** annotated
beat while A has learned to swallow roughly one in eleven. The contradictory labels were costing real
annotated beats, not just tidy points.

## 5. The baseline, and how it was obtained

The task prescribes it: find the R peaks with `find_peaks` on the band-passed signal, then put the P,
QRS and T waves at their median distances from the R peak, measured on the training patients.

**Step 0 — the rule proposed at the whiteboard**: *"the peak is anywhere the signal is above 60 % of
its maximum"*. Taken literally that is a bad detector here, for two reasons we could see from the
data rather than guess: (a) after per-record standardization the largest deflection is not the R peak
in the drift-dominated records, so `max|x|` is not the same quantity in every record; (b) the signal
stays above 60 % of its maximum for tens of samples, so the threshold finds *plateaus*, not complexes,
and needs a refractory gap to become a peak list. Implemented honestly — per-record threshold,
250-sample refractory, the same median windows — it scores on validation 0.673 / 0.874 / 0.776. It is
reported in every table as "your rule", because it is the thing a reader would write first.

**Step 1 — the median windows, measured on the 140 training patients only** (`logs/baseline_buffer200.log`):

| wave | onset | offset | width |
| --- | --- | --- | --- |
| P | −101 | −46 | 55 samples = 110 ms |
| QRS | −21 | +24 | 45 samples = 90 ms |
| T | +81 | +173 | 92 samples = 184 ms |

median R–R interval 450 samples = 900 ms.

**Step 2 — the detector, searched on the validation split** over band-pass band, threshold fraction,
refractory distance, smoothing and whether to scale the windows by the record's own R–R interval. The
winner is a 8–20 Hz band-pass, peaks at ≥ 40 % of the record's own envelope maximum, 150-sample
refractory, no extra smoothing, windows not RR-scaled — validation 0.679 / 0.900 / 0.810. The
neighbouring settings sit within 0.001 of it — the eight rows of the top of that table are all 0.796
— so this baseline is not perched on a lucky cell. Asked under each of the other three conventions
which detector it would have picked, **the same band-pass, threshold and refractory win all four**;
only the optional smoothing differs, and only under `signal edges` (σ = 4 instead of 0). That is the
check that the convention is not doing the work.

On the test split, once, against the ceiling of the whole approach:

| test, span + buffer | P | QRS | T |
| --- | --- | --- | --- |
| your rule, 60 % of max | 0.538 | 0.738 | 0.690 |
| **tuned baseline (what we compare against)** | **0.603** | **0.858** | **0.813** |
| ceiling: true R peaks + the same windows | 0.656 | 0.884 | 0.819 |

The ceiling row is the honest scale for the baseline: even handed the *annotated* R peaks, the fixed
window rule cannot pass 0.656 / 0.884 / 0.819. Everything else it lacks is what a per-sample
classifier can learn — the width of an individual complex, the T wave that does not sit 81 samples
after the R. The detector itself is good, incidentally: 0.965 sensitivity and 0.820 precision on R
peaks, against 0.788 / 0.805 for the 60 % rule.

**About the notebook's 0.74 / 0.89 / 0.85.** We cannot reproduce those numbers with the baseline the
notebook describes, on the split the notebook defines: our best reconstruction, given the true R
peaks, reaches 0.656 / 0.884 / 0.819. QRS and T are within reach of the quoted figures, P is not,
which is what you would expect if their P window was measured on the evaluation patients or defined
around the P peak rather than taken from the training set. We report our own baseline, measured
exactly as described and tuned on validation, and beat it on all three waves under all four
conventions.

## 6. The network, and every choice on validation

`UNet(1, 1, 4, widths)` from Part 1 — four classes out, one lead in — with the dataset handing over a
`(1, 5000)` float32 signal and `(5000,)` int64 labels, `-1` included, so the loss skips what the
metric skips.

**Depth is the interesting knob and it was chosen on the arithmetic, then confirmed on validation.**
Each block halves the resolution and its two 3-wide convolutions add ±2^level samples to what one
output position can see: 4 levels ≈ ±60 samples, 6 ≈ ±250, 7 ≈ ±500 — about one R–R interval, which
is what telling a P wave from the tail of the previous T wave needs. The sweep agrees (validation
macro Dice, mean of the three waves, augmentation on):

| depth | width | parameters | validation mean |
| --- | --- | --- | --- |
| 4 levels | [16, 32, 64, 128] | 0.2 M | 0.851 |
| 5 levels | […, 256] | 0.7 M | 0.885 |
| 6 levels | […, 512] | 2.7 M | 0.906 |
| **7 levels** | **[8, 16, 32, 64, 128, 256, 512]** | **2.7 M** | **0.917** |
| 7 wide | [16, 32, 64, 128, 256, 512, 1024] | 10.9 M | 0.913 |

Learning rate for the winning depth: 3e-4 → 0.885, 3e-3 → 0.896, 1e-2 → 0.909, **1e-3 → 0.917**
(kept; `train` returns the weights of the best validation epoch, so the schedule does not have to be
tuned as carefully as it would with a final-epoch model).

**Augmentation** is a random *circular* shift (labels rolled with it), a per-record gain of 0.8–1.25
and N(0, 0.1) noise. The gain matters because LUDB is stored scaled to 1 mV peak-to-peak, so a gain
change is the only amplitude variation this data can honestly be asked about; it also stops the
network latching on to absolute amplitudes. The shift is circular because the records are closed
10-second windows and the network sees the same seam at training time as at inference time. It helped
at 8 of the 10 depth × run pairs (mean +0.006) and at **every depth that was competitive** — 6 levels
+0.005/+0.010, 7 levels +0.007/+0.013, 7 wide +0.006/+0.012 in the two runs. The two exceptions are
both at the tiny 4- and 5-level nets (−0.002 and −0.006), in runs where those nets were 0.03 and 0.06
behind the winner regardless of augmentation.

**One thing this sweep exposed, and what we did about it.** Run the identical sweep twice and the
winner changes: the first full run preferred *7 wide* (0.911) over *7 levels* (0.909), the
confirmation run preferred *7 levels* (0.917) over *7 wide* (0.913) — `logs/train_full.log` against
`logs/train_confirm.log`. cuDNN's non-deterministic convolution backward plus ties on a 30-record
validation split move a run by more than the margin between architectures. Two consequences: the
architecture we keep is the *confirmation* run's (the weights on disk are `model_ludb.pt`, and the
numbers in §1 are the ones those weights produce), and every later comparison in this report is made
across **three seeds** with `cudnn.deterministic = True`, which is §8.

The lead-II model this selects reaches validation 0.886 / 0.952 / 0.914 against the baseline's
0.679 / 0.900 / 0.810, and on test 0.717 / 0.939 / 0.906 (§1, middle column).

## 7. What "show" means — the four deliverables

| the notebook asks for | where it is |
| --- | --- |
| the specification table | notebook cell 47 — **not written yet, deliberately**: it is the six rows the student fills in by hand (input, output, model, loss, metric, data). Every value those rows need is in §2 and §9 of this file, so it is a five-minute job and it is not the agent's to do |
| the test Dice of every wave next to the baseline's | §1, and `figs/dice_per_record_final.png` (one value per test record, as boxplots: baseline / lead-II network / final network) |
| one test record, true waves above predicted | `figs/one_test_record_final.png` — annotated, baseline, lead-II network, final network, four stacked panels; `figs/one_test_record.png` is the two-panel version for the lead-II model alone |
| the QRS duration in ms of one test patient, predicted against annotated | `figs/qrs_duration_final.png` — left: every beat of one test patient; right: all 288 matched test beats, 100.2 ms annotated vs 99.5 ms predicted, mean absolute error 9.2 ms |

Supporting figures: `figs/repair_edge_32.png` and `figs/repair_interior_103.png` (the label problem
before and after, on a test record and on the broken training record),
`figs/baseline_one_test_record.png`, `figs/curve_*.png` (every training curve of the depth sweep),
`figs/improve_arms.png` (the study of §8 with its error bars), `figs/worst_records.png` (the three
test records the network delineates worst: 95, 33 and 7).

## 8. The improvement study

Once the lead-II U-Net worked, four things that had never been varied were varied, all of them on
validation, all decided before the test split was touched again (`improve_ludb.py`,
`logs/improve.log`). The architecture, learning rate, augmentation and 120-epoch schedule were left
alone: they had already been chosen on these same 30 records, and choosing them twice would be
tuning twice on one small split.

**Which leads.** The annotations are made on lead II, which is why the task says lead II — but a P
wave is often largest in V1 or V5, so it was worth measuring instead of assuming. Three seeds for the
two arms that survived the rounds, one seed for everything else that was tried:

| arm | seeds | validation mean | sd |
| --- | --- | --- | --- |
| **lead II, cross-entropy + soft-Dice** | 3 | **0.908** | **0.001** |
| II + V1 + V5, cross-entropy | 3 | 0.903 | 0.006 |
| lead II, CE + Dice + weights | 1 | 0.908 | — |
| lead II, cross-entropy | 1 | 0.906 | — |
| all 12 leads, cross-entropy | 1 | 0.904 | — |
| II + V1 + V5, CE + weights | 1 | 0.903 | — |
| all 12 leads, CE + Dice | 1 | 0.902 | — |
| II + V1 + V5, CE + Dice | 1 | 0.898 | — |

Round 1 ran all six lead × loss combinations on seed 0; round 2 added class weights to the two best
arms; round 3 gave the two survivors seeds 1 and 2. Only the two rows with three seeds decided the
choice — the one-seed rows are there to show what was tried and that nothing was thrown out for
wanting a look.

**Multi-lead does not help.** Three leads are worth −0.005 against one, with six times the
seed-to-seed spread; twelve leads are no better still. The cardiologists annotated lead II, and the
extra leads mostly add variance about a decision already made there. This is the result that keeps the
submission honest: the task's own input definition is also the better one, so we did not have to
trade comparability for score.

**Which loss.** Cross-entropy weights every sample alike while the metric is Dice per wave and P is a
tenth of the trained-on samples. Two fixes were tried: class weights 1/√frequency (which came out
[0.539, 1.309, 1.214, 0.939] for [background, P, QRS, T]) and a soft-Dice term — the metric itself,
differentiated, with the `-1` samples out of both numerator and denominator and background excluded
from the mean so the network cannot buy Dice by predicting nothing. **Weights do nothing**: 0.903
against 0.910 for II + V1 + V5, and 0.908 against 0.909 for lead II, all four on seed 0, and the
first of those 0.903 matching the three-seed 0.903 of that arm above is a coincidence. **The Dice
term is worth about +0.002**, which is honest only as far as
that: the arm carrying it was one of the two promoted to three seeds and did not lose, but plain
cross-entropy was measured on one seed, so we cannot claim the effect is real. Kept, with that
caveat.

**The seeds, which is the point of the whole exercise.** Lead II + Dice: **0.908 ± 0.001** over three
seeds. That spread is the yardstick for everything else in this section — the multi-lead gap (0.005)
clears it, the Dice term (0.002) does not, and the earlier architecture flip (§6, 0.002 to 0.004
between arms that changed places) was entirely inside it.

**Inference-time options**, tried without retraining anything:

| inference | validation mean |
| --- | --- |
| one model, one look | 0.909 |
| one model, softmax averaged over 5 circular shifts | 0.911 |
| **three independently trained models averaged** | **0.914** |
| three models + 5 shifts | 0.914 |
| any of the above, with predicted runs under 10 ms merged into their neighbour | ≤ the same, never better |

Averaging three seeds is the only inference-time change worth having: **+0.005 on validation, +0.023
on test** (mean of the three waves: 0.853 for a single seed → 0.876 for three). The individual seeds
land at 0.853, 0.872 and 0.862 on test, so the ensemble is also what stops the reported number from
depending on which model happened to win a coin flip. Shift-averaging is worth ~+0.002 on validation
and was not selected; merging short predicted runs is worth 0.000 — the network does not produce many
islands, so there is nothing for a post-processor to clean. Between the 0.914 options the code takes
the first in its own order, so "three models, no shifts" rather than the reverse is a tie broken by
insertion order, not by evidence.

**Classical ECG preprocessing, measured rather than assumed** (`ablate_preproc.py`,
`logs/ablate_preproc.log`). §3 noted that standardization lets record 125 span −25.1 to +3.7, which
looks like a preprocessing bug waiting to be fixed. The chosen arm was retrained twice on each of four
input transforms, same everything else:

| input | what was done to it | validation mean (2 seeds) | against no preprocessing |
| --- | --- | --- | --- |
| **none** | as the notebook hands it over | **0.909** | — |
| clip | saturated at ±5 (0.87 % of samples touched) | 0.904 | −0.004 |
| both | clip and the median below | 0.902 | −0.007 |
| wander | 300 ms running median subtracted, then re-standardized | 0.897 | **−0.011** |

**Nothing helps, and the worst of them is the most standard one.** Baseline-wander removal by a
long median filter is in every ECG preprocessing textbook, and it is aimed at *QRS detectors* — where
it earns its place, because the QRS is the high-frequency part of the signal. Here the task is the
opposite one: the P and T waves *are* low-frequency content, and a 300 ms running median sits right on
top of them, so subtracting it removes part of what we are asked to outline. Saturation costs less
(−0.004, about the 0.004 seed spread of this study) and tells a similar story: those extreme samples
are not noise the network has to be protected from, they are information about where the artifacts
are. The final model therefore does **no** preprocessing beyond the notebook's own standardization —
a claim this table earns rather than asserts.

## 9. What the final model is

```
UNet(1, 1, 4, [8, 16, 32, 64, 128, 256, 512])   2.72 M parameters, lead II in, 4 classes out
loss: cross-entropy (ignore_index=-1) + soft-Dice over P, QRS, T
data: per-record standardized lead II, masked outside the annotated span + 200 ms
augmentation: circular shift, gain 0.8-1.25, noise N(0, 0.1)
120 epochs, Adam lr 1e-3, weights of the best validation epoch
inference: three such models trained with seeds 0, 1, 2; probabilities averaged; argmax
```

`model_final_seed0.pt`, `model_final_seed1.pt`, `model_final_seed2.pt`; the single lead-II reference
model is `model_ludb.pt`. `improve_ludb.py` re-loads `model_ludb.pt` and asserts it still reproduces
`figs/results.json` (it does, to all three decimals: 0.717 / 0.939 / 0.906), so the recorded numbers
are checked rather than trusted.

## 10. Limits, stated plainly

* **30 test records is a small yardstick.** A difference of 0.01 on this split is worth roughly one
  patient, and the seed spread we measured (±0.008 on test) is of that order. The gap to the baseline
  (+0.18 / +0.08 / +0.09) is far outside it; the differences between our own variants mostly are not.
* **Macro Dice over different numbers of records is not the same average.** The final model's P score
  of 0.782 averages 28 records, the single model's 0.717 averages 29 — the single model predicted a
  P wave in one test record that has none annotated, and the ensemble did not. Micro Dice, which
  pools samples instead of records, is unaffected and tells the same story (P 0.806 → 0.843), so the
  P improvement is real; but the macro columns are not strictly comparable to each other across
  models.
* **The 200 ms buffer is a judgement call**, made worse-on-purpose rather than better-on-purpose, and
  reported under three alternative conventions precisely because it is a call and not a measurement.
* **The T wave did not improve** from the lead-II model to the final one (0.906 → 0.904, micro
  0.906 → 0.904). Everything the ensemble bought was in P.
* The reference numbers in the notebook (§5) cannot be reproduced by our reconstruction of the
  baseline they describe, so "beats the baseline" here means our baseline, which is the harder of the
  two we could have quoted.
* Atrial fibrillation records — where no P wave exists at all, 15 of them — are scored by a metric
  that simply does not vote on them. A clinician would want to know that a "0.782 for P" says nothing
  about whether the network invented P waves in fibrillation; the per-record boxplots in
  `dice_per_record_final.png` are the closest thing to that answer here.

## 11. Reproducing it

`README.md` has the exact command lines and their order, the dependency notes (wfdb ≥ 4.3.1; 4.1.x
breaks against NumPy 2), and one table per script. Roughly: parse → inspect → audit → repair →
baseline → train → improve → ablate. The GPU runs are minutes each on a laptop RTX 4060 (a full
120-epoch model is 20–30 s at this size; the whole improvement study is about eight minutes); with
`--quick` everything above runs at 6 epochs and one seed in under two minutes, and its numbers mean
nothing.
