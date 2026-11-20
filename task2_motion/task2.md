# Task 2 — Activity from the phone's accelerometer (1D, a class for every sample)

Data: UCI HAPT (Reyes-Ortiz et al.), 30 people with a smartphone on the waist, 3-axis acceleration in g
at 50 Hz, 61 recordings, every sample labelled with one of six activities or a transition, `-1` in the
pauses between them. Split by person: users 1-20 train, 21-25 validate, 26-30 test.

## 1. Final values

Two numbers, because the task asks for two things: a model trained normally (the baseline), and that model
made to survive the phone being turned round in a pocket.

| | test accuracy | test macro F1 | note |
| --- | --- | --- | --- |
| notebook's baseline: always answering standing | 0.176 | 0.043 | what the notebook says to beat |
| second baseline: nearest class average, one answer per window | 0.693 | 0.651 | no network, the best the app could do by hand |
| **baseline asked for: `UNet(1,3,7,[16,32,64,128])`, trained normally** | **0.918** | **0.918** | `A_plain`; the notebook's target was macro F1 around 0.9 |
| the same idea, made orientation-proof: `D_gravity_aligned` + the axis search | 0.839 | 0.854 | **0.839 on all ten wearings**, against 0.547-0.841 for the same network read plain; costs 0.002 when nothing is turned |
| for the phone, robust by training instead: `B_rot24` | 0.778 | 0.774 | 0.694 mean over the turned wearings, where `A_plain` averages 0.330 |

The trade the whole task turns on, in one line: **training normally buys 0.918 and throws it away the
moment the phone is not mounted the way the data was collected** (0.330 averaged over ten turned wearings,
0.074 at worst), and everything done about that costs between 0.08 and 0.14 of that accuracy — which is
what the comparison in §8 is for.

Deliverables: `activity_model.onnx` (the baseline, 0.74 MB, input `acc` 1x3x256, output `logits` 1x7x256)
and `activity_model_robust.onnx` (the robust pick); the specification table in §2 and §7; the figures in
`figs/`; this report. One deliverable is not here: the recording of your own made in the course app (§7).

## 2. The task, as set

Train a 1-D U-Net that labels **every sample** of the acceleration with one of seven classes, report it
against the baseline the notebook names (always answering standing), and export it to ONNX so it runs on
a phone in the course app. The part this exercise adds on top: account for **the phone being turned
round in a pocket** — when the reading does not look like any activity the model knows, try switching the
axes around, and keep a switch only if the window then matches better.

The specification table (notebook cell 53), from what was actually run:

| | answer |
| --- | --- |
| network input (shape) | `(batch, 3, 256)` — three axes in g at 50 Hz, 5.1 s per window |
| network output (shape) | `(batch, 7, 256)` — seven logits **per sample**, not one answer per window |
| last layer | `nn.Conv1d(16, 7, 1)`, a 1x1 convolution; no activation after it |
| training loss | `CrossEntropyLoss(ignore_index=-1)` |
| evaluation metric | macro F1 over the 7 classes on the test users, accuracy alongside; `-1` excluded |
| baseline | always answering standing: test accuracy **0.176**, macro F1 **0.043** (notebook: "about 0.18 / about 0.04"); a second, stronger baseline in §5 |

Two things the table is doing that are worth stating: the output is per **sample**, because the labels are
per sample and a window that straddles a boundary has no single answer; and the loss ignores `-1`, because
27% of the samples are the pauses between activities, where nobody knows what the person was doing.

## 3. Is the data what it claims to be?

`parse_motion.py` reproduces the notebook's parse line for line, then measures the two things that parse
silently assumes: that label rows belong to one experiment (they are selected by experiment number), and
that rows painted later do not destroy earlier labels (measured: **0** labelled samples overwritten over
all 61 recordings, and **0** painted past the end of a file). The parse then reproduces the notebook's own
printout exactly — `labels -1,0-6 -> [307158 122091 116707 107961 126677 138105 136865 67208]`.

| claim in the description | what the files say |
| --- | --- |
| acceleration in **g** | over 360 still-class stretches the measured acceleration is **1.020 g** (sd 0.013, range 0.990-1.046). A phone at rest reads 1 g, so the files are in g — and the browser's sensor gives m/s², so the app has to divide by 9.80665. The exported file says `"units": "g"` for exactly this reason. |
| **50 Hz** | 126 walking stretches repeat every **1.12 s** (10th-90th pct 1.02-1.22 s) and their spectrum peaks at **1.67 Hz** — twice that, the second step of the stride. A 1.1 s stride and a 3.6 s sit-to-stand are human; read at 100 Hz they would be a 2.2 s stride and a 7.2 s stand up, and slower again at 200 Hz. |
| 61 recordings, 30 people | 61 files, 1,122,772 samples = **374 min**, users 1-30; 29 users appear twice, user 10 three times. |
| every sample labelled | **72.6%** labelled; **27.4%** carry `-1`. People do not start and stop on cue: 633 pause stretches, median 5.6 s, longest 340 s, 511 of them between two activities, plus a median of 8 s before the first label and **337 s after the last**. |
| nothing broken | all samples finite, values in [-2.010, 2.004], \|a\| never above 3.39 g (nothing saturated). 65,338 samples (5.8%) repeat the previous one on all three axes — but 63,641 of those are a single sample held for one extra sample, only 104 runs are longer than 3, and the share is the same 5.5-6.1% in every class and every recording. That is the phone's uneven timestamps landed on a regular grid, not a sensor losing movement. |

The classes are balanced by design (13-17% each) except **transition**: 8.2% of the labelled samples, in
stretches of **3.6 s** at the median against ~20 s for an activity, and six different movements merged
into one label. Macro F1 is therefore decided by the transition class, which is why the notebook asks for
macro F1. The protocol is nearly the same in every recording (3 orderings, the commonest in 58 of 61), and
the two recordings that show only 3-4 classes (`exp20`, `exp21`, both user 10) are early stops, not a
parse loss.

Windows: `W = 256` samples (5.1 s), the length the course app uses. Non-overlapping for training —
2836 full windows on the 41 training recordings, of which **463 hold no label at all** and are left out
(§10). At evaluation a recording is cut into full windows with the last one slid back instead of padded,
so every sample is answered exactly once, always from a full window of real signal: 772 such windows over
the 10 test recordings. 12.3% of them straddle a boundary between two activities.

## 4. How the phone is worn — the question the task is really about

The mean direction of gravity in the phone's own axes, pooled over the recordings that do that class:

```
sitting   [0.98  0.133 0.148]      sitting-standing  19 deg
standing  [0.987 -0.158 -0.013]    sitting-lying     74 deg
lying     [0.078 0.778 0.624]      standing-lying    93 deg
```

Within one class the phones point a median of 8-18° apart, and the same person's two recordings a median
of 9.4° apart (worst 170.8°, user 10 lying down). **So HAPT does not contain the pocket-turning this task
is about** — the phone was belt-mounted at the waist, nearly the same way on everyone. Any honest test of
an orientation-aware method therefore has to manufacture the turning: held-out recordings are rotated by
hand, the target untouched, and the model is asked to recover the answer.

How much of the task lives in the axis order was measured before any network was trained, with a
nearest-class-average rule on single-class test windows:

| rule | as delivered | best / worst of the other 23 | what turning costs |
| --- | --- | --- | --- |
| gravity direction, 3 still classes | 0.848 acc / 0.849 macro F1 | **0 of 23 beat the real orientation** | median −0.553 accuracy, worst 0.047 |
| mean and sd of each axis, 7 classes | 0.668 / 0.656 | 0 of 23 | median −0.532, worst 0.021 |

The axis order carries most of what separates sitting from standing from lying. That sizes both the risk
(a model that leans on the axis order collapses when the phone is turned) and the headroom (a search over
the 24 orientations has something to find).

The 24 candidates are the proper rotations that map the device's axes onto its own — 3! orders × 2³ signs,
halved because half of those are mirror images, which no turning of a phone produces. Two facts about
them, both checked in code: `canonical_rotation`, the textbook "rotate the window until gravity points
along +x", does make the along-gravity row independent of the turn — but it leaves the **roll about
gravity free**, so it is two thirds of a fix, and it also throws away the one angle that separates sitting
from standing (19°) and cannot be exported, since the app feeds raw axes. And three of the plausible-
looking names one would type for a stress set (`y,-x,-z`, `x,-z,-y`, `-y,-z,-x`) are those mirror images:
the stress rotations are built by composing quarter-turns instead.

## 5. The baselines

| baseline | accuracy | macro F1 | how |
| --- | --- | --- | --- |
| always answering standing | 0.176 | 0.043 | the notebook's; standing is 17.6% of the labelled test samples and that is all it has to offer |
| nearest class average, one answer per 256-sample window | **0.693** | **0.651** | the mean and sd of each axis, matched to the closest class average; per class F1 walk 0.54, upst 0.38, down 0.71, sitt 0.80, stan 0.83, lyin 0.85, tran 0.44 |

The second one is worth having because it is what the app can do without a network, and because its weak
spots are structural: its class averages come from the training windows that hold a single class
(walk 401, upst 373, down 347, sitt 224, stan 307, lyin 259, **tran 33**), and a transition runs 181
samples at the median against a 256-sample window — so the transition average is the thinnest of the seven
and is what a window that straddles a boundary drifts towards, because a boundary is the loudest thing in
the window. A per-window rule can only ever give one answer for a window; the network answers each sample
of it separately.

## 6. The network, and the four arms

`part1.UNet(1, 3, 7, [16, 32, 64, 128])` — 170k parameters, Part 1's code unchanged. Adam, batch 32,
lr 1e-3, 400 epochs, best-validation-loss weights kept. Four arms, all scored on the validation users
before anything was chosen:

| arm | what is different | why it is in the comparison |
| --- | --- | --- |
| A `A_plain` | nothing — trained plainly on the axes as delivered | **the baseline the task asks for**, and the only thing that ends up in the `.onnx` |
| B `B_rot24` | every training window turned to a random one of the 24 axis-aligned orientations | robustness learned instead of searched; reads raw axes, so it is app-exportable |
| C `C_rotSO3` | every training window turned to a random *continuous* orientation (Haar SO(3)) | the same idea without the "multiples of 90°" assumption |
| D `D_gravity_aligned` | every window canonicalized so its own gravity points at +x | the textbook approach, included to be argued with |

On top of whichever arm is best, the search the task asks for: read the window 24 more times, once per
candidate orientation, score each reading by how well the network matched it (mean per-sample top-1
probability; mean log-probability and the window's majority share as alternatives), and keep a switch away
from the plain reading only if it beats it by a margin — the "only if it matches better", where the margin
is how much the model has to dislike the plain reading before the axes get shuffled. Margin, score and
whether to decide per window or per recording are all chosen on the **validation** users; the test users
are scored once, after that choice.

## 7. Results, and the deliverables the notebook asks for

Test users 26-30, 10 recordings, 196,733 samples, 145,380 of them labelled. Every number below is
recomputed by `report_motion.py` from the saved prediction files and asserted against the training run's
own JSON, so the tables and the artifacts cannot quietly disagree (`fig_results.png`).

| model | accuracy | macro F1 | walk | upst | down | sitt | stan | lyin | tran |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always standing | 0.176 | 0.043 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 |
| nearest class average, per window | 0.693 | 0.651 | 0.54 | 0.38 | 0.71 | 0.80 | 0.83 | 0.85 | 0.44 |
| **`A_plain`, trained normally** | **0.918** | **0.918** | 0.99 | 0.99 | 1.00 | 0.79 | 0.81 | 0.98 | 0.85 |
| `B_rot24` | 0.778 | 0.774 | 0.98 | 0.97 | 0.99 | 0.18 | 0.61 | 0.87 | 0.81 |
| `C_rotSO3` | 0.679 | 0.688 | 0.98 | 0.96 | 0.97 | 0.00 | 0.64 | 0.42 | 0.84 |
| `D_gravity_aligned` | 0.841 | 0.856 | 0.99 | 0.99 | 1.00 | 0.58 | 0.80 | 0.77 | 0.87 |
| `D` + the orientation search (nothing turned) | 0.839 | 0.854 | 0.99 | 0.99 | 1.00 | 0.58 | 0.80 | 0.76 | 0.86 |

Support: walk 20,345, upst 20,318, down 19,073, sitt 24,307, stan 25,539, lyin 25,790, tran 10,008
labelled test samples. The baseline model clears the notebook's "around 0.9" and takes the transition
class from 0.44 (the per-window rule) to 0.85, which is where macro F1 is actually won; nothing else in
the table beats it as it comes.

**Where the mistakes are** (`fig_confusion.png`, rows = true class). `A_plain` gets 11,938 of 145,380
labelled samples wrong, 8.2%, and **74% of those are one pair**: sitting answered as standing 5,481 times
and standing as sitting 3,372. Everything else is a trickle (tran→lyin 631, tran→sitt 554). That pair is
the 19° tilt of §4 — a phone in a sitting pocket hangs at a different angle than in a standing one, and
the two classes sit 19° apart in gravity direction, so the errors are not noise, they are the boundary.
The always-standing baseline's matrix is one filled column; `D`'s spreads further (23,154 wrong, 15.9%)
and adds sitt→lyin 6,322 and lyin→sitt 4,918, which is what canonicalization costs: it removes roll, and
roll is part of how sitting and lying differ.

**One recording, read end to end** (`fig_timeline.png`, test recording 1, users 26-30): the signal, the
true labels, and the four arms' per-sample answers on the same time axis. `A_plain` 0.877 accuracy / 0.891
macro F1 on it, `B` 0.789, `C` 0.638, `D` 0.710. Visible in it: the clean walk/upstairs/downstairs
stretch, the sitting-standing boundary where `A_plain` hesitates, and the transition samples, which come
out as short flurries of the wrong class rather than as transitions — six different movements sharing one
label means the network has no consistent thing to predict there.

**On the phone.** `export_onnx.py` exports A_plain: input `acc` `1×3×256` float32, output `logits`
`1×7×256`, `model_info` metadata with the seven class names, `"units": "g"`, `sample_rate` 50, `window`
256, `axes` `[x,y,z]`; 0.74 MB; `onnx.checker` passes, and on 200 real test windows the ONNX and PyTorch
agree on **100.0000%** of the 36,391 labelled samples. The batch axis is fixed at 1 (what the app sends);
the net itself is fully convolutional and answers for any length divisible by 8. `export_onnx.py` refuses
an arm that needs pre-processing, which is why `D_gravity_aligned` — the one the search was chosen on —
cannot be exported at all: the app feeds raw axes and does one forward pass per window. `activity_model_robust.onnx` carries `B_rot24` for anyone whose pocket is not a belt mount (0.778 as it comes, 0.694
turned, against 0.918 / 0.330 for A_plain). Then, in the app: load the `.onnx`, Start, phone on the waist,
and the script the notebook suggests — stand 20 s, walk 30 s, sit 20 s, take the stairs, lie down on a
bench — Stop, download the CSV. **That recording is still owed**; it is the one measurement here that
cannot be made from the shipped data, and note that your own phone in your own pocket is exactly the case
`A_plain` is confident about and wrong in: expect it to work well clipped upright and badly in a trouser
pocket, which is what the robust file is for.

## 8. The turned phone: what the search costs and what it buys

HAPT was collected with the phone mounted nearly identically on everyone (§4), so the turning is
manufactured: each held-out recording is rotated by a known rotation before the model sees it, the labels
untouched. Ten wearings — seven axis-aligned (a candidate can undo them exactly) and three continuous
random orientations in SO(3) (none can). The setting — arm, score, pooling, margin — was chosen among 64
combinations on the **validation** users only; the test users were scored once afterwards. Validation
picked `D_gravity_aligned`, decided per recording, scored by mean top-1 probability, margin **0.00** (the
best-fitting of the 24 readings always wins; "matches better" taken literally).

Test users, all 196,733 samples each row:

| worn as | accuracy plain | accuracy searched | gain | macro F1 plain → searched |
| --- | --- | --- | --- | --- |
| `x,y,z` (nothing turned) | 0.841 | 0.839 | **−0.002** | 0.856 → 0.854 |
| `-y,x,z` | 0.832 | 0.839 | +0.007 | 0.848 → 0.854 |
| `-x,-y,z` | 0.798 | 0.839 | +0.041 | 0.810 → 0.854 |
| `x,-y,-z` | 0.555 | 0.839 | +0.284 | 0.505 → 0.854 |
| `x,-z,y` | 0.736 | 0.839 | +0.102 | 0.741 → 0.854 |
| `z,y,-x` | 0.836 | 0.839 | +0.003 | 0.851 → 0.854 |
| `z,x,y` | 0.768 | 0.839 | +0.071 | 0.779 → 0.854 |
| turned at random 1 (continuous) | 0.764 | 0.840 | +0.076 | 0.774 → 0.856 |
| turned at random 2 (continuous) | 0.660 | 0.839 | +0.179 | 0.656 → 0.854 |
| turned at random 3 (continuous) | 0.547 | 0.841 | **+0.295** | 0.493 → 0.856 |

Mean over the seven axis-aligned wearings: 0.766 plain → **0.839** searched. Mean over the three the
candidates cannot represent at all: 0.657 → **0.840**. The searched column is flat — 0.839±0.001 in every
row — which is the whole point: once gravity is lined up and the roll searched, what the phone did to its
axes matters a good deal less than what the person was doing. The cost is in the first row: on the data exactly as
delivered, the search switched away from the plain reading in **41.5%** of the windows and moved accuracy
by **−0.002**. Looking costs essentially nothing when there is nothing to find, because most of those
switches land on readings the network scores the same way.

That last point is worth being precise about, because it is where the method is least transparent: 41.5%
of switches with a 0.002 loss means the 24 readings of a canonicalized window are largely equivalent to
the network, not that the search is picking wisely. Its mean residual orientation error is 37° (within 20°
in 59% of windows), so on the axis-aligned wearings it does not recover the exact turn either; it recovers
a turn the model likes, and for `D` those two are close. What it demonstrably does is remove the
dependence on how the phone is held: the spread across wearings collapses from 0.547-0.841 to a band of
0.002, and the worst wearing gains 0.295.

**Robustness without any search at all**, the same ten wearings, no second look:

| arm | accuracy as it comes | mean over the 10 wearings | worst | below 0.5 |
| --- | --- | --- | --- | --- |
| `A_plain` | 0.918 | 0.330 | 0.074 | 8 of 10 |
| `B_rot24` | 0.778 | 0.694 | 0.464 | 1 of 10 |
| `C_rotSO3` | 0.679 | 0.640 | 0.583 | 0 of 10 |
| `D_gravity_aligned` | 0.841 | 0.734 | 0.547 | 0 of 10 |

This table is the honest answer to "did the orientation accounting work": the plain model is the best
model in the world for one particular pocket and worse than the class-average baseline in eight of ten
others. Augmenting over the 24 orientations (`B`) buys the turned cases at a price of 0.14 as it comes, and
it pays it **exactly where you would expect** — sitting falls to 0.18 F1 and standing to 0.61, because
turning a window mixes up the 19° tilt that separates them, and continuous augmentation (`C`) finishes the
job: sitting reaches **0.00**, unlearnable. Rotation invariance by training is not free; it is a trade of
the static classes for the turned ones. Canonicalization (`D`) gets the same robustness without spending
the training data on it, and keeps walking/upstairs/downstairs at 0.98-1.00.

Two single-recording cases, `fig_turned_1.png` and `fig_turned_9.png`, both drawn the way §7's timeline
is: the turned signal, the truth, the plain reading, the searched reading, and the candidate kept.
Worn as `-y,x,z`: 0.695 → 0.710 — a 180°-ish turn that barely bothered this arm. Worn at random 3 (a
continuous orientation no candidate can express): **0.407 → 0.711**, with the search settling on `z,-y,x`,
which is *not* the true rotation and did not need to be.

## 9. What made it better than the baseline

Against the notebook's baseline (0.176 / 0.043) any trained network is a foregone conclusion, so the
interesting comparisons are the two real ones: the per-window class-average rule at 0.693 / 0.651, and
the plain model itself at 0.918 / 0.918.

1. **Answer per sample, not per window** — 0.693 → 0.918. The labels are per sample; 12.3% of evaluation
   windows straddle a boundary; the last window is slid back rather than zero-padded, so no answer ever
   comes from padding. Most of the gap to the class-average rule is this, and the transition class shows
   it directly: 0.44 → 0.85.
2. **Refusing to train on nonsense.** The 463 all-pause windows return NaN loss (measured, torch 2.14), and
   one NaN step of Adam resets every weight. Excluding them costs no supervision and is the difference
   between a run that converges and one that does not — the same for the validation loss, without which
   "keep the best weights" keeps epoch 1 for ever.
3. **Choose on validation, score test once.** All 64 search settings were ranked on users 21-25; the test
   users were touched once at the end (§8). The margin, the score, and per-window-vs-per-recording were not
   fitted to the numbers reported there.
4. **The axis search, which is the task's actual addition** — +0.073 mean accuracy over the axis-aligned
   wearings, +0.295 on the worst wearing, for −0.002 when nothing is turned. It does not make the model
   better at activities; it makes the model's accuracy stop depending on how the phone is held.
5. **What did *not* work, and is reported anyway**: rotation invariance by training. It is the obvious
   idea, it costs 0.14 accuracy on correctly-worn data, and it destroys exactly the sitting/standing pair
   (0.79 → 0.18 → 0.00 as the augmentation gets more aggressive), because those two classes differ by a
   19° tilt and augmentation spends tilts. Gravity canonicalization does the same job more cheaply but is
   not exportable and still loses 0.062 macro F1 to the plain model (0.856 against 0.918).

So the deliverable is a pair, not a number: `A_plain` if the phone is clipped upright (0.918), and
orientation accounting — the search in research, `B_rot24` (0.778) in the app — if it is in a pocket. The
0.918 and the 0.839 are the same task measured two different ways, and the gap between them is the price
of a claim that holds whatever the phone is doing.

## 10. Problems met

The ones the pipeline would have shipped silently, each caught by measuring instead of assuming:

* **A window of only pauses costs NaN, not 0.** `CrossEntropyLoss(ignore_index=-1)` returns NaN when
  *every* sample in the window is ignored — measured on torch 2.14 — and one NaN step of Adam resets every
  weight in the network. 463 of 2836 training windows and 103 of 759 validation windows are such windows
  (the long unlabelled stretches), so they are left out of training and validation. They carry no
  supervision either way; what would have carried was a poisoned gradient. The validation loss needed the
  same treatment, or the "keep the best weights" rule would have kept the first epoch for ever.
* **The last window of a recording is slid back, so it owns only its tail.** Writing all 256 of its
  predictions into the 70 samples it is responsible for is a shape error — caught at the first recording.
* **Window positions have to be global, not per-recording.** The search picks an orientation per window and
  then stitches the chosen readings into one timeline; with within-recording offsets, the second
  recording's chosen windows overwrote the first recording's samples and 19,484 test samples went
  unanswered. The report now re-scores every saved prediction file from disk and asserts it reproduces the
  numbers in `results_net.json`.
* **`torch.Tensor.mode(1)` squeezes the dimension it reduces** — the "share of the window that agrees with
  its majority class" compared a (batch, 256) tensor against a (batch,) one. Needs `keepdim=True`.
* **`torch.onnx.export` in torch 2.14 goes through the dynamo exporter**, which needs `onnxscript`
  installed, and prints a check mark that stops the script on a cp1250 Windows console.
* Three of the seven stress orientations first written down were **mirror images** (determinant −1).
* `torch.onnx.export` fixes the batch axis at 1, which is what the app sends but not what a batched
  verification expects.
* **The best arm is not the exportable arm.** The search was chosen on `D_gravity_aligned` (0.897
  validation macro F1) and the app can only run arms that read raw axes, so the export step had to gain a
  rule — "best of the exportable arms on validation" — and the report had to stop telling the reader to
  export something the exporter refuses. Caught by running the exporter, not by reading the plan.
* **A "canonicalize gravity" fix has a free parameter it does not fix.** The assertion first written for
  it — that canonicalizing undoes a turned phone — is false: roll about gravity is untouched. Replaced by
  three checks that are true (along-gravity row identical, perpendicular magnitudes identical, roll left
  free) plus the measured residual, which is the number §8 quotes.
* **The augmentation arms converge slower and train slower for the same wall-clock budget**: best epoch 47
  (plain) against 185 (`B_rot24`), and 804 s against 1731 s for 400 epochs. If the arms had been compared
  at a fixed time instead of a fixed epoch count, part of the "augmentation hurts" result would have been
  an artifact of the budget.
* **The margin the search chose is 0.00**, i.e. always take the best of 24 — a rule whose name is
  "switch only if it matches better" and whose behaviour is "switch unless nothing else matches as well".
  Larger margins scored worse on validation (0.870 at margin 0.02, 0.864 per-window). Worth knowing before
  trusting the wording of the rule over its measured behaviour.

## 11. Reproducing it

```
py -I task2_motion/parse_motion.py   <hapt-dir> task2_motion/cache/hapt.npz task2_motion/cache/records.csv
py -I task2_motion/inspect_motion.py task2_motion/cache/hapt.npz task2_motion/figs task2_motion/cache/records.csv
py -I task2_motion/train_motion.py   task2_motion/cache/hapt.npz task2_motion/figs          # ~1 h: 4 arms + the search chosen on validation
py -I task2_motion/export_onnx.py    task2_motion/figs task2_motion/activity_model.onnx task2_motion/cache/hapt.npz
py -I task2_motion/report_motion.py  task2_motion/cache/hapt.npz task2_motion/figs
```

`part1.py` is a byte-identical copy of the notebook's Part 1 (md5 checked), so the scripts run outside the
notebook. `logs/` holds the full console output of each stage; `figs/` holds the figures named below and
the per-arm checkpoints, which carry the window length, units and axis order next to the weights.

Figures: `inspect_gravity.png`, `inspect_classes.png`, `inspect_test_recording.png` (§3-4, before any
training); `curve_<arm>.png` (the four training runs); `fig_results.png` (§7); `fig_confusion.png` (§7);
`fig_timeline.png` (§7); `fig_rotations.png`, `fig_turned_1.png`, `fig_turned_9.png` (§8).
