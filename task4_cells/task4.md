# Task 4 — Counting cells (2D, a density map → a count)

The question the task sets is *"how many cells are in this image?"*, and the way it sets it is: put a
small Gaussian on the centre of every annotated cell, let a U-Net paint that **density map**, and read
the count as the **sum of the network's output** — nobody has to find the individual cells.

Everything below was run from `task4_cells/`; `logs/` holds the raw output of every script and
`figs/` every figure, so each number quoted here can be traced to a file.

## 1. Final values

<!-- filled in once the network has been trained; the baseline half is final already -->

| | count MAE (test) | RMSE | bias | R² | worst | localization (recall / precision / F1) |
| --- | --- | --- | --- | --- | --- | --- |
| constant, train mean | **56.1** | 64.5 | +10.0 | −0.025 | 136 | — (it never looks at an image) |
| Otsu threshold, count the blobs | **94.6** | 120.2 | −83.1 | −2.555 | 265 | — |
| blobs at 0.60, area ÷ 60 px (hand-crafted overlap rule) | 12.7 | 15.1 | −6.9 | 0.944 | 39 | — |
| sum of the brightness above 0.20 ÷ 62 (the leak) | 10.7 | 12.8 | −3.3 | 0.960 | 31 | — |
| peaks of a σ = 1.5 blur | 25.4 | 34.3 | −19.9 | 0.711 | 101 | 0.707 / 0.800 / 0.751 |
| **U-Net, density map, sum of the output** | *pending* | | | | | |
| ceiling: the target map itself, σ = 1.0 | 0.0 by construction | — | — | — | — | 0.950 / 1.000 / 0.974 |

The task's own targets are a MAE of about **56** for the constant answer, about **95** for Otsu blobs,
and "a count MAE well under 10 in most runs" for the network. Both of its baselines are reproduced to
the cell (§5). What the task does not say, and what turned out to matter more than the network, is
that **on this data a count MAE of 10.7 is reachable with one fitted number and no cell-finding at
all** (§5.3) — which is why every network number here is reported next to a localization score (§6).

## 2. The task, as set

`ex09_student.ipynb`, Part 2, Task 4. Data: the synthetic fluorescence images of the VGG counting
paper (Lempitsky & Zisserman, NIPS 2010) — 200 images of 256 × 256 with a 1-pixel dot at every cell
centre, split 80 train / 20 validation / 100 test (the last being the paper's own test set). The
notebook's data cell keeps the **blue** channel as the image and defines a cell as a pixel of the dots
image that is *red and not green*; `parse_cells.py` reproduces both choices line for line, because the
count we are scored on is the pixel sum of that mask.

What the notebook asks to be shown: the specification table (the student's own), the check that the
target sums to the count, the test count MAE of the network and of both baselines, a scatter of
predicted against true count for all test images with the diagonal, and one test image with its target
and its predicted map, each with its sum. All of them are in §8.

## 3. Is the data what it claims to be?

`inspect_cells.py`, output in `logs/inspect.log`. Short answer: yes, and it is internally consistent —
but two of its properties decide the whole task before any model is chosen.

| checked | found |
| --- | --- |
| arrays | `x` (200, 256, 256) float32 in [0.0000, 1.0000], no non-finite values |
| counts | 35 192 dots; per image 74–317, mean 176.0, median 174, sd 61.4 |
| splits | 80 / 20 / 100, disjoint, covering all 200. Train mean 181.5, **test mean 171.5** — the paper's test set is denser by 10 cells, which is why a constant answer carries a +10 bias |
| the channels | blue mean 0.355; red 0.0018 and green 0.0020 — "the other channels are empty" is nearly true, they hold the dot encoding, and the notebook's `green == 0` condition drops **0** of the 35 192 red dots |
| duplicated images | 0 pairs share a coarse fingerprint, 0 byte-identical pairs |
| the dots | 35 192 marked pixels in **34 741** connected components (sizes 1–3): 451 places where two annotated centres touch and their 1-pixel dots merge |
| centre spacing | p1 1.0, p5 2.2, **median 8.5**, p95 20.6 px. 896 pairs are closer than 2 px |
| border cells | 441 centres on the outermost row or column |
| a cell's size | blobs holding exactly one dot: median area 126 px (p5 63, p95 211) → radius ≈ 6.3 px, **a cell is ≈ 13 px across**; median eccentricity 0.41, so "cells are round" is supported |
| label coverage | 1 of 35 192 centres falls on no foreground pixel — the annotations and the image agree |

**Property 1 — the dots are not one per cell in a physical sense.** 451 pairs of centres touch and 896
are within 2 px, and the cell is 13 px across. No smooth map, and no peak finder, can separate two
centres 1 px apart. That is a ceiling on any *localization* score (the target map itself only reaches
recall 0.950, §5.4) and it is not the network's fault. It does not affect the count, which is a sum.

**Property 2 — the field is saturated.** The bright region covers 0.685 ± 0.007 of the frame in every
image, whether it holds 74 cells or 317; the correlation between foreground fraction and count is
−0.031. What changes with the count is not how much of the frame is bright but how much bright area
each cell gets:

| cells in image | foreground fraction | foreground px per cell | in lone-cell areas |
| --- | --- | --- | --- |
| 70–120 | 0.686 | 465 | 3.7 |
| 120–170 | 0.685 | 314 | 2.5 |
| 170–220 | 0.684 | 228 | 1.8 |
| 220–270 | 0.685 | 186 | 1.5 |
| 270–320 | 0.686 | 152 | 1.2 |

Otsu's threshold (median 0.252 over the images) sits **below** the median pixel value (0.290), so it is
not separating cells from background at all — it is cutting the background in half: 26.7 % of what it
calls foreground lies more than 12 px from any annotated centre. Consequence, measured: dividing the
foreground area by the area of a cell gives a test MAE of 56.0, i.e. exactly the constant answer. Area
carries no count information here.

## 4. Overlap is not the edge case, it is the image

The task's premise, measured:

- Otsu finds **17 356 blobs for 35 192 cells** — a blob counter sees **49.3 %** of the cells (the
  notebook's own estimate is "about 45 %").
- cells per blob: 1 → 2 019 blobs, 2 → 657, 3–4 → 460, 5–9 → 251, **10+ → 264**. A non-empty blob holds
  9.64 cells on average, the largest holds 270. **94.3 %** of all cells sit in a blob with at least one
  other cell, and 96.5 % of the foreground area is inside a multi-cell blob.
- 75.3 % of cells have their nearest neighbour within one cell diameter.
- How much room a cell gets depends on how crowded it is, and a blob's shape tells you the count:

| cells in the blob | blobs | median area | area / lone cell | area per cell / lone cell | median intensity | eccentricity |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2019 | 126 | 1.00 | 1.00 | 0.478 | 0.41 |
| 2 | 657 | 221 | 1.75 | 0.88 | 0.505 | 0.81 |
| 3 | 300 | 336 | 2.67 | 0.89 | 0.514 | 0.84 |
| 4 | 160 | 438 | 3.48 | 0.87 | 0.534 | 0.84 |
| 5–6 | 154 | 564 | 4.48 | 0.90 | 0.539 | 0.83 |
| 7–12 | 122 | 888 | 7.04 | 0.88 | 0.552 | 0.84 |
| 13–30 | 36 | 1536 | 12.19 | 0.81 | 0.577 | 0.82 |
| 31+ | 203 | 40 619 | 322.4 | 2.34 | 0.431 | 0.62 |

Two regimes, and they are the whole difficulty. Up to about six cells a blob gains **104 px per extra
cell** — less than the 126 px of a lone cell, because the cells share a boundary: that is precisely the
"two overlapping cells are smaller than two circles" the task describes, and within this regime area
*is* nearly proportional to count (the `area per cell` column is flat at 0.87–0.90). Beyond that the
dense images are one confluent sheet with bright haze between the cells, and the same division returns
2.34 cell-areas per cell. So area works on the sparse blobs and fails on the crowded ones — and 96.5 %
of the area is in the crowded ones. Brightness per pixel says almost nothing either (a blob's median
intensity goes 0.478 → 0.577 from one cell to thirty); only the **integral over the whole image** does,
which is §5.3.

## 5. The baselines

`classical_cells.py`, output in `logs/baseline.log`, everything with a knob tuned on the 20 validation
images and the test split evaluated once at the end.

### 5.1 The two the task names

| baseline | how | test MAE |
| --- | --- | --- |
| answer the training mean (181.5) for every image | no parameters | **56.1** (notebook: about 56) |
| Otsu threshold per image, count the connected blobs | no parameters | **94.6** (notebook: about 95) |

The blob counter is worse than answering a constant, and the reason is visible in its correlation with
the truth: **−0.692**. Denser images merge into *fewer* blobs, so the answer moves the wrong way with
the count; its bias is −83 cells and it is exactly right on 0 % of the images. That negative sign is
the task's premise stated as a number.

### 5.2 Hand-crafted rules that do take overlap into account

The hypothesis the task is built on — an irregular, oversized blob may be two cells — was implemented
directly, and it works:

| rule | knobs (validation-chosen) | test MAE | R² |
| --- | --- | --- | --- |
| bright cores: threshold at 0.70, count blobs | t = 0.70 | 81.2 | −1.18 |
| **area splitting: Σ over blobs of round(area ÷ 60 px)** | t = 0.60, divisor 60 px | **12.7** | 0.944 |
| peaks of a Gaussian-blurred image, σ = 1.5, level 0.30 × max | σ, level | 25.4 | 0.711 |

Counting the blobs at the same 0.60 threshold without splitting them gives 85.4, so **the split alone
is worth 72.8 cells**. The divisor the fit wants (60 px) is half a lone cell (126 px), because at that
threshold nearly every cell already sits inside a fattened sheet — the saturation of §3 absorbed into
one fitted number. The peak counter is the only baseline that can be scored for localization: recall
0.707, precision 0.800 on the test split.

### 5.3 The leak: total brightness

Fluorescence adds up, so the integral of an image is nearly proportional to its cell count —
correlation **0.991** with the count. One number fitted on the training set,

    count ≈ Σ pixels max(image − 0.20, 0) ÷ 62      →  test MAE 10.7, R² 0.960,

beats both of the task's baselines by 5×, beats the hand-crafted overlap rule, and never finds a cell.
This is the number a network has to beat, and it is the reason a second metric was built before the
network was (§6).

### 5.4 What a perfect map could score

The target is a Gaussian on every centre, so its own sum *is* the count: with scipy's default
reflecting border the ratio is 1.00000 for all 200 images at every σ tried. Cutting the kernel off at
the border instead — which is what a convolution layer with zero padding does — loses up to 1.6 % of
the mass at σ = 1.0 and 3.8 % at σ = 3.0 on the worst image (the one with the most of the 441 border
centres). Since the target is computed on the CPU with `gaussian_filter` and not by a convolution
layer, the maps actually used keep the exact sum.

Scored as a counter of its own cells, the target map reaches recall **0.950**, precision **1.000**,
F1 **0.974** on the test split at σ = 1.0 (chosen on validation). That is the ceiling of peak
counting, set by the 451 touching centre pairs and by the width of the bumps — a network scoring
recall 0.95 has found every cell it is possible to find.

## 6. The metric: the count, and whether the count was earned

`cells_metric.py`. The task scores one number, the count MAE; §5.3 shows what one number is worth
here, so the scorer reports the count (MAE, RMSE, bias, R², worst image, relative MAE) **and** a
localization score: turn the map into peaks, match each peak greedily — strongest first — to the
nearest unused annotated centre within `tol` = 5 px, and report micro precision / recall / F1 against
the 35 192 centres. `frac` = 0.25 of each map's own maximum is kept, so the score does not depend on
the units the network happens to paint in, and `min_distance` = 3 px is deliberately permissive (a cell
is 13 px across, so 6 px is the honest physical limit — §8 reports both).

The module checks itself: a map that is the target scores recall 1.000; the **same map shifted 9 px**
scores count MAE 0.00 and recall 0.000 — the count lies, the localization does not; two centres 1 px
apart cap the recall of any peak finder with a 3 px minimum distance at 0.50.

## 7. The network

`UNet(2, 1, 1, [16, 32, 64, 128])` from Part 1 of the notebook — the same code as Task 1's, with 2D
convolutions — followed by a last layer, trained by `train()` and read out by `predict()`, none of them
changed. Input `(1, 256, 256)` blue channel; output `(1, 256, 256)` density map; **the count is the sum
of the output**, so the network never has to decide where one cell ends and the next begins — that is
what makes it the answer to §4: it does not have to separate the sheet.

Every choice was made on the 20 validation images, by the count MAE (the task's metric), with the
localization score printed next to it so that a "win" bought by brightness is visible. The test split
was evaluated once, at the end. `train()` keeps the weights of the best validation epoch.

| stage | what was varied | arms |
| --- | --- | --- |
| A | σ of the bumps: 1.0, 1.5, 2.0, 3.0 | 4 |
| B | loss: MSE, L1, MSE + 100·(sum error)², MSE + 1000·(sum error)²; last layer: linear, softplus | 5 |
| C | what the network may see: raw, raw + brightness jitter, per-image standardized | 3 |
| D | learning rate 3·10⁻⁴, 3·10⁻³; and the same depth at twice the channels | 3 |

Before the sweep, one arm was measured at 5 levels of depth and dropped (§10). <!-- fill: sweep table, chosen arm, why -->

## 8. Results, and the deliverables the notebook asks for

<!-- filled in: test table vs baselines, scatter with the diagonal, one test image with its two maps, worst images, sensitivity of the localization score -->

## 9. What made it better than the baseline

<!-- filled in -->

## 10. Problems met

**The blob baseline is worse than answering a constant, and the sign says why.** Its correlation with
the true count is **−0.692**: as an image gets denser its cells merge into *fewer* blobs, so the answer
moves away from the truth instead of toward it. A baseline that is worse than a constant is a finding,
not a bug, and it is the task's premise in one number.

**Area carries no count information on this data.** The foreground fraction is 0.685 ± 0.007 for every
image, so `foreground area ÷ area of a cell` — the most natural overlap rule there is — scores 56.0 on
test, which is the constant answer to one decimal. It is only *within* the sparse regime (blobs of up
to ~6 cells) that area is proportional to count, and 96.5 % of the foreground is outside it.

**Total brightness is nearly the answer, which makes the task's metric on its own meaningless.**
Correlation 0.991, and `Σ max(x − 0.20, 0) ÷ 62` scores 10.7 (§5.3) with one fitted parameter. The
response was not to hide that number but to widen the metric before training (§6) and to measure, after
training, how much of the network's answer comes in through that door (§9).

**The target's mass leaks at the border.** A Gaussian on a centre 3 px from the edge puts part of
it outside the frame. With scipy's default reflecting border the map still sums to the count exactly
(ratio 1.00000 for all 200 images); with the kernel cut off at the border — which is what a
convolution layer with zero padding effectively does — the worst of the 441 border-centred images loses
1.6 % of its mass at σ = 1.0 and 3.8 % at σ = 3.0. The targets are therefore built on the CPU with
`gaussian_filter`, so the map the network is asked to reproduce already has the exact sum.

**No method can score a perfect localization score here.** 451 pairs of annotated centres touch (their
1-pixel dots merge into one connected component) and 896 pairs are closer than 2 px, while a cell is
13 px across. The target map itself reaches recall 0.950, not 1.000. So recall is read against that
ceiling, and the sensitivity table in §8 shows what the two knobs of the matcher are worth.

**The count is a sum of 65 536 pixels, so it is far more sensitive than the loss looks.** The mean
target pixel value is 176 / 65 536 ≈ 0.0027; a per-pixel bias of 0.001 is 65 cells. At initialisation
the linear head's output noise integrates to answers of order ±1 300 cells, and after 6 epochs a
training run was still answering **1 097.6 for a 160-cell image** with 47 % of the map negative (a
linear 1×1 head is free to go negative, and the MSE does not mind). That is why the sweep contains a
softplus head, and why it contains a loss term that scores the sum directly.

**Depth cost more than it paid.** The first arm swept was 5 levels deep: **3.5× the wall clock**
(1 202 s against 340 s for the same 300 epochs) and **24.41 against 15.09** on validation
(`logs/train_partial_5lv.log`). Each extra pooling throws away detail at exactly the scale the cells
live at — a cell is 13 px, so the 5th level sees 32-px blobs. The sweep runs 4 levels and tries width
instead of depth.

**The test set is denser than the training set** — mean 171.5 against 181.5. Everything that fits a
constant (the baseline, the brightness scale, a bias term) carries a ≈ +10-cell bias for that reason
alone, and it is visible in the bias column of §1.

**20 validation images is a short ruler.** Differences of ~1 cell of count MAE between arms are well
inside what one re-run moves; the three-seed spread on the test split (§8) is the honest size of that
noise, and arm differences smaller than it are reported as ties, not as wins.

**Re-running the same arm under the same seed is not automatically the same number**, so cuDNN
determinism was switched on and the restart of the sweep gave a direct check: the same arm
(σ = 1.0, 4 levels, MSE, seed 0) measured twice, <!-- fill: 15.09 vs X.XX -->.

<!-- two more items will be added from stages B and D: the L1 collapse, and the epoch budget -->


## 11. Reproducing it

```
py -I task4_cells/parse_cells.py     <dir with the 400 pngs> task4_cells/cache/cells.npz task4_cells/cache/cells_records.csv
py -I task4_cells/inspect_cells.py   task4_cells/cache/cells.npz task4_cells/figs      > logs/inspect.log
py -I task4_cells/classical_cells.py task4_cells/cache/cells.npz task4_cells/figs      > logs/baseline.log
py -I task4_cells/train_cells.py     task4_cells/cache/cells.npz task4_cells/figs      > logs/train.log
py -I task4_cells/probe_cells.py     task4_cells/cache/cells.npz task4_cells/figs      > logs/probe.log
py -I task4_cells/report_cells.py    task4_cells/cache/cells.npz task4_cells/figs      > logs/report.log
```

`part1.py` is Part 1 of the notebook (its `UNet`, `train`, `predict`) copied verbatim — the same
checksum as the copy in `task1_ludb/` — and used unchanged: `UNet(2, 1, 1, widths)` is that code with
2D convolutions. `cells_data.py` holds the density target, the dataset and the losses so that the
training script and the leak probe cannot drift apart; `cells_metric.py` holds the scorer and runs its
own self-check when executed directly.

The specification table for the notebook's cell 47 and `REVIEW.md` are the student's own to write;
every number they need is in §1, §5 and §8.
