# REVIEW.md

My initial thoughts about the individual tasks were written down before
I started working on them. They were based purely on manually looking
through the input data and trying to get a general idea of what I was
dealing with. I had not run statistical analysis, checked the scripts'
outputs or done any proper measurements at that point. So some of these
first impressions were right, some were not, and in a few cases the
approach I initially had in mind was later replaced by something that
worked better.

The point of this review is mostly to keep track of that process: what I
thought at the beginning, what I tried or considered, and what I ended
up learning from the actual implementation and results.

## Task 1 -- ECG delineation

At first glance the dataset seemed alright, but one thing that bothered
me was that some QRS complexes did not seem to be annotated. I was
worried that this could confuse the model, since it would see a clear
QRS complex in the signal but the ground truth would tell it that those
samples are just background.

My first idea was to somehow cut the raw signal so that it started and
ended at the same places as the annotated part. I was aware that using
the ground truth to decide where to cut the input could be considered
data leakage. I thought it might still be defensible as a way of dealing
with incorrectly or incompletely annotated records, rather than giving
the model extra information about the actual waves. I also thought about
leaving a small buffer around the cut, so that the model would not
simply learn that the first QRS is always at the beginning and the last
one is always at the end.

This was an early idea rather than the final solution. After looking
into the problem more carefully, cropping turned out to have several
disadvantages: it would remove a fairly large part of the signal,
produce records of different lengths and make the annotation boundaries
visible through the input itself. It could also bring unlabelled beats
back into the evaluated region depending on the buffer size. The final
solution was therefore different. Instead of cutting the signal, the
problematic samples were marked with `-1` and ignored by both the loss
function and the metric. This kept the full signal available to the
network while avoiding training it to classify known annotation errors
as background.

The report also looked at the unlabelled QRS complexes inside the
annotated region, not just the ones at the edges. Those were handled
with the same masking idea. The comparison between training with the
original labels and the repaired labels suggested that this was not just
changing how the score was calculated: the model trained with repaired
labels marked all 288 annotated test beats, while the model trained on
the original labels missed some of them.

The final model used a U-Net on lead II. A few things were tested after
the basic version was working, including using more ECG leads, changing
the loss, trying preprocessing, and combining several independently
trained models. More leads did not improve the result, and the tested
preprocessing steps made it slightly worse. The final ensemble of three
models gave the best reported result, with a mean test Dice of 0.876
compared with 0.758 for the reconstructed baseline. The individual
scores were 0.782 for P waves, 0.942 for QRS complexes and 0.904 for T
waves.

So my initial concern about the missing annotations was valid, but my
first proposed fix was not the one that ended up being used. Masking the
unreliable labels while preserving the signal was a cleaner solution.

## Task 2 -- Smartphone accelerometer / activity recognition

My first impression of this dataset was that the accelerometer
recordings seemed quite variable. I suspected that the phone might not
have been held or attached in exactly the same orientation every time.
Since the data has three acceleration axes, changing the phone's
orientation could change the values on each axis even if the person was
doing the same activity.

I did not know how much this would affect the model, so my initial plan
was to first train a simple model without doing anything special to the
data. That would give me a baseline. After that I wanted to try
rotations and swaps of the axes as an extended version, to see whether
the model could become less sensitive to how the phone was positioned.

The interesting part was that the assumption about the original
recordings was not really supported by the later inspection. The HAPT
dataset uses a waist-mounted phone, and its orientation was fairly
consistent across the recordings. In other words, the variation I
thought I was seeing did not automatically mean that the phone had been
placed in many different orientations. To test orientation robustness
properly, rotated versions of the recordings had to be generated
artificially.

The simple baseline performed very well on the original test data,
reaching 0.918 accuracy and 0.918 macro F1. The orientation-robust
approaches were more complicated and generally gave up some performance
on the unchanged recordings in exchange for handling rotated inputs
better. The delivered orientation-aware model with the
search/canonicalization step reached 0.839 accuracy and 0.854 macro F1
on the normal test set, while being much more stable across the
simulated orientations. A rotation-augmented model was also exported as
a more practical robust alternative.

One important limitation is that the artificial rotations are stress
tests, not evidence that the original HAPT recordings contain those
phone orientations. The gravity-alignment and axis-search approach was
useful for investigating robustness, but it was not directly suitable
for exporting as the final app model. The rotation-augmented model was
the more practical exported option.

In the end, the basic plan of comparing a simple model with an
orientation-aware version made sense. However, the initial explanation
for the apparent variability was only a guess based on looking at the
data. The later inspection showed that the dataset itself was much more
consistent than I had assumed. The rotation experiments were still
useful, but they answered a slightly different question: how well would
the model cope if the phone orientation changed?

## Task 4 -- Cell counting

This one initially looked like the simplest task to me. From manually
looking at the images, my first idea was to count enclosed regions that
had a different colour from the background, treating each region as one
cell regardless of its size or shape. I also thought that a more
advanced version could look at the shapes of the regions to identify
places where multiple cells were touching or overlapping.

That was only an initial visual idea. Once the images were examined more
closely, it became clear that overlap was not just an occasional edge
case. A large proportion of the cells were part of merged regions, and
simply counting connected blobs would miss many cells. In fact, the Otsu
blob-counting baseline found only about half as many blobs as there were
annotated cells and performed worse than just predicting the
training-set mean.

A hand-crafted area-splitting rule was tested as a way of estimating how
many cells were contained in larger blobs. It performed much better than
simple blob counting, with a test MAE of 12.7. A brightness-based
estimate did even better, with a test MAE of 10.7, but this was a
warning sign rather than a satisfying cell-counting solution: it could
estimate the total count without actually locating individual cells. The
report therefore also introduced a localization metric, so that a method
could not look good based only on producing the right total number.

The main approach specified for the task was different from my original
blob idea. Instead of trying to separate every cell, the model was
trained to produce a density map with a small Gaussian centred at each
annotated cell location. The predicted count is then the sum of the
output map. This avoids requiring the network to draw a separate
boundary around every cell, which is particularly useful when cells
overlap and merge into one large bright region.

The report describes the U-Net setup and the experiments planned for it,
including different Gaussian widths, loss functions, input normalization
and model depth. However, the version of the task report I reviewed
still has the final U-Net result table and parts of the results section
marked as pending. I therefore cannot make a fair statement here about
the final network performance or claim that it beat the hand-crafted
estimates. What is already clear from the completed analysis is why the
simple blob-counting idea was inadequate, why overlap-aware rules
helped, and why a density-map approach is a more suitable way to
formulate the actual task.

My initial thought was basically to count visible regions and then
improve the counting of merged regions by looking at their shapes. The
later approach changed the problem: rather than trying to split every
connected region into individual cells, it estimates a spatial density
and obtains the count from the sum.

## Overall thoughts

Across the three tasks, the first step was mostly just looking at the
data and making guesses about what might be difficult. That was useful
for coming up with possible approaches, but it was not enough to
establish what the actual problem was.

For ECG delineation, the missing annotations really were an issue, but
masking the unreliable labels worked better than cutting the signal. For
the accelerometer task, the orientation concern led to useful robustness
experiments, although the original recordings were more consistently
oriented than I first thought. For cell counting, the idea of counting
connected regions was too simplistic because so many cells overlap; the
density-map formulation avoids having to separate them explicitly.

The main thing I take away is that the first approach does not have to
be the final one. In each task, the initial observations helped suggest
what to investigate, but the later measurements and experiments were
what showed which assumptions held up and which solutions were actually
useful.
