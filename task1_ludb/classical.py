"""The classical delineator of Task 1: find the R peaks, then place the waves at fixed distances.

baseline_ludb.py searches this detector's settings on the validation split; train_ludb.py needs the
same detector to score the network against it. Both import it from here, because "the network beats
the baseline" only means something if the two were produced by the same code and judged on the same
samples.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks

FS = 500

# the settings baseline_ludb.py's grid search picked on the VALIDATION split (never on test)
CHOSEN = dict(kind=("bandpass", (8, 20)), frac=0.40, distance=150, sigma=0.0, rr_scaled=False)


def detect(signals, kind, frac, distance, sigma):
    """The R peaks of every record: local maxima above `frac` of the maximum of its own curve.

    kind is "thresh" (the signal itself), "smooth", "energy" (squared derivative) or
    ("bandpass", (low, high)) with a Butterworth band-pass in Hz. The threshold is always relative
    to the record's own maximum - LUDB stores every record scaled to exactly 1 mV peak-to-peak, so
    absolute amplitudes mean nothing here.
    """
    band = kind[1] if isinstance(kind, tuple) else None
    kind = kind[0] if isinstance(kind, tuple) else kind
    if band is not None:
        b, a = butter(3, [lo / (FS / 2) for lo in band], btype="band")
        signals = filtfilt(b, a, signals, axis=-1)
    curve = gaussian_filter1d(signals, sigma, axis=-1) if sigma > 0 else signals
    if kind == "energy":
        curve = gaussian_filter1d(np.diff(curve, axis=-1) ** 2, 3, axis=-1)
    elif kind == "bandpass":
        curve = gaussian_filter1d(np.abs(curve), max(sigma, 1.0), axis=-1)
    return [find_peaks(c, height=frac * c.max(), distance=distance)[0] for c in curve]


def median_offsets(waves, rec_of, records):
    """Median (onset, offset) of every class relative to the R peak of its beat, measured on `records`.

    `waves` is the (record, class, onset, peak, offset) table; measured on the training records only.
    """
    rel = {c: [[], []] for c in (1, 2, 3)}
    for r in records:
        r_peaks = np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3])
        for c in (1, 2, 3):
            for _, _, onset, peak, offset in waves[(rec_of == r) & (waves[:, 1] == c)]:
                if c == 2:
                    anchor = peak                                # the QRS against its own peak
                else:
                    if len(r_peaks) == 0:
                        continue
                    anchor = r_peaks[np.argmin(np.abs(r_peaks - peak))]
                    if abs(peak - anchor) > (600 if c == 1 else 900):
                        continue                                 # the wave belongs to another beat
                rel[c][0].append(onset - anchor)
                rel[c][1].append(offset - anchor)
    return {c: (int(np.median(rel[c][0])), int(np.median(rel[c][1]))) for c in (1, 2, 3)}


def paint(peaks_of_record, offsets, rr_ref=None):
    """A class-per-sample array from the detected R peaks and the median wave intervals.

    With `rr_ref`, the P and T windows stretch with that beat's own R-R interval (a slow beat has a
    longer PR and QT); the QRS keeps its fixed width.
    """
    T = 5000
    out = np.zeros(T, dtype=np.int64)
    p_arr = np.sort(np.asarray(peaks_of_record))
    for i, p in enumerate(p_arr):
        stretch = 1.0
        if rr_ref:
            before = p_arr[i] - p_arr[i - 1] if i > 0 else rr_ref
            after = p_arr[i + 1] - p_arr[i] if i + 1 < len(p_arr) else rr_ref
            stretch = float(np.clip((before + after) / 2 / rr_ref, 0.6, 1.7))
        for c in (1, 3, 2):                                      # QRS last: where waves collide, it wins
            a, b = offsets[c]
            if c != 2:
                a, b = int(round(a * stretch)), int(round(b * stretch))
            out[max(p + a, 0):min(p + b, T - 1) + 1] = c
    return out


def labels(x, waves, rec_of, train_idx, cfg=CHOSEN):
    """The baseline's class-per-sample answer for every record in `x`."""
    offsets = median_offsets(waves, rec_of, train_idx)
    rr_ref = None
    if cfg["rr_scaled"]:
        rr = [np.diff(np.sort(waves[(rec_of == r) & (waves[:, 1] == 2), 3])) for r in train_idx]
        rr_ref = float(np.median(np.concatenate([g for g in rr if len(g)])))
    peaks = detect(x, cfg["kind"], cfg["frac"], cfg["distance"], cfg["sigma"])
    return np.array([paint(p, offsets, rr_ref) for p in peaks]), offsets, rr_ref
