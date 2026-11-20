"""Per-wave Dice for ECG delineation, with an optional ignore mask (Task 1).

The same functions score the classical baseline and the network, so the two are always
compared on identical samples. Where `mask` is given, only samples with mask >= 0 count:
they are neither trained on nor judged, which is how the unannotated beats at the record
edges (see repair_ludb.py) are kept from being scored as errors.
"""
import numpy as np

WAVES = ["background", "P", "QRS", "T"]
CLASSES = (1, 2, 3)


def per_record(y_true, y_pred, cls, mask=None):
    """Dice of one wave class per record; None for a record with nothing to say about that wave."""
    keep_all = np.zeros_like(y_true) if mask is None else mask
    vals = []
    for t_row, p_row, k_row in zip(y_true, y_pred, keep_all):
        keep = k_row >= 0
        t, q = (t_row == cls) & keep, (p_row == cls) & keep
        vals.append(None if (not t.any() and not q.any()) else float(2 * np.logical_and(t, q).sum()
                                                                      / max(t.sum() + q.sum(), 1)))
    return vals


def evaluate(y_true, y_pred, mask=None):
    """macro = mean of the per-record Dice; micro = pooled over all counted samples."""
    keep_all = np.zeros_like(y_true) if mask is None else mask
    counted = keep_all >= 0
    out = {}
    for c in CLASSES:
        vals = [v for v in per_record(y_true, y_pred, c, mask) if v is not None]
        t, q = (y_true == c) & counted, (y_pred == c) & counted
        out[WAVES[c]] = dict(macro=float(np.mean(vals)), n_records=len(vals),
                             micro=float(2 * np.logical_and(t, q).sum() / max(t.sum() + q.sum(), 1)))
    return out


def fmt(res):
    return "  / ".join(f"{WAVES[c]} {res[WAVES[c]]['macro']:.3f}" for c in CLASSES)


def runs_of(row, cls=2):
    """(start, end) inclusive of every contiguous run of `cls` in one row of predicted labels."""
    inside = (np.asarray(row) == cls).astype(np.int8)
    step = np.diff(np.concatenate(([0], inside, [0])))               # +1 entering a run, -1 leaving it
    return [(int(s), int(e) - 1) for s, e in zip(np.flatnonzero(step == 1), np.flatnonzero(step == -1))]


def qrs_durations(annotated, pred_row, ms):
    """(annotated, predicted) length in ms of every annotated QRS the answer overlaps.

    `annotated` is [(onset, offset), ...] of the cardiologists' QRS runs, `ms` is 1000/FS. Where the
    answer splits one beat into two runs, the widest overlap is the one measured - the question the
    notebook asks is how long the network thinks the complex is, not how many pieces it drew.
    """
    runs = runs_of(pred_row, 2)
    pairs = []
    for a, b in annotated:
        over = [(p, q) for p, q in runs if p <= b and q >= a]
        if over:
            p, q = max(over, key=lambda pq: min(b, pq[1]) - max(a, pq[0]))
            pairs.append(((b - a + 1) * ms, (q - p + 1) * ms))
    return pairs
