"""Parse the LUDB database into the arrays the notebook's Task 1 data cell produces.

Parses lead II of all 200 records plus the cardiologists' wave annotations, and caches
the result in an .npz so that later scripts (baseline, model) do not have to re-read it.

Usage:  py -I task1_ludb/parse_ludb.py <path-to-ludb-folder> <cache.npz>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

FS = 500                                            # Hz: one sample = 2 ms
WAVES = ["background", "P", "QRS", "T"]
wave_code = {"p": 1, "N": 2, "t": 3}                # wfdb symbols of P, QRS, T


LEADS_MODEL = ["II", "V1", "V5"]         # the three-lead input tried against lead II alone


def load(ludb_dir):
    """Lead II of every patient + a class per sample + the wave list. As in the notebook.

    The notebook's data cell hands the network lead II only, and `x` is exactly that. All 12 leads
    are cached alongside it as `x12` (standardized lead by lead with the same formula, so lead II's
    column is identical to `x`), because the annotations are made on lead II while a P wave is often
    largest in another lead - worth measuring rather than assuming.
    """
    ludb_x, ludb_y, ludb_waves, ludb_x12, extra = [], [], [], [], []
    for rec in range(1, 201):
        path = str(Path(ludb_dir) / "data" / str(rec))
        record = wfdb.rdrecord(path)                                    # all 12 leads
        names = [n.upper() for n in record.sig_name]
        i_ii = names.index("II")
        x = record.p_signal[:, i_ii]                                    # lead II, in mV
        ludb_x12.append(np.stack([(c - np.median(c)) / (c.std() + 1e-6) for c in record.p_signal.T]).astype(np.float32))
        ann = wfdb.rdann(path, "ii")                               # the annotation of lead II
        y, waves = np.zeros(len(x), dtype=np.int64), []
        matched = set()
        for k in range(1, len(ann.symbol) - 1):   # a wave is a triplet: "(" onset, symbol peak, ")" offset
            if ann.symbol[k] in wave_code and ann.symbol[k - 1] == "(" and ann.symbol[k + 1] == ")":
                c, onset, peak, offset = wave_code[ann.symbol[k]], ann.sample[k - 1], ann.sample[k], ann.sample[k + 1]
                y[onset:offset + 1] = c
                waves.append((c, onset, peak, offset))
                matched |= {k - 1, k, k + 1}
        extra.append({
            "n_samples": len(x),
            "fs": record.fs,
            "n_signals": record.sig_len,
            "nan": int(np.isnan(x).sum()),
            "raw_median": float(np.median(x)),
            "raw_std": float(np.std(x)),
            "raw_min": float(np.nanmin(x)),
            "raw_max": float(np.nanmax(x)),
            "n_annotations": len(ann.symbol),
            "symbols_all": list(ann.symbol),
            "unused_symbols": [s for i, s in enumerate(ann.symbol) if i not in matched],
        })
        ludb_x.append((x - np.median(x)) / (x.std() + 1e-6))        # standardized per record
        ludb_y.append(y)
        ludb_waves.append(waves)
    ludb_x, ludb_y = np.stack(ludb_x).astype(np.float32), np.stack(ludb_y)
    meta = pd.read_csv(Path(ludb_dir) / "ludb.csv")
    rhythm_of = dict(zip(meta["ID"], meta["Rhythms"]))
    ludb_rhythm = [str(rhythm_of[rec]).strip().splitlines()[0] for rec in range(1, 201)]
    perm = np.random.default_rng(0).permutation(200)               # one record = one patient
    ludb_split = {"train": perm[:140], "valid": perm[140:170], "test": perm[170:]}
    return ludb_x, np.stack(ludb_x12), names, ludb_y, ludb_waves, ludb_rhythm, ludb_split, extra


if __name__ == "__main__":
    ludb_dir, cache = sys.argv[1], Path(sys.argv[2])
    ludb_x, ludb_x12, lead_names, ludb_y, ludb_waves, ludb_rhythm, ludb_split, extra = load(ludb_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # one row per wave: the record it belongs to is a column, not something to be inferred
    waves = np.array([[rec, *w] for rec, ws in enumerate(ludb_waves) for w in ws], dtype=np.int64)
    np.savez_compressed(
        cache, x=ludb_x, x12=ludb_x12, lead_names=np.array(lead_names), y=ludb_y,
        rhythm=np.array(ludb_rhythm), waves=waves,   # waves: record, class, onset, peak, offset
        split_train=ludb_split["train"], split_valid=ludb_split["valid"], split_test=ludb_split["test"],
    )
    pd.DataFrame([{k: (",".join(map(str, v)) if k.endswith("symbols") else v)
                   for k, v in e.items()} | {"record": r} for r, e in enumerate(extra)]).to_csv(
        cache.parent / "ludb_records.csv", index=False)
    print("saved", cache, f"{cache.stat().st_size / 1e6:.1f} MB", "+ ludb_records.csv")
    print("signals", ludb_x.shape, "labels", ludb_y.shape, {s: len(v) for s, v in ludb_split.items()})
    print("fraction of the samples in each class:",
          dict(zip(WAVES, np.round(np.bincount(ludb_y.ravel()) / ludb_y.size, 3))))
