from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def paired_interval(a, b, families, *, ratio=True, repeats=4000, seed=960631):
    """Resample paired independent units within prespecified workload families."""
    a, b, families = np.asarray(a), np.asarray(b), np.asarray(families)
    if not len(a):
        return [None, None, None]
    if ratio and b.sum() == 0:
        return [None, None, None]  # 0/0 is undefined, not a zero cost ratio.
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(families == x) for x in np.unique(families)]
    values = []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        if ratio and b[ix].sum() == 0:
            continue
        values.append(float(a[ix].sum() / b[ix].sum()) if ratio else float((a[ix] - b[ix]).mean()))
    center = float(a.sum() / b.sum()) if ratio else float((a - b).mean())
    return [center, *np.quantile(values, [0.025, 0.975]).tolist()]
