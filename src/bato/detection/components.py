"""8-connected component labelling of a sparse boolean mask with numpy only (no scipy dependency).

Works on the coordinates of the True pixels, so cost scales with the number of candidates, not the raster.
Labels are made equal along 8-neighbour edges by min-label propagation with pointer jumping until stable.
"""

from __future__ import annotations

import numpy as np

# forward neighbours (row, col): right, down-left, down, down-right -> every 8-neighbour pair once
_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))


def label_sparse(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(flat indices of True pixels, component root per pixel); roots are positions into the index array."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    h, w = mask.shape
    idx = np.flatnonzero(mask)
    n = idx.size
    labels = np.arange(n, dtype=np.int64)
    if n == 0:
        return idx, labels
    rows, cols = idx // w, idx % w
    a_parts, b_parts = [], []
    for dr, dc in _OFFSETS:
        ok = (rows + dr < h) & (cols + dc >= 0) & (cols + dc < w)
        target = idx[ok] + dr * w + dc
        pos = np.searchsorted(idx, target)
        pos_c = np.minimum(pos, n - 1)
        hit = idx[pos_c] == target
        a_parts.append(np.flatnonzero(ok)[hit])
        b_parts.append(pos_c[hit])
    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    while True:
        la, lb = labels[a], labels[b]
        low = np.minimum(la, lb)
        before = labels.copy()
        np.minimum.at(labels, a, low)
        np.minimum.at(labels, b, low)
        np.minimum.at(labels, la, low)            # also pull the current roots down (faster convergence)
        np.minimum.at(labels, lb, low)
        while True:                              # pointer jumping
            jumped = labels[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        if np.array_equal(labels, before):
            return idx, labels


def component_sizes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flat indices of True pixels, component root per pixel, size of each pixel's component)."""
    idx, roots = label_sparse(mask)
    if idx.size == 0:
        return idx, roots, np.zeros(0, dtype=np.int64)
    counts = np.bincount(roots, minlength=idx.size)
    return idx, roots, counts[roots]
