# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MESFET5198 - Lumina-Vec (https://github.com/MESFET5198/Lumina-Vec)
"""Eye fine channel (prototype). An eye is drawn by the artist as clean layers (iris bands,
pupil, highlights, lid shadow); the global pipeline blurs them (AI clean, smoothing, a palette
shared with the whole image). Inside each eye: original pixels, a palette of the eye's own
colours, only a light mode filter, nothing merged."""
import numpy as np
import cv2

from .bands import fit_palette, assign


def eye_patch(post, src, boxes, k=12, feather=0.15):
    """post: label image (work px), src: original at work size. boxes: (x, y, w, h) work px.
    Returns (new label image, mask of replaced pixels)."""
    from .contour import smooth_labels
    out = post.copy()
    H, W = post.shape[:2]
    mask = np.zeros((H, W), bool)
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y)); x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        sub = np.ascontiguousarray(src[y0:y1, x0:x1])
        cen = fit_palette(sub, k)
        lab = assign(sub, cen).reshape(sub.shape[:2])
        cols = np.zeros((lab.max() + 1, 3), np.uint8)
        flat = lab.ravel(); px = sub.reshape(-1, 3)
        for c in np.unique(flat):
            cols[c] = np.median(px[flat == c], 0)
        loc = smooth_labels(cols[lab], sigma=0.6)
        # elliptical mask (an eye sits in the middle of its box), shrunk a little so the patch
        # meets the surrounding skin where both are skin anyway
        m = np.zeros(sub.shape[:2], np.uint8)
        cv2.ellipse(m, ((sub.shape[1] / 2, sub.shape[0] / 2), (sub.shape[1] * (1 - feather), sub.shape[0] * (1 - feather)), 0), 1, -1)
        m = m > 0
        out[y0:y1, x0:x1][m] = loc[m]
        mask[y0:y1, x0:x1] |= m
    return out, mask
