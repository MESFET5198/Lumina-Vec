"""Shading bands: turn smooth gradients into clean, contour-like colour steps.

vtracer grows regions by merging similar neighbours, so a smooth gradient chains into one
flat region whatever the threshold. Here we quantise explicitly first:
  smooth (edge-preserving) -> OKLab k-means palette (frequency-flattened sampling)
  -> spatial soft vote (removes speckle, rounds band edges) -> posterised image.
Band boundaries then follow iso-contours of the smoothed image, like Vectorizer.AI's bands.
"""
import numpy as np, cv2
from .color import srgb_to_oklab, flat_sample


def smooth_for_bands(rgb, iters=2, d=9, sc=12, ss=6):
    out = rgb
    for _ in range(iters):
        out = cv2.bilateralFilter(out, d, sc, ss)
    return out


def palette_kmeans(rgb, k, sample=200_000, flatten=0.5, l_weight=1.6, c_weight=1.0):
    """k-means in OKLab; lightness weighted up so clusters split along shading (light/dark)."""
    lab = srgb_to_oklab(rgb).reshape(-1, 3).astype(np.float32)
    w = np.array([l_weight, c_weight, c_weight], np.float32)
    idx = flat_sample(rgb.reshape(-1, 3), sample, flatten)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    _, _, cen = cv2.kmeans(lab[idx] * w, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lw = lab * w
    labels = np.empty(len(lab), np.int32)
    for s in range(0, len(lab), 400_000):
        labels[s:s + 400_000] = ((lw[s:s + 400_000, None] - cen[None]) ** 2).sum(-1).argmin(1)
    return labels.reshape(rgb.shape[:2])


def _relabel(rgb, labels):
    """Per-pixel nearest palette entry (palette = mean OKLab of each label)."""
    lab = srgb_to_oklab(rgb).reshape(-1, 3)
    ids = np.unique(labels)
    cen = np.stack([lab[labels.ravel() == c].mean(0) for c in ids])
    out = np.empty(len(lab), np.int32)
    for s in range(0, len(lab), 400_000):
        out[s:s + 400_000] = ids[((lab[s:s + 400_000, None] - cen[None]) ** 2).sum(-1).argmin(1)]
    return out.reshape(labels.shape)


def soft_vote(labels, sigma):
    if sigma <= 0:
        return labels
    best = np.full(labels.shape, -1.0, np.float32); out = labels.copy()
    for c in np.unique(labels):
        p = cv2.GaussianBlur((labels == c).astype(np.float32), (0, 0), sigma)
        m = p > best; out[m] = c; best[m] = p[m]
    return out


def edge_strength(rgb):
    L = srgb_to_oklab(rgb)[..., 0].astype(np.float32)
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.dilate(np.hypot(gx, gy), np.ones((3, 3), np.uint8))


def posterize(rgb, k=128, vote_sigma=1.2, smooth_iters=2, l_weight=1.6, color_src=None, edge_thr=0.12, c_weight=1.0):
    """Return (posterised rgb, palette hex list). Colours are the median of `color_src`
    (default: rgb) inside each final label, so bands keep the true tone.
    Smoothing + vote only act on low-contrast pixels (gradients); along strong edges
    (lines, knit texture) the raw per-pixel label is kept so detail survives."""
    raw = palette_kmeans(rgb, k, l_weight=l_weight, c_weight=c_weight)
    if smooth_iters or vote_sigma > 0:
        sm = smooth_for_bands(rgb, iters=smooth_iters) if smooth_iters else rgb
        soft = soft_vote(palette_kmeans(sm, k, l_weight=l_weight, c_weight=c_weight) if smooth_iters else raw, vote_sigma)
        if edge_thr:
            # map raw labels to the same palette as `soft` by nearest colour if palettes differ
            edges = edge_strength(rgb) > edge_thr
            labels = np.where(edges, raw if not smooth_iters else _relabel(rgb, soft), soft)
        else:
            labels = soft
    else:
        labels = raw
    src = rgb if color_src is None else color_src
    used = np.unique(labels)
    pal = np.zeros((labels.max() + 1, 3), np.uint8)
    flat_l, flat_s = labels.ravel(), src.reshape(-1, 3)
    order = np.argsort(flat_l, kind="stable"); sl = flat_l[order]; ss = flat_s[order]
    b = np.searchsorted(sl, np.arange(labels.max() + 2))
    for c in used:
        pal[c] = np.median(ss[b[c]:b[c + 1]], 0)
    return pal[labels], ["#%02x%02x%02x" % tuple(int(v) for v in pal[c]) for c in used]
