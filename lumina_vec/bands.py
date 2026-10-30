"""Shading bands: turn smooth gradients into clean, contour-like colour steps.

vtracer grows regions by merging similar neighbours, so a smooth gradient chains into one
flat region whatever the threshold. Here we quantise explicitly first:
  OKLab k-means palette (frequency-flattened sampling)
  -> low-contrast pixels: labels from a Gaussian-smoothed copy + spatial vote, so band
     boundaries follow smooth iso-contours (printable curves, no staircase)
  -> high-contrast pixels (lines, texture, eye highlights): raw per-pixel labels.
"""
import math, numpy as np, cv2
from .color import srgb_to_oklab, flat_sample


def _weights(l_weight, c_weight):
    return np.array([l_weight, c_weight, c_weight], np.float32)


def fit_palette(rgb, k, sample=200_000, flatten=0.5, l_weight=1.3, c_weight=2.0, mask=None):
    lab = srgb_to_oklab(rgb).reshape(-1, 3).astype(np.float32) * _weights(l_weight, c_weight)
    if mask is not None:
        sel = np.flatnonzero(mask.ravel())
        if len(sel) < k * 4:
            return np.zeros((0, 3), np.float32)
        lab = lab[sel]
        idx = flat_sample(rgb.reshape(-1, 3)[sel], sample, flatten)
    else:
        idx = flat_sample(rgb.reshape(-1, 3), sample, flatten)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    _, _, cen = cv2.kmeans(lab[idx], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return cen


def assign(rgb, cen, l_weight=1.3, c_weight=2.0):
    """Nearest palette entry per pixel. argmin ||x-c||^2 = argmin (||c||^2 - 2 x.c), done as one
    float32 matmul per chunk instead of materialising an (N, k, 3) difference array."""
    lw = srgb_to_oklab(rgb).reshape(-1, 3).astype(np.float32) * _weights(l_weight, c_weight)
    c = np.ascontiguousarray(cen, np.float32)
    cn = (c * c).sum(1)
    labels = np.empty(len(lw), np.int32)
    for s in range(0, len(lw), 1_000_000):
        labels[s:s + 1_000_000] = (cn[None, :] - 2.0 * (lw[s:s + 1_000_000] @ c.T)).argmin(1)
    return labels.reshape(rgb.shape[:2])


def soft_vote(labels, sigma, mask=None):
    """Gaussian-weighted majority vote. With `mask`, only masked pixels vote/are changed."""
    if sigma <= 0:
        return labels
    best = np.full(labels.shape, -1.0, np.float32); out = labels.copy()
    m_f = None if mask is None else mask.astype(np.float32)
    for c in np.unique(labels if mask is None else labels[mask]):
        ind = (labels == c).astype(np.float32)
        if m_f is not None:
            ind *= m_f
        p = cv2.GaussianBlur(ind, (0, 0), sigma)
        upd = p > best
        if mask is not None:
            upd &= mask
        out[upd] = c; best[upd] = p[upd]
    return out


def masked_blur(rgb, mask, sigma):
    """Normalised convolution: blur using only `mask` pixels, so dark lines don't bleed
    into the skin while low-frequency mottling (which makes band edges wavy) is removed."""
    m = cv2.dilate((~mask).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0   # stay clear of edges
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * mf[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)[..., None]
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), rgb)
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_blur(rgb, low, sigma, near_sigma=2.0, ramp=(4, 24)):
    """Blur radius grows with distance from the nearest edge: near outlines a small blur keeps
    real shadows (no pale halo along the jaw), in open areas a large blur gives smooth contours."""
    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(low)
    w = np.clip((d - ramp[0]) / (ramp[1] - ramp[0]), 0, 1)[..., None].astype(np.float32)
    near = masked_blur(rgb, low, near_sigma).astype(np.float32)
    far = masked_blur(rgb, low, sigma).astype(np.float32)
    return np.clip(near * (1 - w) + far * w, 0, 255).astype(np.uint8)


def edge_strength(rgb):
    L = srgb_to_oklab(rgb)[..., 0].astype(np.float32)
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.dilate(np.hypot(gx, gy), np.ones((3, 3), np.uint8))


def merge_specks(labels, rgb, max_area=48, keep_de=0.10, iters=2):
    """Merge small connected regions into their most similar neighbour class - unless the
    region contrasts strongly with it (OKLab dE > keep_de): eye highlights, nose dots and
    mouth lines are small but high-contrast, noise islands are small and low-contrast."""
    from skimage.measure import label as cc_label
    lab = srgb_to_oklab(rgb).reshape(-1, 3)
    n_cls = labels.max() + 1
    cnt = np.bincount(labels.ravel(), minlength=n_cls).astype(np.float64)
    cen = np.stack([np.bincount(labels.ravel(), lab[:, i], minlength=n_cls) for i in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    H, W = labels.shape
    for _ in range(iters):
        reg = cc_label(labels + 1, connectivity=1, background=0)
        sizes = np.bincount(reg.ravel())
        small = sizes < max_area; small[0] = False
        if not small[reg].any():
            break
        # 4-neighbour pairs (pixel in small region, neighbour in a different region)
        pairs = []
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            ys, xs = slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))
            yo, xo = slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx))
            ra, rb = reg[yo, xo], reg[ys, xs]
            la, lb = labels[yo, xo], labels[ys, xs]
            m = small[ra] & (ra != rb)
            pairs.append(np.stack([ra[m], la[m], lb[m]], 1))
        p = np.concatenate(pairs)
        if not len(p):
            break
        d = ((cen[p[:, 1]] - cen[p[:, 2]]) ** 2).sum(1) ** 0.5
        # per small region: neighbour class with the smallest colour distance
        order = np.lexsort((d, p[:, 0])); p, d = p[order], d[order]
        first = np.r_[True, p[1:, 0] != p[:-1, 0]]
        regs, tgt, dist = p[first, 0], p[first, 2], d[first]
        merge = dist < keep_de
        lut = np.full(reg.max() + 1, -1, np.int64); lut[regs[merge]] = tgt[merge]
        sel = lut[reg] >= 0
        labels = np.where(sel, lut[reg], labels)
    return labels


def printable_labels(labels, min_px, rgb=None, distinct_de=0.07, narrow_frac=0.5):
    """Print mode: decide per WHOLE colour region, not per pixel (per-pixel decisions chopped one
    line into printable/unprintable pieces -> dotted lines).
      - mostly-wide region: keep its wide part, slivers go to neighbours (as before);
      - mostly-narrow region (a line): if its colour is distinct from the colours on BOTH sides
        (a real line: outline, hair strand, chromatic-aberration edge) keep the whole region and
        thicken it to min_px; if it is close to one side (an anti-alias blend between two fills)
        drop the whole region into its neighbours.
    The image stays fully covered (no holes in any layer)."""
    from scipy import ndimage as ndi
    from skimage.measure import label as cc_label
    d = int(math.ceil(min_px))
    if d < 2:
        return labels
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    keep = np.zeros(labels.shape, bool)
    for c in np.unique(labels):
        keep |= cv2.morphologyEx((labels == c).astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    if keep.all():
        return labels
    lab = srgb_to_oklab(rgb) if rgb is not None else None
    reg = cc_label(labels + 1, connectivity=1, background=0)
    area = np.bincount(reg.ravel()); kept = np.bincount(reg.ravel(), weights=keep.ravel())
    H, W = labels.shape
    lines = np.zeros(labels.shape, bool)
    drop = ~keep
    for r, sl in enumerate(ndi.find_objects(reg), start=1):
        if sl is None or kept[r] >= narrow_frac * area[r]:
            continue                                   # mostly wide: per-pixel rule is fine
        ys = slice(max(0, sl[0].start - 2), min(H, sl[0].stop + 2))
        xs = slice(max(0, sl[1].start - 2), min(W, sl[1].stop + 2))
        m = reg[ys, xs] == r
        drop[ys, xs] |= m                              # default: whole region goes
        if lab is None or area[r] < 2 * d * d:
            continue
        ring = ndi.binary_dilation(m, iterations=2) & ~m
        if not ring.any():
            continue
        own = np.median(lab[ys, xs][m], 0)
        dist = np.sqrt(((lab[ys, xs][ring] - own) ** 2).sum(1))
        if np.quantile(dist, 0.15) >= distinct_de:     # distinct from every side -> a line
            lines[ys, xs] |= m
    out = labels
    gone = drop & ~lines
    if gone.any():
        _, (iy, ix) = ndi.distance_transform_edt(gone, return_indices=True)
        out = labels[iy, ix]
    if lines.any():                                    # thicken whole lines uniformly
        grown = cv2.dilate(lines.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (d // 2 * 2 + 1,) * 2)) > 0
        grown &= ~(keep & ~drop & (cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0))
        _, (sy, sx) = ndi.distance_transform_edt(~lines, return_indices=True)
        out = np.where(grown, labels[sy, sx], out)
    return out


def posterize(rgb, k=112, vote_sigma=1.0, l_weight=1.3, c_weight=2.0, edge_thr=0.4,
              blur_sigma=0.0, smooth_src=None, speck_area=48, speck_keep_de=0.10, k_detail=48, progress=None, min_px=0, **_):
    """Return (posterised rgb, palette hex list).
    blur_sigma>0: low-contrast pixels are labelled from a Gaussian-blurred copy (smooth
    contours); vote only among low-contrast pixels so bands don't eat lines."""
    step = progress or (lambda *_: None)
    step(0.0, "调色板")
    edges = edge_strength(rgb) > edge_thr if edge_thr else np.zeros(rgb.shape[:2], bool)
    cen = fit_palette(rgb, k, l_weight=l_weight, c_weight=c_weight)
    if k_detail and edges.any():
        # dedicated colours for high-contrast pixels: a tiny mouth or nose line otherwise has no
        # palette entry of its own and snaps to the surrounding skin tone
        cen = np.concatenate([cen, fit_palette(rgb, k_detail, l_weight=l_weight, c_weight=c_weight, mask=edges)])
    step(0.2, "分配颜色")
    raw = assign(rgb, cen, l_weight, c_weight)
    step(0.35, "平滑色带")
    if blur_sigma > 0:
        src = rgb if smooth_src is None else smooth_src
        low = ~edges
        smooth = assign(adaptive_blur(src, low, blur_sigma), cen, l_weight, c_weight)
        step(0.6, "空间投票")
        labels = np.where(low, soft_vote(np.where(low, smooth, raw), vote_sigma, low), raw)
    else:
        labels = np.where(edges, raw, soft_vote(raw, vote_sigma))
    step(0.85, "清理碎块")
    if speck_area:
        labels = merge_specks(labels, rgb, max_area=speck_area, keep_de=speck_keep_de)
    if min_px:
        step(0.92, "可打印性")
        labels = printable_labels(labels, min_px, rgb)
    # palette colour = median of the source pixels in each final label
    used = np.unique(labels)
    flat_l, flat_s = labels.ravel(), rgb.reshape(-1, 3)
    order = np.argsort(flat_l, kind="stable"); sl = flat_l[order]; ss = flat_s[order]
    b = np.searchsorted(sl, np.arange(labels.max() + 2))
    pal = np.zeros((labels.max() + 1, 3), np.uint8)
    for c in used:
        pal[c] = np.median(ss[b[c]:b[c + 1]], 0)
    return pal[labels], ["#%02x%02x%02x" % tuple(int(v) for v in pal[c]) for c in used]
