# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MESFET5198 - Lumina-Vec (https://github.com/MESFET5198/Lumina-Vec)
"""Global palette (the reference look: ~70-110 colours, each reused on many shapes, neighbouring
palette colours >= ~2-3 OKLab*100 apart).

Every layer - colour regions, details, lines - only uses colours from this one palette, the way
an artist mixes a set of paints first and then picks one for each area. Adjacent regions that
end up with the same colour merge, which gives fewer, larger, calmer shapes.

Merging is by colour DISTANCE only (closest pair first), the merged colour is the area-weighted
mean: small but distinct colours (eye highlights, iris) survive until the budget is reached,
while near-duplicates of large areas (gradient steps, anti-alias tones) are folded together."""
import re
import numpy as np
import cv2

from .color import srgb_to_oklab


def _lab(rgb):
    return srgb_to_oklab(np.asarray(rgb, np.uint8).reshape(1, -1, 3))[0].astype(np.float64) * 100


def _lab_w(rgb):
    """OKLab*100 with chroma weighted up for light colours: white vs pale skin differ mostly in
    a little chroma, invisible to plain OKLab distance but obvious on a face (it went paper-white)."""
    lab = _lab(rgb)
    f = 1.0 + 2.0 * np.clip((lab[:, 0] - 80) / 15, 0, 1)
    lab[:, 1:] *= f[:, None]
    return lab


def build(colors, weights, n=96, min_de=2.5, max_de=4.0):
    """colors: (k,3) uint8, weights: (k,). Returns (palette (m,3) uint8, index (k,) into it).
    Merges the closest pair until there are <= n colours AND no two are closer than min_de;
    never merges a pair further apart than max_de (the budget n is a target, not forced)."""
    colors = np.asarray(colors, np.uint8); w = np.asarray(weights, np.float64).copy()
    if len(colors) > 800:
        # pairwise merging is O(k^2) memory: first reduce to 800 seeds by k-means on the colour
        # LIST (unweighted, so small but distinct colours keep their own seed)
        lab0 = _lab_w(colors).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        _, lb, _ = cv2.kmeans(lab0, 800, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        lb = lb.ravel()
        sw = np.bincount(lb, weights=w, minlength=800)
        seed = np.stack([np.bincount(lb, weights=w * colors[:, c], minlength=800) for c in range(3)], 1)
        seed = np.clip(np.round(seed / np.maximum(sw, 1e-9)[:, None]), 0, 255).astype(np.uint8)
        pal, idx = build(seed, sw, n=n, min_de=min_de, max_de=max_de)
        return pal, idx[lb]
    k = len(colors)
    lab = _lab_w(colors)
    rgb = colors.astype(np.float64)
    alive = np.ones(k, bool); parent = np.arange(k)
    D = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    count = k
    while count > 1:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if (count <= n and D[i, j] >= min_de) or D[i, j] >= max_de:
            break
        a, b = (i, j) if w[i] >= w[j] else (j, i)          # b folds into a
        tot = w[a] + w[b]
        lab[a] = (lab[a] * w[a] + lab[b] * w[b]) / tot
        rgb[a] = (rgb[a] * w[a] + rgb[b] * w[b]) / tot
        w[a] = tot
        alive[b] = False; parent[parent == b] = a; count -= 1
        D[b, :] = np.inf; D[:, b] = np.inf
        d = np.sqrt(((lab[alive] - lab[a]) ** 2).sum(-1))
        idx = np.flatnonzero(alive)
        D[a, idx] = d; D[idx, a] = d; D[a, a] = np.inf
    keep = np.flatnonzero(alive)
    remap = -np.ones(k, int); remap[keep] = np.arange(len(keep))
    return np.clip(np.round(rgb[keep]), 0, 255).astype(np.uint8), remap[parent]


def nearest(pal, colors):
    pl, cl = _lab(pal), _lab(colors)
    return ((cl[:, None] - pl[None]) ** 2).sum(-1).argmin(1)


_FILL = re.compile(r'fill="#([0-9a-fA-F]{6})"')


def path_colors(paths):
    out = []
    for p in paths:
        m = _FILL.search(p)
        out.append(m.group(1).lower() if m else None)
    return out


def recolor(paths, mapping):
    """Replace fill colours (hex without #, lower) using mapping {old: new}."""
    def rep(p):
        return _FILL.sub(lambda m: 'fill="#%s"' % mapping.get(m.group(1).lower(), m.group(1)), p, count=1)
    return [rep(p) for p in paths]


def unify(label_rgb, extra_paths, n=96, min_de=2.5, line_weight=40.0):
    """One palette for the colour layer (label image, weighted by pixel count) and the overlay
    paths (lines/details, `line_weight` px each). Returns (new label image, new extra paths, pal)."""
    H, W = label_rgb.shape[:2]
    key = (label_rgb[..., 0].astype(np.int64) << 16) | (label_rgb[..., 1].astype(np.int64) << 8) | label_rgb[..., 2]
    ukeys, inv, cnt = np.unique(key.ravel(), return_inverse=True, return_counts=True)
    cols = np.stack([(ukeys >> 16) & 255, (ukeys >> 8) & 255, ukeys & 255], 1).astype(np.uint8)
    hexes = path_colors(extra_paths)
    ex = sorted({h for h in hexes if h})
    ex_rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in ex], np.uint8).reshape(-1, 3)
    ex_w = np.array([line_weight * sum(1 for x in hexes if x == h) for h in ex], np.float64)
    allc = np.concatenate([cols, ex_rgb]); allw = np.concatenate([cnt.astype(np.float64), ex_w])
    pal, idx = build(allc, allw, n=n, min_de=min_de)
    new_lab = pal[idx[:len(cols)]][inv].reshape(H, W, 3)
    mapping = {h: "%02x%02x%02x" % tuple(pal[idx[len(cols) + i]]) for i, h in enumerate(ex)}
    return new_lab, recolor(extra_paths, mapping), pal


# ------------------------------------------------------------- region colours (v0.9 experiment)

def _region_map(label_rgb, barrier=None):
    from skimage import measure
    key = (label_rgb[..., 0].astype(np.int64) << 16) | (label_rgb[..., 1].astype(np.int64) << 8) | label_rgb[..., 2]
    key = key + 1
    if barrier is not None:
        key[barrier] = 0                 # line pixels: walls between regions (label 0)
    return measure.label(key, connectivity=1, background=0)


def region_colors(label_rgb, src, scale=0.5, barrier=None):
    """Shapes from the posterised map, COLOUR per connected region from a downscaled copy of the
    source. A shared colour bucket gave every member the bucket's median, so a big white
    background painted pale skin white; a region's own pixels cannot be outvoted by another
    region. Downscaling averages grain/texture away: the colour the artist meant.
    Returns (region id map, region colours (n,3) uint8, region areas)."""
    from scipy import ndimage as ndi
    H, W = label_rgb.shape[:2]
    reg = _region_map(label_rgb, barrier)
    small = cv2.resize(src, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    n = reg.max() + 1
    idx = np.arange(n)
    # mean, not median: a large region's median snaps to its lightest plateau (a face went
    # whiter than the original), the mean keeps its overall tone (blush, soft shading)
    cols = np.stack([np.asarray(ndi.mean(sm[..., c].astype(np.float64), reg, index=idx)) for c in range(3)], 1)
    # tiny regions (a few px: highlights, pupils) are washed out at 50 %: use full-res pixels
    area = np.bincount(reg.ravel(), minlength=n)
    tiny = area < 40
    if tiny.any():
        full = np.stack([np.asarray(ndi.median(src[..., c], reg, index=idx)) for c in range(3)], 1)
        cols[tiny] = full[tiny]
    return reg, np.nan_to_num(cols).astype(np.uint8), area


def merge_adjacent(reg, cols, area, max_de=2.0):
    """Merge neighbouring regions whose colours are within max_de (OKLab*100), closest pairs
    first, colours area-weighted. Returns (new region map, colours, areas)."""
    n = len(cols)
    a = np.concatenate([np.stack([reg[:, :-1].ravel(), reg[:, 1:].ravel()], 1),
                        np.stack([reg[:-1].ravel(), reg[1:].ravel()], 1)])
    a = a[a[:, 0] != a[:, 1]]
    a = np.unique(np.sort(a, 1), axis=0)
    lab = _lab(cols); w = area.astype(np.float64).copy(); rgb = cols.astype(np.float64)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for _ in range(3):                                   # colours move as regions merge
        d = np.sqrt(((lab[a[:, 0]] - lab[a[:, 1]]) ** 2).sum(1))
        changed = False
        for k in np.argsort(d):
            if d[k] >= max_de:
                break
            i, j = find(a[k, 0]), find(a[k, 1])
            if i == j or i == 0 or j == 0:
                continue
            if np.sqrt(((lab[i] - lab[j]) ** 2).sum()) >= max_de:
                continue
            if w[i] < w[j]:
                i, j = j, i
            t = w[i] + w[j]
            lab[i] = (lab[i] * w[i] + lab[j] * w[j]) / t; rgb[i] = (rgb[i] * w[i] + rgb[j] * w[j]) / t
            w[i] = t; parent[j] = i; changed = True
        if not changed:
            break
        a = np.stack([np.array([find(x) for x in a[:, 0]]), np.array([find(x) for x in a[:, 1]])], 1)
        a = a[a[:, 0] != a[:, 1]]; a = np.unique(np.sort(a, 1), axis=0)
    root = np.array([find(i) for i in range(n)])
    uniq, newid = np.unique(root, return_inverse=True)
    return newid[reg], np.clip(np.round(rgb[uniq]), 0, 255).astype(np.uint8), w[uniq]


def region_pipeline(label_rgb, src, extra_paths, n=128, min_de=2.0, merge_de=2.0, scale=0.5, barrier=None,
                    max_de=2.5):
    """Colour pipeline: region colours from a downscaled source -> merge similar neighbours ->
    one global palette (regions by area + overlay paths). Returns (label image, overlay paths)."""
    reg, cols, area = region_colors(label_rgb, src, scale, barrier)
    reg, cols, area = merge_adjacent(reg, cols, area, merge_de)
    if barrier is not None and (reg == 0).any():
        # under the lines: continue the nearest region (lines are drawn on top)
        from scipy import ndimage as ndi
        _, (iy, ix) = ndi.distance_transform_edt(reg == 0, return_indices=True)
        reg = reg[iy, ix]
    hexes = path_colors(extra_paths)
    ex = sorted({h for h in hexes if h})
    ex_rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in ex], np.uint8).reshape(-1, 3)
    ex_w = np.array([40.0 * sum(1 for x in hexes if x == h) for h in ex], np.float64)
    pal, idx = build(np.concatenate([cols, ex_rgb]), np.concatenate([area, ex_w]), n=n, min_de=min_de,
                     max_de=max_de)
    out = pal[idx[:len(cols)]][reg]
    mapping = {h: "%02x%02x%02x" % tuple(pal[idx[len(cols) + i]]) for i, h in enumerate(ex)}
    return out, recolor(extra_paths, mapping)
