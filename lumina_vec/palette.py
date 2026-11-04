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

from .color import srgb_to_oklab


def _lab(rgb):
    return srgb_to_oklab(np.asarray(rgb, np.uint8).reshape(1, -1, 3))[0].astype(np.float64) * 100


def build(colors, weights, n=96, min_de=2.5):
    """colors: (k,3) uint8, weights: (k,). Returns (palette (m,3) uint8, index (k,) into it).
    Merges the closest pair until there are <= n colours AND no two are closer than min_de."""
    colors = np.asarray(colors, np.uint8); w = np.asarray(weights, np.float64).copy()
    k = len(colors)
    lab = _lab(colors)
    rgb = colors.astype(np.float64)
    alive = np.ones(k, bool); parent = np.arange(k)
    D = np.sqrt(((lab[:, None] - lab[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    count = k
    while count > 1:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if count <= n and D[i, j] >= min_de:
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
