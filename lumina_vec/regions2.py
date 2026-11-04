# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MESFET5198 - Lumina-Vec (https://github.com/MESFET5198/Lumina-Vec)
"""Colour regions drawn like the lines: every boundary is a few smooth curves (one straight line
/ C / gentle S between real corners), not a pixel-faithful trace; tiny and thin bands are merged
into their most similar neighbour first, so shading reads as a few calm shapes."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import measure

from .color import srgb_to_oklab
from .draw import simple_pieces, _turn_corners, _resample1


def _regions(post):
    key = (post[..., 0].astype(np.int64) << 16) | (post[..., 1].astype(np.int64) << 8) | post[..., 2]
    return measure.label(key + 1, connectivity=1, background=0)


def merge_small(post, min_area=80, min_halfwidth=2.0, iters=3, keep_de=6.0, tiny=20, protect=None):
    """Regions smaller than min_area px, or thinner than 2*min_halfwidth, take the colour of the
    neighbour closest in colour (OKLab), largest-first so merges cascade sensibly."""
    out = post.copy()
    for _ in range(iters):
        reg = _regions(out)
        n = reg.max() + 1
        area = np.bincount(reg.ravel(), minlength=n)
        edge = np.zeros(reg.shape, bool)
        edge[:, 1:] |= reg[:, 1:] != reg[:, :-1]; edge[:, :-1] |= reg[:, 1:] != reg[:, :-1]
        edge[1:] |= reg[1:] != reg[:-1]; edge[:-1] |= reg[1:] != reg[:-1]
        dt = ndi.distance_transform_edt(~edge)
        hw = np.asarray(ndi.maximum(dt, reg, index=np.arange(n)))
        bad = (area < min_area) | (hw < min_halfwidth); bad[0] = False
        if protect is not None:          # regions inside a protected area (eyes) are never merged
            inside = np.bincount(reg[protect], minlength=n) > 0
            bad &= ~inside
        if not bad.any():
            break
        objs = ndi.find_objects(reg)
        lab = srgb_to_oklab(out).astype(np.float32)
        changed = 0
        for r in np.flatnonzero(bad):
            sl = objs[r - 1]
            if sl is None:
                continue
            y0, y1 = max(0, sl[0].start - 1), min(reg.shape[0], sl[0].stop + 1)
            x0, x1 = max(0, sl[1].start - 1), min(reg.shape[1], sl[1].stop + 1)
            sub = reg[y0:y1, x0:x1]; m = sub == r
            ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
            nb = np.unique(sub[ring])
            nb = nb[(nb > 0) & ~bad[nb]] if (~bad[nb]).any() else nb[nb > 0]
            if not len(nb):
                continue
            my = lab[y0:y1, x0:x1][m].mean(0)
            best, bd = None, 1e9
            for q in nb:
                qm = (sub == q) & ring
                c = lab[y0:y1, x0:x1][qm].mean(0)
                d = np.sum((c - my) ** 2)
                if d < bd:
                    bd, best = d, q
            # a small/thin region that clearly differs from every neighbour is a deliberate detail
            # (highlight on a belt, eye detail, ruffle stripe): keep it unless it is a speck
            if np.sqrt(bd) * 100 > keep_de and area[r] >= tiny:
                continue
            qm = (sub == best) & ring
            col = out[y0:y1, x0:x1][qm][0]
            out[y0:y1, x0:x1][m] = col
            changed += 1
        if not changed:
            break
    return out


def fit_closed_smooth(poly, tol=1.5, corner_deg=50.0, smooth=1.5):
    """Closed boundary -> list of 4x2 cubics: cut only at real corners, each run = one straight
    line / C (gentle S if long), loose tolerance."""
    p = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    p = _resample1(np.vstack([p, p[:1]]))[:-1]
    n = len(p)
    if n < 8:
        return []
    k = min(10, n)
    pw = np.vstack([p[-k:], p, p[:k]])
    pw = ndi.gaussian_filter1d(pw, smooth, axis=0, mode="nearest")[k:k + n]
    ext = np.vstack([pw[-5:], pw, pw[:5]])
    cs = [c - 5 for c in _turn_corners(ext, corner_deg, 5) if 5 <= c < n + 5]
    cs = sorted(set(int(c) % n for c in cs))
    if len(cs) < 2:                                     # no corners: split at two far points
        d = np.hypot(*(pw - pw[0]).T); j = int(d.argmax())
        cs = sorted({0, j}) if j > 0 else [0, n // 2]
    segs = []
    for a, b in zip(cs, cs[1:] + [cs[0] + n]):
        idx = np.arange(a, b + 1) % n
        run = pw[idx]
        if len(run) < 2:
            continue
        for c, _, _ in simple_pieces(run, tol_min=tol, tol_rel=0.05, tol_max=tol * 2, s_len=40.0, s_px=1.5):
            segs.append(c)
    return segs


def region_paths_smooth(post, tol=1.5, sigma=0.5, ss=2, level=0.5, min_area=6):
    """Like contour.region_paths (stacked, largest first), boundaries drawn with fit_closed_smooth."""
    H, W = post.shape[:2]
    reg = _regions(post)
    objs = ndi.find_objects(reg)
    area = np.bincount(reg.ravel())
    order = sorted(range(1, len(area)), key=lambda r: -area[r])
    rank = np.zeros(len(area), np.int64); rank[order] = np.arange(len(order))
    rmap = rank[reg]
    items = []
    for i, r in enumerate(order):
        sl = objs[r - 1]
        if sl is None or (area[r] < min_area and i > 0):
            continue
        y0, y1 = max(0, sl[0].start - 3), min(H, sl[0].stop + 3)
        x0, x1 = max(0, sl[1].start - 3), min(W, sl[1].stop + 3)
        own = reg[y0:y1, x0:x1] == r
        m = own | (rmap[y0:y1, x0:x1] > i)
        lab, _ = ndi.label(m)
        ids = np.unique(lab[own]); m = np.isin(lab, ids[ids > 0])
        yy, xx = np.argwhere(own)[0]; col = post[y0 + yy, x0 + xx]
        hexc = "#%02x%02x%02x" % tuple(int(v) for v in col)
        if i == 0:
            items.append(f'<path d="M0,0H{W}V{H}H0Z" fill="{hexc}"/>'); continue
        f = np.pad(m.astype(np.float32), 2)
        f = cv2.resize(f, None, fx=ss, fy=ss, interpolation=cv2.INTER_LINEAR)
        f = cv2.GaussianBlur(f, (0, 0), sigma * ss)
        parts = []
        for c in measure.find_contours(f, level):
            if len(c) < 8:
                continue
            xy = (c[:, ::-1] + 0.5) / ss + np.array([x0 - 2, y0 - 2])
            segs = fit_closed_smooth(xy, tol=tol)
            if not segs:
                continue
            parts.append("M%.2f,%.2f" % tuple(segs[0][0]) + "".join(
                "C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*s[1], *s[2], *s[3]) for s in segs) + "Z")
        if parts:
            items.append(f'<path d="{"".join(parts)}" fill="{hexc}" fill-rule="evenodd"/>')
    return items
