# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MESFET5198 - Lumina-Vec (https://github.com/MESFET5198/Lumina-Vec)
"""Line pipeline v4: STROKES.

Line shape used to be decided pixel by pixel (model map x the original's ink), so every weak
pixel cut a line in two (white-hair image: 1,850 pieces after the models -> 4,841 after the
ink test). Here the models decide WHERE a line runs, geometry decides how pieces CONNECT, and
the original only adjusts each stroke's WIDTH and COLOUR - nothing can break a stroke:

1. P = max(manga/0.6, anime/0.5); lineart_anime specks that touch no manga_line line dropped
2. centreline graph at 2x: branches between junctions
3. at every junction the two branches that continue each other (most opposite directions)
   become one stroke; free ends that nearly meet and point at each other are bridged
4. width along the stroke = ink measured across the line in the original, clamped between a
   minimum and the model's own width, smoothed along the stroke; free ends taper to a point
5. one colour per stroke (darker half of the original under its centreline)
6. outline = centreline +- width/2, fitted with cubic Beziers
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .contour import fit_closed, ink_field

_NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _order(pix, ends):
    s = {tuple(p) for p in pix}
    start = tuple(ends[0]) if len(ends) else tuple(pix[0])
    path = [start]; s.discard(start); cur = start
    while True:
        for dy, dx in _STEPS:
            q = (cur[0] + dy, cur[1] + dx)
            if q in s:
                break
        else:
            break
        s.discard(q); path.append(q); cur = q
    return np.array(path, float)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def line_prob(M, A, manga_norm=0.6, anime_norm=0.5, speck_len=30):
    """Combined line probability; anime-only pieces that touch no manga line and are short
    (its specks) are removed."""
    Mn = np.clip(M / manga_norm, 0, 1); An = np.clip(A / anime_norm, 0, 1)
    am = An > 0.5
    near_m = cv2.dilate((Mn > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n, cc, st, _ = cv2.connectedComponentsWithStats(am.astype(np.uint8), connectivity=8)
    touch = np.bincount(cc[near_m & am], minlength=n) > 0
    sk = skeletonize(am)
    L = np.bincount(cc[sk], minlength=n)
    drop = (~touch) & (L < speck_len); drop[0] = False
    An = np.where(drop[cc], 0, An)
    return np.maximum(Mn, An).astype(np.float32)


def blob_mask(P, cov, wide_px=5.0, ink=0.35, core_area=30, hole_px=60):
    """Thick dark masses (eyeliner + lash roots, dark accents): drawn as FILLED shapes, not
    strokes - a skeleton of a blob is a web of loops. Core = model mask deeper than wide_px/2
    with real ink; grown back to the mask's edge."""
    mask = P > 0.5
    # a lash cluster is a dark mass full of tiny gaps: close those gaps first (their skeleton is
    # a web of loops), so the mass reads as one thick shape
    holes = ndi.binary_fill_holes(mask) & ~mask
    hl, hn = ndi.label(holes)
    small = np.bincount(hl.ravel(), minlength=hn + 1) < hole_px; small[0] = False
    filled = mask | small[hl]
    d = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    core = (d > wide_px / 2) & (cov > ink)
    # only real masses: a thick spot along an ordinary line is not one (it chopped hair lines)
    n0, c0, s0, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    ok = np.zeros(n0, bool); ok[1:] = s0[1:, cv2.CC_STAT_AREA] >= core_area
    core = ok[c0]
    if not core.any():
        return np.zeros_like(mask)
    r = int(np.ceil(wide_px / 2)) + 1
    grown = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    blob = grown & filled
    n, cc, st, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= 12
    return big[cc]


def line_spacing(P, thr=0.5, win=25):
    """Typical gap between neighbouring lines around each pixel (work px): 2 x the mean distance
    from background pixels to the nearest line, averaged over a window. Small = dense area."""
    mask = P > thr
    D = ndi.distance_transform_edt(~mask).astype(np.float32)
    bg = (~mask).astype(np.float32)
    num = cv2.boxFilter(D * bg, -1, (win, win), normalize=False)
    den = cv2.boxFilter(bg, -1, (win, win), normalize=False)
    return 2 * num / np.maximum(den, 1)


def build_strokes(P, ss=2, thr=0.5, join_cos=-0.7, bridge_px=4.0, bridge_cos=0.8, min_len=2.5, blob=None,
                  spacing=None, bridge_max=12.0, bridge_frac=0.7, bridge_cos_loose=0.6):
    """Centreline strokes from the probability map P (work px).
    Returns list of dict(xy (n,2) in work px, mw (n,) model width px, free=[start,end])."""
    up = cv2.resize(P, None, fx=ss, fy=ss, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 0.5 * ss)
    mask = up > thr
    skel = skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)
    near_blob = None
    if blob is not None and blob.any():
        bu = cv2.resize(blob.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        skel &= ~bu                                   # strokes stop at the blob's edge...
        near_blob = cv2.dilate(bu.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, _NB, borderType=cv2.BORDER_CONSTANT) * skel
    junc = skel & (nb >= 3)
    jl, nj = ndi.label(junc, structure=np.ones((3, 3)))
    jcent = np.array(ndi.center_of_mass(junc, jl, range(1, nj + 1))) if nj else np.zeros((0, 2))
    jl_d = cv2.dilate(jl.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)
    br = skel & ~junc
    bl, _ = ndi.label(br, structure=np.ones((3, 3)))
    branches = []                          # each: path (k,2 row/col at ss), junction id per end
    for i, sl in enumerate(ndi.find_objects(bl), start=1):
        if sl is None:
            continue
        sub = bl[sl] == i
        pix = np.argwhere(sub) + [sl[0].start, sl[1].start]
        cnt = cv2.filter2D(sub.astype(np.uint8), -1, _NB, borderType=cv2.BORDER_CONSTANT)[sub]
        path = _order(pix, pix[cnt <= 1])
        if len(path) < 1:
            continue
        js = [int(jl_d[int(path[0][0]), int(path[0][1])]), int(jl_d[int(path[-1][0]), int(path[-1][1])])]
        if near_blob is not None:
            for e, q in ((0, path[0]), (1, path[-1])):
                if js[e] == 0 and near_blob[int(q[0]), int(q[1])]:
                    js[e] = -1                        # attached to a blob
        if len(path) == 1:
            js[1] = js[0] if js[1] == 0 else js[1]
        branches.append(dict(p=path, j=js))
    nb_ = len(branches)

    def end_dir(b, e, k=6):
        p = branches[b]["p"]
        if e == 0:
            return _unit(p[min(k, len(p) - 1)] - p[0])
        return _unit(p[max(-k - 1, -len(p))] - p[-1])
    # spurs: short branch with a free end hanging off a junction
    blen = [np.hypot(*np.diff(b["p"], axis=0).T).sum() if len(b["p"]) > 1 else 0 for b in branches]
    keep = np.ones(nb_, bool)
    for b, br_ in enumerate(branches):
        j0, j1 = br_["j"]
        if (j0 > 0) != (j1 > 0) and min(j0, j1) >= 0:
            w = dist[int(br_["p"][len(br_["p"]) // 2][0]), int(br_["p"][len(br_["p"]) // 2][1])]
            if blen[b] < 1.5 * w + 2:
                keep[b] = False
    link = {}                              # (b, e) -> (b2, e2, junction id or 0)
    # 3a. continuation through junctions
    ends_at = {}
    for b in range(nb_):
        if not keep[b]:
            continue
        for e in (0, 1):
            j = branches[b]["j"][e]
            if j > 0:
                ends_at.setdefault(j, []).append((b, e))
    for j, items in ends_at.items():
        dirs = [end_dir(b, e) for b, e in items]
        cand = []
        for a in range(len(items)):
            for c in range(a + 1, len(items)):
                if items[a][0] == items[c][0]:
                    continue
                cs = float(np.dot(dirs[a], dirs[c]))
                if cs < join_cos:
                    cand.append((cs, a, c))
        used = set()
        for cs, a, c in sorted(cand):
            if a in used or c in used:
                continue
            used |= {a, c}
            link[items[a]] = (*items[c], j); link[items[c]] = (*items[a], j)
    # 3b. bridge free ends that nearly meet and point at each other
    free = [(b, e) for b in range(nb_) if keep[b] for e in (0, 1)
            if branches[b]["j"][e] == 0 and not (e == 1 and len(branches[b]["p"]) == 1)]
    if len(free) > 1:
        pts = np.array([branches[b]["p"][0 if e == 0 else -1] for b, e in free])
        tree = cKDTree(pts)
        reach = bridge_max if spacing is not None else bridge_px
        pairs = sorted(tree.query_pairs(reach * ss), key=lambda ab: np.linalg.norm(pts[ab[0]] - pts[ab[1]]))
        for a, c in pairs:
            fa, fc = free[a], free[c]
            if fa[0] == fc[0] or fa in link or fc in link:
                continue
            gl = np.linalg.norm(pts[c] - pts[a]) / ss                 # gap in work px
            if spacing is not None:
                # adaptive: a gap may be bridged up to ~0.7 x the local line spacing (dense hair:
                # short bridges only, so a strand never jumps to its neighbour; open areas: long)
                mid = (pts[a] + pts[c]) / 2 / ss
                sp = spacing[int(np.clip(mid[0], 0, spacing.shape[0] - 1)), int(np.clip(mid[1], 0, spacing.shape[1] - 1))]
                lim = float(np.clip(bridge_frac * sp, bridge_px, bridge_max))
                if gl > lim:
                    continue
                cmin = bridge_cos if gl <= bridge_px else bridge_cos_loose
                if gl > bridge_px:                                    # longer bridges need faint support
                    k = np.linspace(0, 1, 9)[1:-1, None]
                    q = (pts[a] + (pts[c] - pts[a]) * k) / ss
                    if ndi.map_coordinates(P, [q[:, 0], q[:, 1]], order=1).mean() < 0.12:
                        continue
            else:
                cmin = bridge_cos
            da, dc = -end_dir(*fa), -end_dir(*fc)          # outward directions
            gap = _unit(pts[c] - pts[a])
            if np.dot(da, gap) > cmin and np.dot(dc, -gap) > cmin and np.dot(da, dc) < -cmin:
                link[fa] = (*fc, 0); link[fc] = (*fa, 0)
    # 4. walk chains
    seen = np.zeros(nb_, bool)
    strokes = []
    order = sorted(range(nb_), key=lambda b: (b, 0))
    starts = [b for b in order if keep[b] and ((b, 0) not in link or (b, 1) not in link)] + \
             [b for b in order if keep[b]]
    for b0 in starts:
        if seen[b0]:
            continue
        # begin at an end with no link (or anywhere for loops)
        e_in = 0 if (b0, 0) not in link else (1 if (b0, 1) not in link else 0)
        pts, b, e_in_cur = [], b0, e_in
        start_free = branches[b0]["j"][e_in] == 0
        while True:
            seen[b] = True
            p = branches[b]["p"] if e_in_cur == 0 else branches[b]["p"][::-1]
            pts.append(p)
            e_out = 1 - e_in_cur
            nxt = link.get((b, e_out))
            if nxt is None or seen[nxt[0]]:
                end_free = branches[b]["j"][e_out] == 0 and nxt is None
                break
            b2, e2, j = nxt
            if j > 0:
                pts.append(jcent[j - 1][None])
            b, e_in_cur = b2, e2
        # ends attached to an unpaired junction: extend into the junction centre (strokes meet)
        first_b = b0; first_e = e_in
        if not start_free and branches[first_b]["j"][first_e] > 0:
            pts.insert(0, jcent[branches[first_b]["j"][first_e] - 1][None])
        last_j = branches[b]["j"][1 - e_in_cur]
        if not end_free and last_j > 0 and link.get((b, 1 - e_in_cur)) is None:
            pts.append(jcent[last_j - 1][None])
        P_ = np.vstack(pts)
        keepi = np.r_[True, np.any(np.diff(P_, axis=0) != 0, axis=1)]
        P_ = P_[keepi]
        mw = 2 * dist[np.clip(P_[:, 0].round().astype(int), 0, dist.shape[0] - 1),
                      np.clip(P_[:, 1].round().astype(int), 0, dist.shape[1] - 1)] / ss
        xy = (P_[:, ::-1] + 0.5) / ss
        L = np.hypot(*np.diff(xy, axis=0).T).sum() if len(xy) > 1 else 0
        if L < min_len and start_free and end_free:
            continue
        strokes.append(dict(xy=xy, mw=np.maximum(mw, 0.5), free=[bool(start_free), bool(end_free)], L=L))
    return strokes


def _resample(xy, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if d[-1] < step:
        return xy, d
    s = np.linspace(0, d[-1], max(2, int(round(d[-1] / step)) + 1))
    return np.c_[np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])], s


def shape_stroke(st, cov, wmin=0.8, wmax_cap=6.0, step=0.5, smooth_px=2.0, wsmooth_px=3.0,
                 taper_px=6.0, probe=4.0):
    """Smoothed centreline + width (from ink across the line) -> outline polygon (n,2)."""
    xy0 = st["xy"]
    if len(xy0) < 2:
        return None, None
    d0 = np.r_[0, np.cumsum(np.hypot(*np.diff(xy0, axis=0).T))]
    xy, s = _resample(xy0, step)
    if len(xy) < 3:
        return None, None
    mw = np.interp(s, d0, st["mw"])
    sm = ndi.gaussian_filter1d(xy, smooth_px / step, axis=0, mode="nearest")
    sm[0], sm[-1] = (sm[0] + xy[0]) / 2, (sm[-1] + xy[-1]) / 2
    xy = sm
    t = np.gradient(xy, axis=0); t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    n = np.c_[-t[:, 1], t[:, 0]]
    # ink across the line: sum of coverage along the normal (px)
    offs = np.arange(-probe, probe + 1e-6, 0.5)
    sx = xy[:, 0][:, None] + n[:, 0][:, None] * offs[None] - 0.5
    sy = xy[:, 1][:, None] + n[:, 1][:, None] * offs[None] - 0.5
    c = ndi.map_coordinates(cov, [sy.ravel(), sx.ravel()], order=1, mode="nearest").reshape(sx.shape)
    ink_w = c.sum(1) * 0.5
    w = np.clip(ink_w, wmin, np.clip(mw, wmin, wmax_cap))
    w = ndi.gaussian_filter1d(w, wsmooth_px / step, mode="nearest")
    L = s[-1]
    tp = min(taper_px, L / 3)
    if tp > 0:
        if st["free"][0]:
            w = w * np.clip(0.2 + 0.8 * s / tp, 0, 1)
        if st["free"][1]:
            w = w * np.clip(0.2 + 0.8 * (L - s) / tp, 0, 1)
    left = xy + n * (w / 2)[:, None]; right = xy - n * (w / 2)[:, None]
    caps = []
    for c_, tt, ww, fr in ((xy[-1], t[-1], w[-1], st["free"][1]), (xy[0], -t[0], w[0], st["free"][0])):
        if fr:
            caps.append(np.array([c_ + tt * ww * 0.6]))
        else:                                  # round cap: overlaps the strokes it meets
            nn = np.array([-tt[1], tt[0]])
            a = np.linspace(0, np.pi, 7)[1:-1]
            caps.append(c_ + (np.cos(a)[:, None] * nn[None] + np.sin(a)[:, None] * tt[None]) * ww / 2)
    poly = np.vstack([left, caps[0], right[::-1], caps[1]])
    return poly, xy


def stroke_layer(geo, M, A, wmin=0.8, tol=0.35):
    """Returns (svg path strings, cover mask (work px) of the drawn strokes and blobs)."""
    from .contour import field_paths
    P = line_prob(M, A)
    cov = ink_field(geo, P > 0.5)
    blob = blob_mask(P, cov)
    strokes = build_strokes(P, blob=blob)
    H, W = P.shape
    cover = blob.astype(np.uint8).copy()
    out = []
    lum = geo.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    # blobs: filled outlines of the ink inside them, one colour each (darker half)
    n, cc = cv2.connectedComponents(blob.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        m = cc == i
        l = lum[m]; col = geo[m][l <= np.median(l)].mean(0)
        tag = field_paths(np.where(m, np.maximum(cov, 0.6 * P), 0).astype(np.float32), m, col, ss=2, tol=0.4, min_area=4)
        if tag:
            out.append(tag)
    for st in strokes:
        poly, cl = shape_stroke(st, cov, wmin=wmin)
        if poly is None:
            continue
        cv2.fillPoly(cover, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        segs = fit_closed(np.vstack([poly, poly[:1]]), tol=tol, corner_deg=75)
        if not segs:
            continue
        xi = np.clip(cl[:, 0].astype(int), 0, W - 1); yi = np.clip(cl[:, 1].astype(int), 0, H - 1)
        px = geo[yi, xi].astype(float); l = lum[yi, xi]
        col = px[l <= np.median(l)].mean(0) if len(px) else px.mean(0)
        d = ["M%.2f,%.2f" % tuple(segs[0][0])]
        for sg in segs:
            d.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*sg[1], *sg[2], *sg[3]))
        out.append('<path d="%sZ" fill="#%02x%02x%02x"/>' % ("".join(d), *(int(v) for v in col)))
    return out, cover > 0
