"""Line pipeline v5 (prototype): DRAW the line-model map with Bezier strokes, one at a time.

Like an artist inking over a sketch: find the strongest line not yet drawn, put the pen on it,
follow it in one movement (direction changes limited, so it goes straight through crossings,
short faint gaps are bridged), fit the path with as few cubic Beziers as possible, measure its
width, then erase what the stroke covers from the sketch and repeat.

Eyes: iris / pupil are ellipses, highlights white dots placed LAST, on top of the lines - the way
they are painted - so a highlight no longer cuts the eye's outline."""
import numpy as np
import cv2
from scipy import ndimage as ndi

from .contour import fit_cubic, ink_field, _fit_one, _reparam, _max_err, _chord_params, _bezier_d1, _bezier_d2

DIRS = np.stack([np.cos(np.linspace(0, np.pi, 36, endpoint=False)),
                 np.sin(np.linspace(0, np.pi, 36, endpoint=False))], 1)


def _sample(R, pts):
    return ndi.map_coordinates(R, [pts[..., 1], pts[..., 0]], order=1, mode="constant", cval=0.0)


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _follow(R, P, p, d, thr=0.22, step=1.0, max_turn=np.radians(22), gap=8, overlap=10, max_len=3000):
    """Walk along a line from p in direction d. Steering uses the full sketch P (so a crossing
    already drawn by another stroke does not stop the pen), continuation needs P >= thr
    (lower than the start threshold), gaps up to `gap` px are bridged, and the walk stops once
    it has run `overlap` px along a line that is already drawn (residual R empty)."""
    out = []
    angs = np.radians(np.arange(-22, 23, 5.5))
    misses = 0; drawn = 0
    pend = []
    for _ in range(max_len):
        best, bd = -1, d
        for a in angs:
            nd = _rot(d, a)
            look = p + np.outer(np.arange(1, 4), nd) * step
            v = (0.5 * _sample(P, look) + 0.5 * _sample(R, look)).mean() * (1 - 0.15 * abs(a) / max_turn)
            if v > best:
                best, bd = v, nd
        q = p + bd * step
        nrm = np.array([-bd[1], bd[0]])
        offs = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        vals = _sample(P, q + np.outer(offs, nrm))
        if vals.max() > 0:
            q = q + nrm * offs[int(vals.argmax())] * 0.6
        vp = _sample(P, q[None])[0]; vr = _sample(R, q[None])[0]
        if vp >= thr:
            out.extend(pend); pend = []; misses = 0
            out.append(q)
            drawn = drawn + 1 if vr < 0.1 else 0
            if drawn > overlap:
                out = out[:-overlap]
                break
        else:
            misses += 1
            pend.append(q)
            if misses > gap:
                break
        d = _unit(0.7 * d + 0.3 * (q - p)) if np.linalg.norm(q - p) > 0 else d
        p = q
    return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _no_inflection(ctrl, rel=0.05):
    """True if the cubic bends one way only (C shape, no S)."""
    t = np.linspace(0.02, 0.98, 25)
    d1, d2 = _bezier_d1(ctrl, t), _bezier_d2(ctrl, t)
    k = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    big = np.abs(k) > rel * np.abs(k).max() if np.abs(k).max() > 0 else np.zeros_like(k, bool)
    return not (np.any(k[big] > 0) and np.any(k[big] < 0))


def _minor_bulge(pts):
    """Size (px) of the smaller of the two bulges either side of the chord: ~0 for a straight
    line or a C, large only for a real S."""
    a, b = pts[0], pts[-1]; ch = b - a; cl = np.hypot(*ch)
    if cl < 1e-6:
        return 0.0
    sd = ((pts[:, 0] - a[0]) * ch[1] - (pts[:, 1] - a[1]) * ch[0]) / cl
    return float(min(sd.max(), -sd.min(), ) if sd.max() > 0 and sd.min() < 0 else 0.0)


def _fit_free(pts, u):
    """One cubic, ends fixed, both handles free (least squares)."""
    b0, b1, b2, b3 = _bern(u)
    M = np.hstack([b1, b2])
    sol = np.linalg.lstsq(M, pts - b0 * pts[0] - b3 * pts[-1], rcond=None)[0]
    return np.array([pts[0], sol[0], sol[1], pts[-1]])


def simple_pieces(pts, tol_min=1.5, tol_rel=0.06, tol_max=4.0, min_len=4.0, depth=0, s_len=50.0, s_px=1.0):
    """Split a hand path into pieces that are each ONE straight line or ONE C-shaped cubic
    (no S-bends, no wiggles). A piece that does not fit is cut at its worst point, and both
    halves become separate strokes. Returns list of (ctrl 4x2, i0, i1) index ranges into pts."""
    n = len(pts)
    L = np.hypot(*np.diff(pts, axis=0).T).sum()
    tol = float(np.clip(tol_rel * L, tol_min, tol_max))
    a, b = pts[0], pts[-1]
    ch = b - a; cl = np.hypot(*ch)
    if cl > 1e-6:
        dline = np.abs((pts[:, 0] - a[0]) * ch[1] - (pts[:, 1] - a[1]) * ch[0]) / cl
    else:
        dline = np.hypot(*(pts - a).T)
    if dline.max() <= 0.8 * tol or n < 4 or L < min_len * 1.5 or depth > 8:
        return [(np.array([a, a + ch / 3, b - ch / 3, b]), 0, n - 1)]
    u = _chord_params(pts)
    ctrl = _fit_free(pts, u)
    for _ in range(4):
        u = _reparam(pts, ctrl, u); ctrl = _fit_free(pts, u)
    err, at = _max_err(pts, ctrl, u)
    # one cubic: straight / C always; an S only if it is tiny (< s_px, i.e. jitter) or the stroke
    # is long (hair strands are S-shaped); short strokes never visibly wiggle
    if err <= tol and (L >= s_len or _minor_bulge(pts) < s_px):
        return [(ctrl, 0, n - 1)]
    # cut at the worst point (or at the line's farthest point if the curve had an S)
    if err > tol:
        cut = at
    else:                                   # an S: cut where the path crosses its chord
        ch = pts[-1] - pts[0]; cl = max(np.hypot(*ch), 1e-6)
        sd = ((pts[:, 0] - pts[0][0]) * ch[1] - (pts[:, 1] - pts[0][1]) * ch[0]) / cl
        zc = np.flatnonzero(np.diff(np.sign(sd[1:-1]))) + 1
        cut = int(zc[np.argmin(np.abs(zc - n // 2))]) if len(zc) else int(np.abs(sd).argmax())
    cut = int(np.clip(cut, 2, n - 3))
    left = simple_pieces(pts[:cut + 1], tol_min, tol_rel, tol_max, min_len, depth + 1, s_len, s_px)
    right = simple_pieces(pts[cut:], tol_min, tol_rel, tol_max, min_len, depth + 1, s_len, s_px)
    return left + [(c, i0 + cut, i1 + cut) for c, i0, i1 in right]


def _turn_corners(pts, ang_deg=35.0, k=4):
    """Indices where the path really turns (angle between the directions k samples before and
    after > ang_deg), local maxima only."""
    n = len(pts)
    if n < 2 * k + 3:
        return []
    a = pts[k:-k] - pts[:-2 * k]; b = pts[2 * k:] - pts[k:-k]
    cs = (a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-9)
    ang = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    idx = []
    for i in np.flatnonzero(ang > ang_deg):
        if ang[i] >= ang[max(i - 1, 0)] and ang[i] >= ang[min(i + 1, len(ang) - 1)]:
            j = i + k
            if not idx or j - idx[-1] > 2 * k:
                idx.append(j)
    return idx


def _resample1(pts, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < 2 * step:
        return pts
    s = np.arange(0, d[-1] + 1e-6, step)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def adaptive_pieces(pts, spacing_px, s_len=40.0):
    """Cut only at real turns; then each run is one cubic (C, or a gentle S when long) fitted with
    a tolerance that follows the local line spacing: loose in open areas, tight where lines
    are dense (there a loose curve would wander onto the neighbouring strand)."""
    cuts = [0] + _turn_corners(pts, ang_deg=40.0, k=5) + [len(pts) - 1]   # pts at 1 px steps
    out = []
    for i0, i1 in zip(cuts[:-1], cuts[1:]):
        if i1 - i0 < 1:
            continue
        run = pts[i0:i1 + 1]
        L = np.hypot(*np.diff(run, axis=0).T).sum()
        tmax = float(np.clip(0.3 * spacing_px, 1.2, 4.0))
        for c, a, b in simple_pieces(run, tol_min=1.2, tol_rel=0.06, tol_max=tmax, s_len=s_len):
            out.append((c, a + i0, b + i0))
    return out


def _width_at(Pm, pts, tang, maxw=8.0):
    """Width of the sketch line across each point: length of the >0.5 run along the normal."""
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    offs = np.arange(-maxw / 2, maxw / 2 + 1e-6, 0.25)
    sp = pts[:, None, :] + nrm[:, None, :] * offs[None, :, None]
    v = _sample(Pm, sp) > 0.5
    c = len(offs) // 2
    w = np.zeros(len(pts))
    for i in range(len(pts)):
        if not v[i, c]:
            w[i] = 0.0; continue
        l = c
        while l > 0 and v[i, l - 1]:
            l -= 1
        r = c
        while r < len(offs) - 1 and v[i, r + 1]:
            r += 1
        w[i] = (r - l + 1) * 0.25
    return w


def bez_point(seg, t):
    t = np.asarray(t)[:, None]
    return ((1 - t) ** 3) * seg[0] + 3 * ((1 - t) ** 2) * t * seg[1] + 3 * (1 - t) * t * t * seg[2] + t ** 3 * seg[3]


def _join_paths(paths, dist=8.0, cos_min=0.8):
    """Join paths whose ends nearly meet and point at each other (one line drawn in pieces)."""
    from scipy.spatial import cKDTree
    def odir(pts, e):                      # outward direction at end e
        k = min(6, len(pts) - 1)
        return _unit(pts[0] - pts[k]) if e == 0 else _unit(pts[-1] - pts[-1 - k])
    changed = True
    while changed and len(paths) > 1:
        changed = False
        ends = [(i, e) for i in range(len(paths)) for e in (0, 1)]
        P = np.array([paths[i][0 if e == 0 else -1] for i, e in ends])
        pairs = sorted(cKDTree(P).query_pairs(dist), key=lambda ab: np.linalg.norm(P[ab[0]] - P[ab[1]]))
        used = set()
        for a, b in pairs:
            (i, ei), (j, ej) = ends[a], ends[b]
            if i == j or i in used or j in used:
                continue
            di, dj = odir(paths[i], ei), odir(paths[j], ej)
            gap = P[b] - P[a]; gl = np.linalg.norm(gap)
            if np.dot(di, dj) > -cos_min:
                continue
            if gl > 1e-6 and (np.dot(di, gap / gl) < cos_min * 0.9 or np.dot(dj, -gap / gl) < cos_min * 0.9):
                continue
            pi = paths[i] if ei == 1 else paths[i][::-1]
            pj = paths[j] if ej == 0 else paths[j][::-1]
            paths[i] = np.vstack([pi, pj]); paths[j] = None
            used |= {i, j}; changed = True
        paths = [p for p in paths if p is not None]
    return paths


def draw_strokes(P, cov, thr=0.5, min_len=3.0, max_strokes=8000, wscale=1.0, wmin=0.8, width_from=None):
    """P: line probability (0..1), cov: original's ink coverage. Returns list of strokes:
    dict(segs=[4x2 cubic], w=(n,) widths, pts=(n,2) samples, free=[start,end])."""
    R = P.astype(np.float32).copy()
    H, W = R.shape
    Rs = cv2.GaussianBlur(R, (0, 0), 1.0)
    paths = []
    for _ in range(max_strokes):
        y, x = np.unravel_index(np.argmax(Rs), Rs.shape)
        if Rs[y, x] < thr:
            break
        p = np.array([float(x), float(y)])
        t = np.arange(-6, 7)
        scores = [_sample(R, p + np.outer(t, d)).sum() for d in DIRS]
        d = DIRS[int(np.argmax(scores))]
        fwd = _follow(R, P, p, d); bwd = _follow(R, P, p, -d)
        pts = np.array(bwd[::-1] + [p] + fwd)
        L = np.hypot(*np.diff(pts, axis=0).T).sum() if len(pts) > 1 else 0
        if L < min_len:
            cv2.circle(R, (int(x), int(y)), 2, 0, -1); cv2.circle(Rs, (int(x), int(y)), 3, 0, -1)
            continue
        paths.append(pts)
        tang = np.gradient(pts, axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
        w = np.maximum(_width_at(P, pts, tang), 1.0)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(outline(pts, w + 1.5, tang, [False, False]) * 16).astype(np.int32)], 1,
                     lineType=cv2.LINE_8, shift=4)
        R[m > 0] = 0
        Rs = np.where(cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0, cv2.GaussianBlur(R, (0, 0), 1.0), Rs)
    paths = _join_paths(paths)
    strokes = []
    for pts in paths:
        pts = ndi.gaussian_filter1d(pts, 3.0, axis=0, mode="nearest")
        pieces = simple_pieces(pts)
        for pi, (ctrl, i0, i1) in enumerate(pieces):
            seglen = np.hypot(*np.diff(pts[i0:i1 + 1], axis=0).T).sum() if i1 > i0 else 0
            sm = bez_point(ctrl, np.linspace(0, 1, max(4, int(seglen * 2))))
            tang = np.gradient(sm, axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
            w = _width_at(P, sm, tang)
            if width_from is not None:
                ink = _width_at(width_from, sm, tang)
                w = np.where(ink > 0, np.minimum(np.where(w > 0, w, ink), np.maximum(ink, wmin)), w)
            w = ndi.gaussian_filter1d(np.maximum(w, wmin) * wscale, 3, mode="nearest")
            strokes.append(dict(segs=[ctrl], pts=sm, w=w, free=[pi == 0, pi == len(pieces) - 1], path=len(strokes)))
    return strokes


def outline(pts, w, tang, taper=(True, True), taper_px=6.0):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    L = s[-1]; w = w.copy()
    tp = min(taper_px, L / 3)
    if tp > 0:
        if taper[0]:
            w *= np.clip(0.2 + 0.8 * s / tp, 0, 1)
        if taper[1]:
            w *= np.clip(0.2 + 0.8 * (L - s) / tp, 0, 1)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left = pts + nrm * (w / 2)[:, None]; right = pts - nrm * (w / 2)[:, None]
    def cap(c, tt, ww, free):
        if free:
            return (c + tt * ww * 0.6)[None]
        nn = np.array([-tt[1], tt[0]]); a = np.linspace(0, np.pi, 7)[1:-1]
        return c + (np.cos(a)[:, None] * nn[None] + np.sin(a)[:, None] * tt[None]) * ww / 2
    return np.vstack([left, cap(pts[-1], tang[-1], w[-1], taper[1]), right[::-1],
                      cap(pts[0], -tang[0], w[0], taper[0])])


# ------------------------------------------------------------------------- eyes

def find_eyes(rgb, line_mask, min_r=6, max_r=80):
    """Irises: saturated, roughly round blobs (highlight holes filled) that sit next to line art.
    Returns list of dict(iris=ellipse, pupil=ellipse|None, highlights=[ellipse], colors)."""
    from .color import srgb_to_oklab
    lab = srgb_to_oklab(rgb).astype(np.float32)
    C = np.hypot(lab[..., 1], lab[..., 2]); L = lab[..., 0]
    sat = (C > 0.08) & (L > 0.25) & (L < 0.9)
    sat = cv2.morphologyEx(sat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near_line = cv2.dilate(line_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    n, cc, st, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    eyes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < np.pi * min_r ** 2 * 0.5 or max(w, h) > 2 * max_r or w < 2 * min_r or h < 2 * min_r:
            continue
        m = cc == i
        m = ndi.binary_fill_holes(m)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=len)
        if len(c) < 10:
            continue
        el = cv2.fitEllipse(c)
        (cx, cy), (ew, eh), ang = el
        ell_area = np.pi * ew * eh / 4
        fill = m.sum() / max(ell_area, 1)
        if not (0.75 < fill < 1.25) or max(ew, eh) / max(min(ew, eh), 1) > 2.2:
            continue
        if not near_line[m].mean() > 0.3:
            continue
        # an iris has a highlight (near-white dot) and a darker centre / pupil
        em0 = np.zeros(m.shape, np.uint8); cv2.ellipse(em0, el, 1, -1)
        g0 = cv2.dilate(em0, np.ones((5, 5), np.uint8)) > 0
        if not ((g0 & (L > 0.9) & (C < 0.06)).sum() >= 2):
            continue
        em = np.zeros(m.shape, np.uint8); cv2.ellipse(em, el, 1, -1); em = em > 0
        pix = rgb[em & ~(L > 0.9)]
        iris_col = np.median(pix, 0) if len(pix) else rgb[int(cy), int(cx)]
        # lower half of the iris is usually lighter (reflected light): a second, lower ellipse
        low = em & (np.arange(m.shape[0])[:, None] > cy + 0.15 * eh / 2)
        low_col = np.median(rgb[low & ~(L > 0.9)], 0) if (low & ~(L > 0.9)).sum() > 10 else None
        # pupil: darkest compact part inside
        inner = em & (L < np.percentile(L[em], 20))
        inner = cv2.morphologyEx(inner.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        pupil = None
        if inner.sum() > 8:
            pc, _ = cv2.findContours(inner.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pc = max(pc, key=cv2.contourArea)
            if len(pc) >= 5:
                pe = cv2.fitEllipse(pc)
                pupil = (pe, np.median(rgb[inner], 0))
        # highlights: small near-white blobs inside the (slightly grown) iris
        grow = cv2.dilate(em.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        hl = grow & (L > 0.9) & (C < 0.06)
        hn, hc, hs, _ = cv2.connectedComponentsWithStats(hl.astype(np.uint8), connectivity=8)
        hls = []
        for k in range(1, hn):
            if 2 <= hs[k, 4] <= 0.25 * ell_area:
                hx, hy, hw, hh, _ = hs[k]
                pk = np.argwhere(hc == k)[:, ::-1].astype(np.float32)
                if len(pk) >= 5:
                    he = cv2.fitEllipse(pk)
                else:
                    (ccx, ccy), r = cv2.minEnclosingCircle(pk)
                    he = ((ccx, ccy), (2 * r + 1, 2 * r + 1), 0)
                hls.append(he)
        eyes.append(dict(iris=el, iris_col=iris_col, low_col=low_col, pupil=pupil, highlights=hls, mask=em))
    return eyes


def ellipse_path(el, col, k=8):
    """Exact ellipse as 4 cubic Beziers."""
    (cx, cy), (w, h), ang = el
    a, b = w / 2, h / 2; th = np.radians(ang)
    kk = 0.5522847498
    pts = [(1, 0), (1, kk), (kk, 1), (0, 1), (-kk, 1), (-1, kk), (-1, 0), (-1, -kk), (-kk, -1), (0, -1), (kk, -1), (1, -kk)]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = [R @ np.array([px * a, py * b]) + [cx, cy] for px, py in pts]
    d = "M%.2f,%.2f" % tuple(P[0])
    for i in range(4):
        c1, c2, e = P[3 * i + 1], P[3 * i + 2], P[(3 * i + 3) % 12]
        d += "C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*c1, *c2, *e)
    return '<path d="%sZ" fill="#%02x%02x%02x"/>' % (d, *(int(v) for v in col))


def eye_paths(eyes):
    """(under-lines paths, over-lines paths): iris, lower iris, pupil under; highlights over."""
    under, over = [], []
    for e in eyes:
        under.append(ellipse_path(e["iris"], e["iris_col"]))
        if e["low_col"] is not None:
            (cx, cy), (w, h), ang = e["iris"]
            under.append(ellipse_path(((cx, cy + h * 0.18), (w * 0.8, h * 0.55), ang), e["low_col"]))
        if e["pupil"] is not None:
            under.append(ellipse_path(*e["pupil"]))
        for hl in e["highlights"]:
            over.append(ellipse_path(hl, (255, 255, 255)))
    return under, over


def stroke_svg(strokes, rgb, tol=0.4):
    from .contour import fit_closed
    H, W = rgb.shape[:2]
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    out = []
    for st in strokes:
        pts, w = st["pts"], st["w"]
        tang = np.gradient(pts, axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
        poly = outline(pts, w, tang, st.get("free", [True, True]))
        segs = fit_closed(np.vstack([poly, poly[:1]]), tol=tol, corner_deg=75)
        if not segs:
            continue
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1); yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        px = rgb[yi, xi].astype(float); l = lum[yi, xi]
        col = px[l <= np.median(l)].mean(0)
        d = "M%.2f,%.2f" % tuple(segs[0][0]) + "".join("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*s[1], *s[2], *s[3]) for s in segs)
        out.append('<path d="%sZ" fill="#%02x%02x%02x"/>' % (d, *(int(v) for v in col)))
    return out


# ------------------------------------------------- lens output: a line = TWO cubic Beziers

def _bern(t):
    t = t[:, None]; mt = 1 - t
    return mt ** 3, 3 * mt * mt * t, 3 * mt * t * t, t ** 3


def _edge_cubic(E, t, a, b):
    """One cubic with fixed ends a, b that best fits edge points E(t) (least squares)."""
    b0, b1, b2, b3 = _bern(t)
    rhs = E - b0 * a - b3 * b
    M = np.hstack([b1, b2])                        # (n,2) basis for P1, P2
    G = M.T @ M + 1e-6 * np.eye(2)
    sol = np.linalg.solve(G, M.T @ rhs)            # (2,2): rows P1, P2
    return np.array([a, sol[0], sol[1], b])


def lens_paths(strokes, rgb, wmin=0.8):
    """Each stroke -> closed path of exactly two cubics (left edge, right edge), sharing the tips
    at free ends; at joined ends the edges stay apart by the line width (short straight cap)."""
    H, W = rgb.shape[:2]
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    out = []
    for st in strokes:
        C = st["segs"][0]
        n = len(st["pts"]); t = np.linspace(0, 1, n)
        pts = bez_point(C, t)
        d1 = _bezier_d1(C, t); d1 /= np.maximum(np.hypot(*d1.T), 1e-9)[:, None]
        nrm = np.stack([-d1[:, 1], d1[:, 0]], 1)
        w = np.maximum(st["w"], wmin)
        f0, f1 = st.get("free", [True, True])
        hw0 = 0.0 if f0 else w[0] / 2
        hw1 = 0.0 if f1 else w[-1] / 2
        # target half-width: measured, pinned to the end widths (0 at a free tip)
        half = w / 2
        half = half.copy(); half[0], half[-1] = hw0, hw1
        Lp = _edge_cubic(pts + nrm * half[:, None], t, pts[0] + nrm[0] * hw0, pts[-1] + nrm[-1] * hw1)
        Rp = _edge_cubic(pts - nrm * half[:, None], t, pts[0] - nrm[0] * hw0, pts[-1] - nrm[-1] * hw1)
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1); yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        px = rgb[yi, xi].astype(float); l = lum[yi, xi]
        col = px[l <= np.median(l)].mean(0)
        f = lambda p: "%.2f,%.2f" % tuple(p)
        d = (f"M{f(Lp[0])}C{f(Lp[1])} {f(Lp[2])} {f(Lp[3])}"
             + ("" if f1 else f"L{f(Rp[3])}")
             + f"C{f(Rp[2])} {f(Rp[1])} {f(Rp[0])}Z")
        out.append('<path d="%s" fill="#%02x%02x%02x"/>' % (d, *(int(v) for v in col)))
    return out


def strokes_from_graph(P, cov=None, wmin=0.8, adaptive=True):
    """Line FINDING from the centreline graph (strokes2.build_strokes: complete, pieces paired
    through crossings, gaps bridged) + simple pieces (line / C, gentle S on long) per stroke,
    in the same format as draw_strokes (for lens_paths)."""
    from .strokes2 import build_strokes, line_spacing
    out = []
    S = line_spacing(P) if adaptive else None
    for st in build_strokes(P, spacing=S):
        pts = st["xy"]
        if len(pts) < 3:
            continue
        pts = _resample1(pts)
        pts = ndi.gaussian_filter1d(pts, 1.5, axis=0, mode="nearest")
        if adaptive:
            yi = np.clip(pts[:, 1].astype(int), 0, S.shape[0] - 1); xi = np.clip(pts[:, 0].astype(int), 0, S.shape[1] - 1)
            pieces = adaptive_pieces(pts, float(np.median(S[yi, xi])))
        else:
            pieces = simple_pieces(pts)
        for pi, (ctrl, i0, i1) in enumerate(pieces):
            seglen = np.hypot(*np.diff(pts[i0:i1 + 1], axis=0).T).sum() if i1 > i0 else 0
            sm = bez_point(ctrl, np.linspace(0, 1, max(4, int(seglen * 2))))
            tang = np.gradient(sm, axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
            w = _width_at(P, sm, tang)
            if cov is not None:
                ink = _width_at(cov, sm, tang)
                w = np.where(ink > 0, np.minimum(np.where(w > 0, w, ink), np.maximum(ink, wmin)), w)
            w = ndi.gaussian_filter1d(np.maximum(w, wmin), 3, mode="nearest")
            free = [st["free"][0] if pi == 0 else False, st["free"][1] if pi == len(pieces) - 1 else False]
            out.append(dict(segs=[ctrl], pts=sm, w=w, free=free))
    return out


def lens_layer(geo, M, A):
    """Line layer v5 for the pipeline: centreline-graph strokes, each drawn as two cubic Beziers;
    thick dark masses (lash roots etc.) as filled shapes. Returns (svg paths, cover mask)."""
    from .strokes2 import line_prob, blob_mask
    from .contour import field_paths
    P = line_prob(M, A)
    cov = ink_field(geo, P > 0.5)
    blob = blob_mask(P, cov)
    Pd = np.where(blob, 0, P)
    st = strokes_from_graph(Pd, cov)
    H, W = P.shape
    cover = blob.astype(np.uint8).copy()
    for s in st:
        tang = np.gradient(s["pts"], axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
        poly = outline(s["pts"], s["w"], tang, s.get("free", [True, True]))
        cv2.fillPoly(cover, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    out = []
    lum = geo.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    n, cc = cv2.connectedComponents(blob.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        m = cc == i
        l = lum[m]; col = geo[m][l <= np.median(l)].mean(0)
        tag = field_paths(np.where(m, np.maximum(cov, 0.6 * Pd), 0).astype(np.float32), m, col, ss=2, tol=0.4, min_area=4)
        if tag:
            out.append(tag)
    # dots: small dark marks the model sees as line art but too short to be a stroke (beauty
    # marks, pupils' dots, freckles) - drawn as small filled shapes so they are not lost
    mask = (P > 0.5) & (cov > 0.4) & ~(cover > 0)
    n, cc, stt, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w_, h_, a = stt[i]
        if a < 4 or a > 150 or max(w_, h_) > 3.5 * max(1, min(w_, h_)):
            continue
        m = cc == i
        l = lum[m]; col = geo[m][l <= np.median(l)].mean(0)
        tag = field_paths(np.where(m, np.maximum(cov, 0.6 * P), 0).astype(np.float32), m, col, ss=2, tol=0.3, min_area=2)
        if tag:
            out.append(tag); cover[m] = 1
    out += lens_paths(st, geo)
    return out, cover > 0
