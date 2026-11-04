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


def _follow(R, p, d, thr=0.3, step=1.0, max_turn=np.radians(22), gap=5, max_len=2000):
    """Walk along the ridge of R from p in direction d. Returns list of points (excl. p)."""
    out = []
    angs = np.radians(np.arange(-22, 23, 5.5))
    misses = 0
    pend = []
    for _ in range(max_len):
        # look 3 px ahead in a fan of directions; prefer straight (a pen keeps its direction)
        best, bd = -1, d
        for a in angs:
            nd = _rot(d, a)
            look = p + np.outer(np.arange(1, 4), nd) * step
            v = _sample(R, look).mean() * (1 - 0.15 * abs(a) / max_turn)
            if v > best:
                best, bd = v, nd
        q = p + bd * step
        # re-centre on the ridge across the line (sub-pixel)
        nrm = np.array([-bd[1], bd[0]])
        offs = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        vals = _sample(R, q + np.outer(offs, nrm))
        if vals.max() > 0:
            q = q + nrm * offs[int(vals.argmax())] * 0.6
        if _sample(R, q[None])[0] >= thr:
            out.extend(pend); pend = []; misses = 0
            out.append(q)
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


def simple_pieces(pts, tol_min=1.3, tol_rel=0.05, tol_max=3.0, min_len=4.0, depth=0):
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
    k = max(2, n // 6)
    t1 = _unit(pts[k] - pts[0]); t2 = _unit(pts[-1 - k] - pts[-1])
    u = _chord_params(pts)
    ctrl = _fit_one(pts, u, t1, t2)
    for _ in range(6):
        u = _reparam(pts, ctrl, u); ctrl = _fit_one(pts, u, t1, t2)
    err, at = _max_err(pts, ctrl, u)
    if err <= tol and _no_inflection(ctrl):
        return [(ctrl, 0, n - 1)]
    # cut at the worst point (or at the line's farthest point if the curve had an S)
    cut = at if err > tol else int(dline.argmax())
    cut = int(np.clip(cut, 2, n - 3))
    left = simple_pieces(pts[:cut + 1], tol_min, tol_rel, tol_max, min_len, depth + 1)
    right = simple_pieces(pts[cut:], tol_min, tol_rel, tol_max, min_len, depth + 1)
    return left + [(c, i0 + cut, i1 + cut) for c, i0, i1 in right]


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


def draw_strokes(P, cov, thr=0.5, min_len=3.0, fit_tol=0.9, max_strokes=6000, wscale=1.0, wmin=0.8,
                 width_from=None):
    """P: line probability (0..1), cov: original's ink coverage. Returns list of strokes:
    dict(segs=[4x2 cubic], w=(n,) widths along samples, pts=(n,2) samples, free=[True,True])."""
    R = P.astype(np.float32).copy()
    H, W = R.shape
    Rs = cv2.GaussianBlur(R, (0, 0), 1.0)
    strokes = []
    for _ in range(max_strokes):
        y, x = np.unravel_index(np.argmax(Rs), Rs.shape)
        if Rs[y, x] < thr:
            break
        p = np.array([x + 0.5, y + 0.5], float) - 0.5 + 0.5
        p = np.array([float(x), float(y)])
        # initial direction: the one with most sketch along +-6 px
        t = np.arange(-6, 7)
        scores = [_sample(R, p + np.outer(t, d)).sum() for d in DIRS]
        d = DIRS[int(np.argmax(scores))]
        fwd = _follow(R, p, d)
        bwd = _follow(R, p, -d)
        pts = np.array(bwd[::-1] + [p] + fwd)
        L = np.hypot(*np.diff(pts, axis=0).T).sum() if len(pts) > 1 else 0
        if L < min_len:
            # a dot / speck: erase it and move on
            cv2.circle(R, (int(x), int(y)), 2, 0, -1); cv2.circle(Rs, (int(x), int(y)), 3, 0, -1)
            continue
        # smooth the hand path a little, then: only straight lines or single C-curves
        pts = ndi.gaussian_filter1d(pts, 3.0, axis=0, mode="nearest")
        pieces = simple_pieces(pts)
        allpolys = []
        for pi, (ctrl, i0, i1) in enumerate(pieces):
            seglen = np.hypot(*np.diff(pts[i0:i1 + 1], axis=0).T).sum() if i1 > i0 else 0
            sm = bez_point(ctrl, np.linspace(0, 1, max(4, int(seglen * 2))))
            tang = np.gradient(sm, axis=0); tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
            w = _width_at(P, sm, tang)
            if width_from is not None:
                ink = _width_at(width_from, sm, tang)
                w = np.where(ink > 0, np.minimum(w, np.maximum(ink, wmin)), w)
            w = ndi.gaussian_filter1d(np.maximum(w, wmin) * wscale, 3, mode="nearest")
            free = [pi == 0, pi == len(pieces) - 1]          # cut points join: no taper
            strokes.append(dict(segs=[ctrl], pts=sm, w=w, free=free))
            allpolys.append(outline(sm, w + 1.5, tang, [False, False]))
        # erase what these strokes cover (a little wider: neighbouring pixels of the same line)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(pp * 16).astype(np.int32) for pp in allpolys], 1, lineType=cv2.LINE_8, shift=4)
        R[m > 0] = 0
        Rs = np.where(cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0, cv2.GaussianBlur(R, (0, 0), 1.0), Rs)
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
