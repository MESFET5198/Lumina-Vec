"""Line geometry from CONTINUOUS values.

The old line path thresholded strokes into a 0/1 pixel mask and let vtracer trace it: a binary
mask can only have staircase edges, and vtracer follows them. Here the line is a continuous
"ink" field (how much of each pixel is covered by the stroke, from the original's darkness),
its edge is the 0.5 iso-contour found with sub-pixel precision (marching squares), and that
contour is fitted with as few cubic Béziers as the error tolerance allows (Schneider's
algorithm, corners kept sharp). Anti-aliased pixels of the original are what place the edge,
so smooth artwork lines stay smooth.
"""
import math
import numpy as np
import cv2
from skimage import measure


# ---------------------------------------------------------------- ink field

def ink_field(src_rgb, stroke, win=21, gate_px=2.0):
    """Continuous stroke coverage in [0, 1]: local darkness relative to the surroundings,
    normalised by the stroke core's darkness, kept only near detected strokes (`stroke`)."""
    from .color import srgb_to_oklab
    L = srgb_to_oklab(src_rgb)[..., 0].astype(np.float32)
    bg = cv2.medianBlur(np.clip(L * 255, 0, 255).astype(np.uint8), win).astype(np.float32) / 255
    dark = np.clip(bg - L, 0, None)
    # core darkness of each stroke: local max of `dark` over the stroke's width
    core = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cov = np.clip(dark / np.maximum(core, 0.03), 0, 1)
    gate = cv2.GaussianBlur(cv2.dilate(stroke.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32),
                            (0, 0), gate_px)
    gate = np.clip(gate * 2, 0, 1)
    return cov * gate


# ---------------------------------------------------------------- Bézier fitting (Schneider)

def _bezier(ctrl, t):
    t = t[:, None]; mt = 1 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def _bezier_d1(ctrl, t):
    t = t[:, None]; mt = 1 - t
    return 3 * (mt ** 2) * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * (t ** 2) * (ctrl[3] - ctrl[2])


def _bezier_d2(ctrl, t):
    t = t[:, None]; mt = 1 - t
    return 6 * mt * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def _chord_params(p):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(p))


def _unit(v):
    n = np.hypot(*v)
    return v / n if n > 1e-12 else v


def _fit_one(p, u, t1, t2):
    a1 = np.outer(3 * (1 - u) ** 2 * u, t1); a2 = np.outer(3 * (1 - u) * u ** 2, t2)
    c00 = (a1 * a1).sum(); c01 = (a1 * a2).sum(); c11 = (a2 * a2).sum()
    b0 = (1 - u) ** 3 + 3 * (1 - u) ** 2 * u; b1 = 3 * (1 - u) * u ** 2 + u ** 3
    tmp = p - np.outer(b0, p[0]) - np.outer(b1, p[-1])
    x0 = (a1 * tmp).sum(); x1 = (a2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = np.hypot(*(p[-1] - p[0]))
    if abs(det) > 1e-12:
        al1 = (x0 * c11 - x1 * c01) / det; al2 = (c00 * x1 - c01 * x0) / det
    else:
        al1 = al2 = seg / 3
    if al1 < 1e-6 * seg or al2 < 1e-6 * seg or al1 > 3 * seg or al2 > 3 * seg:
        al1 = al2 = seg / 3
    return np.array([p[0], p[0] + t1 * al1, p[-1] + t2 * al2, p[-1]])


def _max_err(p, ctrl, u):
    d = np.hypot(*(_bezier(ctrl, u) - p).T)
    i = int(d.argmax())
    return d[i], i


def _reparam(p, ctrl, u):
    q = _bezier(ctrl, u) - p; d1 = _bezier_d1(ctrl, u); d2 = _bezier_d2(ctrl, u)
    num = (q * d1).sum(1); den = (d1 * d1).sum(1) + (q * d2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(np.abs(den) > 1e-12, u - num / den, u)
    return np.clip(nu, 0, 1)


def fit_cubic(p, t1, t2, tol, depth=0):
    """Fit points p (n x 2) with cubic Béziers, max error tol. Returns list of 4x2 control arrays."""
    if len(p) == 2 or depth > 12:
        seg = np.hypot(*(p[-1] - p[0])) / 3
        return [np.array([p[0], p[0] + t1 * seg, p[-1] + t2 * seg, p[-1]])]
    u = _chord_params(p)
    ctrl = _fit_one(p, u, t1, t2)
    err, split = _max_err(p, ctrl, u)
    if err < tol:
        return [ctrl]
    if err < tol * 4:
        for _ in range(4):
            u = _reparam(p, ctrl, u)
            ctrl = _fit_one(p, u, t1, t2)
            err, split = _max_err(p, ctrl, u)
            if err < tol:
                return [ctrl]
    split = min(max(split, 1), len(p) - 2)
    tc = _unit(p[split - 1] - p[split + 1])
    return fit_cubic(p[:split + 1], t1, tc, tol, depth + 1) + fit_cubic(p[split:], -tc, t2, tol, depth + 1)


def _corners(p, angle_deg=55, k=3):
    """Indices of sharp corners along a closed polyline (turn angle over +-k points)."""
    n = len(p)
    if n < 2 * k + 3:
        return []
    a = p[(np.arange(n) - k) % n] - p; b = p[(np.arange(n) + k) % n] - p
    cos = (a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))      # 180 = straight
    sharp = ang < (180 - angle_deg)
    idx = []
    for i in np.flatnonzero(sharp):                         # local minima of the angle only
        if ang[i] <= ang[(i - 1) % n] and ang[i] <= ang[(i + 1) % n] and (not idx or i - idx[-1] > k):
            idx.append(int(i))
    return idx


def _resample(p, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < step * 3:
        return p
    s = np.arange(0, d[-1], step)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def fit_closed(poly, tol=0.6, corner_deg=55):
    """Closed polyline -> list of cubic segments (each 4x2)."""
    p = _resample(poly[:-1] if np.allclose(poly[0], poly[-1]) else poly, 1.0)
    if len(p) < 4:
        return []
    # light smoothing along the curve (marching-squares zig-zag), corners protected below
    pc = np.r_[p[-2:], p, p[:2]]
    sm = (pc[:-4] + 2 * pc[1:-3] + 3 * pc[2:-2] + 2 * pc[3:-1] + pc[4:]) / 9
    cs = _corners(p, corner_deg)
    if cs:
        sm[cs] = p[cs]
    p = sm
    n = len(p)
    if not cs:
        # no corner: start anywhere, use wrap-around tangents
        q = np.r_[p, p[:1]]
        t = _unit(p[1] - p[-1])
        return fit_cubic(q, t, -t, tol)
    segs = []
    for j, c0 in enumerate(cs):
        c1 = cs[(j + 1) % len(cs)]
        idx = np.arange(c0, c1 + (n if c1 <= c0 else 0) + 1) % n
        q = p[idx]
        if len(q) < 2:
            continue
        t1 = _unit(q[min(2, len(q) - 1)] - q[0]); t2 = _unit(q[max(-3, -len(q))] - q[-1])
        segs += fit_cubic(q, t1, t2, tol)
    return segs


# ---------------------------------------------------------------- field -> SVG paths

def field_paths(field, mask, color, ss=2, tol=0.6, min_area=4.0):
    """Iso-contour (0.5) of `field` restricted to `mask` region (bbox), fitted with Béziers.
    Returns one evenodd <path> string (coordinates in field pixels) or ''."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ""
    y0, y1 = max(0, ys.min() - 3), min(field.shape[0], ys.max() + 4)
    x0, x1 = max(0, xs.min() - 3), min(field.shape[1], xs.max() + 4)
    f = np.where(cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0,
                 field[y0:y1, x0:x1], 0).astype(np.float32)
    f = np.pad(f, 2)
    if ss > 1:
        f = cv2.resize(f, None, fx=ss, fy=ss, interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), 0.6 * ss)
    parts = []
    for c in measure.find_contours(f, 0.5):
        if len(c) < 6:
            continue
        # array index u in the ss-upsampled crop -> SVG coordinate (pixel i spans [i, i+1])
        xy = (c[:, ::-1] + 0.5) / ss + np.array([x0 - 2, y0 - 2])
        area = 0.5 * abs(np.dot(xy[:-1, 0], xy[1:, 1]) - np.dot(xy[1:, 0], xy[:-1, 1]))
        if area < min_area:
            continue
        segs = fit_closed(xy, tol=tol)
        if not segs:
            continue
        d = ["M%.2f,%.2f" % tuple(segs[0][0])]
        for s in segs:
            d.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*s[1], *s[2], *s[3]))
        parts.append("".join(d) + "Z")
    if not parts:
        return ""
    col = "#%02x%02x%02x" % tuple(int(v) for v in color)
    return f'<path d="{"".join(parts)}" fill="{col}" fill-rule="evenodd"/>'
