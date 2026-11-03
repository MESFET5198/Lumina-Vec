"""Line layer as STROKES (prototype): the line model's map -> centrelines + width profile ->
filled outlines with tapered free ends, fitted with cubic Beziers.

Why: tracing line pixels (or their iso-contour) reproduces every wobble and break of the raster.
An artist's line is one gesture: a smooth centreline whose width changes slowly and tapers
at the tips. So we recover exactly that: the path (skeleton of the model map, smoothed along
its length) and a width profile (smoothed), and rebuild the outline from them."""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .contour import fit_closed

_NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def _order(pix, ends):
    """Order the pixels (k x 2, row/col) of one skeleton branch into a path."""
    s = {tuple(p) for p in pix}
    start = tuple(ends[0]) if len(ends) else tuple(pix[0])
    path, cur, prev = [start], start, None
    s.discard(start)
    while True:
        nxt = None
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (cur[0] + dy, cur[1] + dx)
            if q in s:
                nxt = q
                break
        if nxt is None:
            break
        s.discard(nxt); path.append(nxt); cur = nxt
    return np.array(path, float)


def _gauss1d(v, sigma, pin_ends=False):
    if sigma <= 0 or len(v) < 3:
        return v
    out = ndi.gaussian_filter1d(v, sigma, axis=0, mode="nearest")
    if pin_ends:
        out[0], out[-1] = v[0], v[-1]
    return out


def _resample_w(p, w, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < step:
        return p, w, d[-1]
    s = np.linspace(0, d[-1], max(2, int(round(d[-1] / step)) + 1))
    return (np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])], np.interp(s, d, w), d[-1])


def extract_strokes(lm, ss=2, norm=0.5, thr=0.5, min_len=3.0):
    """Centreline strokes from the line model's map (1 = line). Returns list of dicts with
    'xy' (n x 2, x/y in input pixels), 'w' (full width per point), 'free' (start, end)."""
    f = np.clip(lm.astype(np.float32) / norm, 0, 1)
    up = cv2.resize(f, None, fx=ss, fy=ss, interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 0.5 * ss)
    mask = up > thr
    skel = skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, _NB, borderType=cv2.BORDER_CONSTANT) * skel
    junc = skel & (nb >= 3)
    jl, nj = ndi.label(junc, structure=np.ones((3, 3)))
    jcent = np.array(ndi.center_of_mass(junc, jl, range(1, nj + 1))) if nj else np.zeros((0, 2))
    br = skel & ~junc
    bl, nb_ = ndi.label(br, structure=np.ones((3, 3)))
    jl_d = cv2.dilate(jl.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int64)  # junction id near
    strokes = []
    objs = ndi.find_objects(bl)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = bl[sl] == i
        pix = np.argwhere(sub) + [sl[0].start, sl[1].start]
        if len(pix) == 0:
            continue
        # end pixels: one neighbour inside the branch
        cnt = cv2.filter2D(sub.astype(np.uint8), -1, _NB, borderType=cv2.BORDER_CONSTANT)[sub]
        ends = pix[cnt <= 1]
        path = _order(pix, ends)
        if len(path) < 2:
            continue
        free = [True, True]
        pts = [path]
        # attach the junction centre at each end that touches one, so strokes meet
        for k, e in ((0, path[0]), (1, path[-1])):
            j = jl_d[int(e[0]), int(e[1])]
            if j > 0:
                free[k] = False
                c = jcent[j - 1][None]
                pts = [c] + pts if k == 0 else pts + [c]
        p = np.vstack(pts)
        w = 2 * dist[np.clip(p[:, 0].round().astype(int), 0, dist.shape[0] - 1),
                     np.clip(p[:, 1].round().astype(int), 0, dist.shape[1] - 1)]
        xy = (p[:, ::-1] + 0.5) / ss
        w = np.maximum(w, 1.0) / ss
        L = np.hypot(*np.diff(xy, axis=0).T).sum()
        # skeleton spurs: short branches with a free end hanging off a junction
        if L < min_len or (L < 1.5 * np.median(w) + 1 and (free[0] != free[1])):
            continue
        strokes.append(dict(xy=xy, w=w, free=free, L=L))
    return strokes


def stroke_outline(st, step=0.5, smooth_px=2.0, wsmooth_px=4.0, taper_px=5.0, wscale=1.0, wmin=0.6):
    """Smooth centreline + smoothed width -> closed outline polygon (n x 2)."""
    xy, w, L = _resample_w(st["xy"], st["w"], step)
    if len(xy) < 3:
        return None
    xy = _gauss1d(xy, smooth_px / step, pin_ends=False)
    w = _gauss1d(w, wsmooth_px / step) * wscale
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    # taper free ends to a point (an artist's stroke lifts off); joined ends keep their width
    tp = min(taper_px, L / 3)
    if tp > 0:
        if st["free"][0]:
            w = w * np.clip(0.15 + 0.85 * s / tp, 0, 1)
        if st["free"][1]:
            w = w * np.clip(0.15 + 0.85 * (L - s) / tp, 0, 1)
    w = np.maximum(w, wmin * (w > 0))
    t = np.gradient(xy, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    n = np.c_[-t[:, 1], t[:, 0]]
    left = xy + n * (w / 2)[:, None]
    right = xy - n * (w / 2)[:, None]
    caps = []
    for k, (c, tt, ww) in enumerate(((xy[-1], t[-1], w[-1]), (xy[0], -t[0], w[0]))):
        if st["free"][1 - k]:          # k=0: end of the stroke, k=1: its start
            caps.append(np.array([c + tt * ww * 0.5]))           # pointed tip
        else:                                                   # round cap: overlaps the joint
            nn = np.array([-tt[1], tt[0]])                     # = +n at the end, -n at the start
            a = np.linspace(0, np.pi, 7)[1:-1]
            caps.append(c + (np.cos(a)[:, None] * nn[None] + np.sin(a)[:, None] * tt[None]) * ww / 2)
    return np.vstack([left, caps[0], right[::-1], caps[1]])


def stroke_color(st, rgb):
    H, W = rgb.shape[:2]
    xy = st["xy"]
    x = np.clip(xy[:, 0].astype(int), 0, W - 1); y = np.clip(xy[:, 1].astype(int), 0, H - 1)
    px = rgb[y, x].astype(float)
    lum = px @ [0.299, 0.587, 0.114]
    dark = px[lum <= np.median(lum)]
    c = dark.mean(0) if len(dark) else px.mean(0)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def stroke_paths(lm, rgb, tol=0.3, **kw):
    out = []
    for st in extract_strokes(lm):
        poly = stroke_outline(st, **kw)
        if poly is None:
            continue
        segs = fit_closed(np.vstack([poly, poly[:1]]), tol=tol, corner_deg=70)
        if not segs:
            continue
        d = ["M%.2f,%.2f" % tuple(segs[0][0])]
        for sg in segs:
            d.append("C%.2f,%.2f %.2f,%.2f %.2f,%.2f" % (*sg[1], *sg[2], *sg[3]))
        out.append(f'<path d="{"".join(d)}Z" fill="{stroke_color(st, rgb)}"/>')
    return out
