"""Detail overlay: vtracer's hierarchical clustering merges tiny clusters into their
neighbours, so a 3px mouth line or a nose dot vanishes even when the posterised image has it.
We find small connected regions that contrast with their surroundings and trace each one
as an exact binary mask, appended on top of the base layer (overlay -> no seams)."""
import re, numpy as np, vtracer
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from .color import srgb_to_oklab


def find_details(post, max_area=400, min_area=6, min_de=0.10, bg_size=15, min_px=0):
    """Small blobs that contrast with the local background (median filter). Anti-aliased lines
    are split over many palette colours, so detect on contrast, then fill each blob with one
    colour (the median of its most contrasting half)."""
    import cv2
    lab = srgb_to_oklab(post)
    bg = srgb_to_oklab(cv2.medianBlur(post, bg_size))
    de = np.sqrt(((lab - bg) ** 2).sum(-1))
    mask = de > min_de
    reg = cc_label(mask, connectivity=2)
    sizes = np.bincount(reg.ravel())
    H, W = reg.shape
    out = []
    for r, sl in enumerate(ndi.find_objects(reg), start=1):
        if sl is None or not (min_area <= sizes[r] <= max_area):
            continue
        ys = slice(max(0, sl[0].start - 1), min(H, sl[0].stop + 1))
        xs = slice(max(0, sl[1].start - 1), min(W, sl[1].stop + 1))
        m = reg[ys, xs] == r
        d = de[ys, xs][m]
        px = post[ys, xs][m][d >= np.median(d)]
        if min_px:
            if m.sum() < 0.6 * min_px * min_px:     # smaller than one extrusion dot: unprintable
                continue
            from .lines import enforce_min_width
            m = np.pad(m, int(min_px)); ys = slice(ys.start - int(min_px), ys.stop + int(min_px))
            xs = slice(xs.start - int(min_px), xs.stop + int(min_px))
            m = enforce_min_width(m, min_px)
            if ys.start < 0 or xs.start < 0 or ys.stop > H or xs.stop > W:   # keep it simple at borders
                cy0, cx0 = max(0, -ys.start), max(0, -xs.start)
                ys = slice(max(0, ys.start), min(H, ys.stop)); xs = slice(max(0, xs.start), min(W, xs.stop))
                m = m[cy0:cy0 + (ys.stop - ys.start), cx0:cx0 + (xs.stop - xs.start)]
        out.append((ys, xs, m, np.median(px, 0)))
    return out


def trace_details(post, simplify=0.5, **kw):
    cfg = vtracer.Config(clustering="bw", filter_speckle=0, simplify=simplify, mode="spline")
    paths = []
    for ys, xs, m, col in find_details(post, **kw):
        sub = np.pad(np.where(m, 0, 255).astype(np.uint8), 2, constant_values=255)
        h, w = sub.shape
        svg = cfg.convert_pixels(np.dstack([sub, sub, sub, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
        hexc = "#%02x%02x%02x" % tuple(int(v) for v in col)
        x0, y0 = xs.start - 2, ys.start - 2
        for p in re.findall(r"<path\b[^>]*/>", svg):
            if re.search(r'fill="#(fff|ffffff)"', p, re.I):
                continue
            p = re.sub(r'\sfill="[^"]*"', f' fill="{hexc}"', p)
            paths.append(p.replace("<path", f'<path transform="translate({x0},{y0})"', 1)
                         if "transform=" not in p else p)
    return paths
