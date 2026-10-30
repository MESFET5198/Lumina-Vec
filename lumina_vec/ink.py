"""Ink layer: thin, dark, neutral strokes (outlines) traced on their own from a smoothed,
2x-supersampled mask, so line edges come out as clean curves of even width instead of
inheriting the ragged per-pixel edge left by de-graining. Large dark areas (black hair)
are excluded by a morphological opening and keep their shading in the colour layer."""
import re, numpy as np, cv2, vtracer
from scipy import ndimage as ndi
from .color import srgb_to_oklab


def ink_mask(rgb, l_thr=0.36, c_thr=0.07, max_width=12, min_area=40):
    lab = srgb_to_oklab(rgb)
    dark = ((lab[..., 0] < l_thr) & (np.hypot(lab[..., 1], lab[..., 2]) < c_thr)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width, max_width))
    blobs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)          # areas wider than max_width
    thin = dark & (1 - cv2.dilate(blobs, np.ones((3, 3), np.uint8)))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[cc]


def fill_under(rgb, mask):
    """Replace masked pixels by the nearest unmasked pixel, so colour regions run under the ink."""
    if not mask.any():
        return rgb
    grow = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    _, (iy, ix) = ndi.distance_transform_edt(grow, return_indices=True)
    return rgb[iy, ix]


def trace_ink(rgb, mask, sigma=1.0, simplify=1.0, ss=2):
    """Supersample the mask, blur, threshold at 0.5 -> sub-pixel smooth edges; trace; scale back."""
    if not mask.any():
        return []
    H, W = mask.shape
    m = cv2.resize(mask.astype(np.float32), (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), sigma * ss) > 0.5
    img = np.where(m, 0, 255).astype(np.uint8)
    h, w = img.shape
    svg = vtracer.Config(clustering="bw", filter_speckle=4 * ss, simplify=simplify * ss, mode="spline") \
        .convert_pixels(np.dstack([img, img, img, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    col = "#%02x%02x%02x" % tuple(int(v) for v in np.median(rgb[mask], 0))
    out = []
    for p in re.findall(r"<path\b[^>]*/>", svg):
        if re.search(r'fill="#(fff|ffffff)"', p, re.I):
            continue
        p = re.sub(r'\sfill="[^"]*"', f' fill="{col}"', p)
        tr = re.search(r'transform="translate\(([-\d.]+),\s*([-\d.]+)\)"', p)
        if tr:
            p = p.replace(tr.group(0), f'transform="scale({1/ss}) translate({tr.group(1)},{tr.group(2)})"')
        else:
            p = p.replace("<path", f'<path transform="scale({1/ss})"', 1)
        out.append(p)
    return out
