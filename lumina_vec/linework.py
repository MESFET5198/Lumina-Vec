"""Line layer v3 (prototype): WHERE from two line models, HOW WIDE from the original's ink,
SHAPE traced Vecto-style (one outline per connected line network, junctions come out clean).

- manga_line (MangaLineExtraction) gives continuous artist ink lines; lineart_anime adds the
  thin ones it misses.
- width: the original's ink coverage inside the model's line, capped at `rmax` px from the
  model's centreline: wide dark areas (pupils, dark fills) are fills, not lines, and stay in the
  colour layer. Tips get no extra width, so the original's tapering survives.
- traced at 2x (the measured knee: 1x -> 2x +14 points of completeness, beyond 2x nothing) by
  the Vecto CLI if available, else our own iso-contour tracer.
- each line piece takes the darker pixels of the original under it (eyelashes stay red-brown)."""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .contour import ink_field


def line_field(geo, clean, weights_dir="weights", rmax=1.6, manga_norm=0.6, anime_norm=0.5,
               maps=None, detect_max=1536):
    from .lines import line_map
    h, w = clean.shape[:2]
    if maps is None:
        # models run at <= detect_max px (memory: manga_line at 2.5k px needs > 7 GB; and the
        # models are not better above ~1.2k, see docs/NOTES.md), output resized to the work size
        dl = min(max(h, w), detect_max)
        M = line_map(clean, "manga", detect_long=dl, weights_dir=weights_dir)
        A = line_map(clean, "anime", detect_long=dl, weights_dir=weights_dir)
    else:
        M, A = maps
    # rmax is in work pixels; scale it with how much the work image is larger than the model's view
    rmax = rmax * max(1.0, max(h, w) / 1216) if maps is None else rmax
    P = np.maximum(np.clip(M / manga_norm, 0, 1), np.clip(A / anime_norm, 0, 1)).astype(np.float32)
    mask = P > 0.5
    # centreline of the model's lines, and how far each pixel is from it
    skel = skeletonize(mask)
    d = ndi.distance_transform_edt(~skel)
    allow = np.clip(rmax + 0.5 - d, 0, 1).astype(np.float32)
    cov = ink_field(geo, mask)          # 0..1 coverage of the original's dark ink
    # a line pixel needs the model AND real ink; on the centreline keep at least the model's
    # verdict so faint (light-grey) outlines are not lost
    F = np.minimum(P, np.maximum(cov, 0.55 * skel)) * allow
    return F, dict(M=M, A=A, P=P)


def _vecto():
    exe = os.environ.get("VECTO_CLI") or shutil.which("vecto") or \
        "/home/claude/ext/Vecto/Vecto.Cli/bin/Release/net8.0/vecto"
    return exe if os.path.exists(exe) else None


def trace_field(F, ss=2, steep=3.0):
    """Outline paths (SVG <path> strings, coordinates in input px) of F > 0.5."""
    up = cv2.resize(F, None, fx=ss, fy=ss, interpolation=cv2.INTER_CUBIC)
    up = np.clip((up - 0.5) * steep + 0.5, 0, 1)
    exe = _vecto()
    if exe is None:
        from .contour import field_paths
        return [field_paths(F, F > 0.2, "#000000")]
    with tempfile.TemporaryDirectory() as td:
        png = os.path.join(td, "l.png"); svg = os.path.join(td, "l.svg")
        cv2.imwrite(png, (255 * (1 - up)).astype(np.uint8))
        env = dict(os.environ); env.setdefault("DOTNET_ROOT", "/home/claude/dotnet")
        subprocess.run([exe, "trace", png, "-o", svg, "--colors", "2", "--detail", "high"],
                       check=True, capture_output=True, env=env)
        s = open(svg).read()
    paths = re.findall(r"<path[^>]*/>", s, flags=re.S)
    out = []
    for p in paths:
        m = re.search(r'fill="#([0-9a-fA-F]{6})"', p)
        if not m or int(m.group(1)[:2], 16) > 128:      # keep the dark (line) faces only
            continue
        d = re.search(r'\sd="([^"]*)"', p, flags=re.S).group(1)
        # scale back to input pixels
        d = re.sub(r"-?\d+\.?\d*", lambda t: "%.2f" % (float(t.group(0)) / ss), d)
        out.append(d)
    return out


def color_paths(ds, rgb, dark_pct=35):
    """Fill colour per path: mean of the darker pixels of `rgb` under it."""
    H, W = rgb.shape[:2]
    from .svgraster import rasterize_ids
    lab = rasterize_ids(ds, W, H, ss=2)
    big = cv2.resize(rgb, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC).astype(float)
    lum = big @ [0.299, 0.587, 0.114]
    idx = lab.ravel(); ok = idx > 0
    order = np.argsort(idx[ok], kind="stable")
    ids = idx[ok][order]; px = big.reshape(-1, 3)[ok][order]; lu = lum.ravel()[ok][order]
    starts = np.searchsorted(ids, np.arange(1, len(ds) + 1))
    ends = np.searchsorted(ids, np.arange(1, len(ds) + 1), side="right")
    out = []
    for i, d in enumerate(ds):
        a, b = starts[i], ends[i]
        if b <= a:
            continue
        l = lu[a:b]; c = px[a:b][l <= np.percentile(l, dark_pct)].mean(0)
        out.append(f'<path d="{d}" fill="#%02x%02x%02x"/>' % tuple(int(v) for v in c))
    return out
