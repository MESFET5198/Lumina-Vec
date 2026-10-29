"""AI-assisted raster -> SVG vectorizer.

Pipeline (see README for why each step exists):
  1. AI de-grain:  box-downscale by `pre_down` (averages film grain away) -> Real-ESRGAN
                   anime model x4 (re-synthesises crisp line art) -> resample to `work_scale`.
  2. Flatten:      mean-shift filtering collapses residual painterly texture into flat colour.
  2b. Bands (optional, --bands N): OKLab k-means palette; low-contrast pixels are labelled
                   from an edge-masked, distance-adaptive blur + spatial vote, so gradients become
                   smooth contour-like steps; lines/texture keep per-pixel labels (lumina_vec/bands.py).
  3. Trace:        vtracer 1.0 hierarchical (stacked) clustering + spline fitting.
  4. Output:       SVG at the input's pixel size (viewBox keeps the working resolution).
"""
import argparse, json, re, time
import numpy as np, cv2, vtracer
from PIL import Image

_BANDS = dict(bands=112, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4)
_OFF = dict(bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4)

PRESETS = {
    # tuned on anime / illustration with heavy grain
    "illustration": dict(pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
                         filter_speckle=12, layer_difference=10, simplify=1.5, **_OFF),
    # illustration + explicit shading bands: smooth light/dark transitions (more colours/paths)
    "shading": dict(pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
                    filter_speckle=4, layer_difference=10, simplify=1.5, **_BANDS),
    # painterly / detailed art (light grain, fine texture) + shading bands
    "shading-detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                           filter_speckle=4, layer_difference=16, simplify=1.0, **_BANDS),
    # fewer, larger shapes; loses tiny strokes (mouth lines etc.)
    "flat": dict(pre_down=2.0, work_scale=1.5, ms_sp=10, ms_sr=16,
                 filter_speckle=24, layer_difference=12, simplify=1.5, **_OFF),
    # keep more small detail, bigger file
    "detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                   filter_speckle=8, layer_difference=16, simplify=1.0, **_OFF),
}


GRAIN_THRESHOLD = 4.0          # mean |img - median3(img)| on native-resolution grey
MAX_WORK_PX = {"shading": 14e6, "shading-detail": 6e6}


def grain_level(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"))
    return float(np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16)).mean())


def auto_params(img: Image.Image):
    """Pick preset from measured grain, and shrink the working size of very large images.
    Calibrated on 3 images: heavy-grain flat anime (8.8), painterly (1.9), clean cel (0.2)."""
    grain = grain_level(img)
    name = "shading" if grain >= GRAIN_THRESHOLD else "shading-detail"
    params = dict(PRESETS[name])
    W, H = img.size
    cap = (MAX_WORK_PX[name] / (W * H)) ** 0.5
    if params["work_scale"] > cap:
        params["work_scale"] = round(max(0.25, cap), 2)
    return name, grain, params


def vectorize(img: Image.Image, weights=None, pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
              filter_speckle=12, layer_difference=10, simplify=1.5, color_precision=6, ai=True,
              bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4, detail_overlay=True):
    W, H = img.size
    if ai:
        from .denoise import ai_clean
        work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale)
    else:
        work = img.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    rgb = np.asarray(work.convert("RGB"))
    clean = rgb
    if ms_sp > 0:
        rgb = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr)
    h, w = rgb.shape[:2]
    palette = None
    if bands:
        from .bands import posterize
        # contours come from the (pre-mean-shift) cleaned image: mean-shift plateaus are blocky
        rgb, palette = posterize(rgb, k=bands, vote_sigma=vote_sigma, l_weight=l_weight, c_weight=c_weight,
                                 edge_thr=edge_thr, blur_sigma=blur_sigma, smooth_src=clean)
        color_precision, layer_difference = 8, 1   # keep every band as its own layer
    cfg = vtracer.Config(filter_speckle=filter_speckle, layer_difference=layer_difference,
                         color_precision=color_precision, simplify=simplify, palette=palette)
    svg = cfg.convert_pixels(np.dstack([rgb, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    if bands and detail_overlay:
        from .overlay import trace_details
        extra = trace_details(rgb)
        svg = svg.replace("</svg>", "\n".join(extra) + "\n</svg>")
    svg = re.sub(r"<svg\b([^>]*)>", lambda m: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m.group(1))
                 + f' width="{W}" height="{H}" viewBox="0 0 {w} {h}">', svg, count=1)
    return svg, work


def main():
    ap = argparse.ArgumentParser(description="AI-assisted raster -> SVG vectorizer")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--preset", choices=["auto", *PRESETS], default="auto")
    ap.add_argument("--weights", default="weights/realesr-animevideov3.pth")
    ap.add_argument("--no-ai", action="store_true", help="skip the Real-ESRGAN clean step")
    ap.add_argument("--save-clean", default=None, help="also save the AI-cleaned raster here")
    for k, v in PRESETS["illustration"].items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    a = ap.parse_args()
    src = Image.open(a.input)
    if a.preset == "auto":
        chosen, grain, params = auto_params(src)
        print(json.dumps(dict(auto_preset=chosen, grain=round(grain, 2), work_scale=params["work_scale"])))
    else:
        params = dict(PRESETS[a.preset])
    for k in params:
        v = getattr(a, k)
        if v is not None:
            params[k] = v
    t = time.time()
    svg, work = vectorize(src, a.weights, ai=not a.no_ai, **params)
    open(a.output, "w").write(svg)
    if a.save_clean:
        work.save(a.save_clean)
    print(json.dumps(dict(paths=svg.count("<path"), kb=round(len(svg) / 1024),
                          secs=round(time.time() - t, 1), **params)))


if __name__ == "__main__":
    main()
