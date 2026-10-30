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
import argparse, json, os, re, time
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
                    filter_speckle=4, layer_difference=10, simplify=1.5, ink=True, **_BANDS),
    # painterly / detailed art (light grain, fine texture) + shading bands
    "shading-detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                           filter_speckle=4, layer_difference=16, simplify=1.0, ink=True, **_BANDS),
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
              clean=None,
              bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4, detail_overlay=True, ink=False, progress=None,
              lines="anime", print_mm=0.0, line_mm=0.42, weights_dir=None):
    W, H = img.size
    # stage cost model (seconds per megapixel, CPU), calibrated on two test images; the AI step
    # scales with its (downscaled) input, everything else with the working resolution
    mp_in = W * H / pre_down ** 2 / 1e6
    mp_w = W * H * work_scale ** 2 / 1e6
    stages = [("ai", "AI 去颗粒", 12.5 * mp_in if (ai and clean is None) else 0.2),
              ("flat", "压平", 1.0 * mp_w),
              ("ink", "线条识别", (1.0 + (2.2 if lines else 0)) * mp_w if (bands and ink) else 0),
              ("bands", "明暗分层", 6.4 * mp_w if bands else 0),
              ("trace", "矢量描摹", 2.5 * mp_w),
              ("detail", "细节补回", 0.4 * mp_w if (bands and detail_overlay) else 0.05)]
    total = sum(w for *_, w in stages)
    start, acc = {}, 0.0
    for key, _, wgt in stages:
        start[key] = acc / total; acc += wgt
    names = {k: n for k, n, _ in stages}; weight = {k: w / total for k, _, w in stages}

    def report(key, frac=0.0, sub=None):
        if progress:
            label = names[key] + (f" · {sub}" if sub else "")
            progress(min(1.0, start[key] + weight[key] * frac), label)

    report("ai")
    if clean is not None:   # AI step already done elsewhere (python -m lumina_vec.denoise)
        work = clean.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    elif ai:
        from .denoise import ai_clean
        work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale,
                        progress=lambda f: report("ai", f, f"{f * 100:.0f}%"))
    else:
        work = img.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    rgb = np.asarray(work.convert("RGB"))
    clean = rgb
    report("flat")
    if ms_sp > 0:
        rgb = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr)
    h, w = rgb.shape[:2]
    palette = None
    ink_paths = []
    # printability: smallest feature a `line_mm` extrusion can lay down, in working pixels
    min_px = line_mm * max(w, h) / print_mm if print_mm else 0.0
    if bands and ink:
        from .ink import ink_mask, fill_under, trace_ink, trace_ink_colored
        report("ink")
        wdir = weights_dir or (os.path.dirname(weights) if weights else "weights")
        if lines and os.path.exists(os.path.join(wdir, "netG.pth" if lines == "anime" else "sk_model.pth")):
            from .lines import line_map, ai_ink_mask, enforce_min_width
            report("ink", 0.1, "AI 识别线条")
            im = ai_ink_mask(clean, line_map(clean, lines, weights_dir=wdir),
                             max_width=max(12, int(3 * min_px)))
            if min_px:
                im = enforce_min_width(im, min_px)
            report("ink", 0.7, "描摹线条")
            ink_paths = trace_ink_colored(clean, im)
        else:                       # no line model available: colour rule (dark, neutral, thin)
            im = ink_mask(clean)
            ink_paths = trace_ink(clean, im)
        rgb, clean = fill_under(rgb, im), fill_under(clean, im)
    if bands:
        from .bands import posterize
        # contours come from the (pre-mean-shift) cleaned image: mean-shift plateaus are blocky
        rgb, palette = posterize(rgb, k=bands, vote_sigma=vote_sigma, l_weight=l_weight, c_weight=c_weight,
                                 edge_thr=edge_thr, blur_sigma=blur_sigma, smooth_src=clean,
                                 progress=lambda f, s: report("bands", f, s), min_px=min_px)
        color_precision, layer_difference = 8, 1   # keep every band as its own layer
    report("trace", 0.0, "这一步无法细分，请稍候")
    cfg = vtracer.Config(filter_speckle=filter_speckle, layer_difference=layer_difference,
                         color_precision=color_precision, simplify=simplify, palette=palette)
    svg = cfg.convert_pixels(np.dstack([rgb, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    report("detail")
    if bands and detail_overlay:
        from .overlay import trace_details
        extra = trace_details(rgb, min_px=min_px)
        svg = svg.replace("</svg>", "\n".join(extra) + "\n</svg>")
    if ink_paths:
        svg = svg.replace("</svg>", "\n".join(ink_paths) + "\n</svg>")
    report("detail", 1.0)
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
    ap.add_argument("--clean", default=None,
                    help="use an image already cleaned by `python -m lumina_vec.denoise` (skips the AI step)")
    for k, v in PRESETS["illustration"].items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    ap.add_argument("--lines", choices=["anime", "sketch", "none"], default=None,
                    help="AI line model for the outline layer (default anime; falls back to a colour rule)")
    ap.add_argument("--print-mm", type=float, default=0.0,
                    help="printed size of the long side in mm; enables printability (min feature = --line-mm)")
    ap.add_argument("--line-mm", type=float, default=0.42, help="extrusion line width in mm (default 0.42)")
    ap.add_argument("--ink", type=int, choices=[0, 1], default=None,
                    help="separate smoothed outline layer (default: on for the grainy `shading` preset)")
    a = ap.parse_args()
    src = Image.open(a.input)
    if a.preset == "auto":
        chosen, grain, params = auto_params(src)
        print(json.dumps(dict(auto_preset=chosen, grain=round(grain, 2), work_scale=params["work_scale"])))
    else:
        params = dict(PRESETS[a.preset])
    if a.ink is not None:
        params["ink"] = bool(a.ink)
    if a.lines:
        params["lines"] = None if a.lines == "none" else a.lines
    params["print_mm"], params["line_mm"] = a.print_mm, a.line_mm
    for k in params:
        if k in ("ink", "lines", "print_mm", "line_mm"):
            continue
        v = getattr(a, k)
        if v is not None:
            params[k] = v
    t = time.time()
    svg, work = vectorize(src, a.weights, ai=not a.no_ai,
                          clean=Image.open(a.clean) if a.clean else None, **params)
    open(a.output, "w").write(svg)
    if a.save_clean:
        work.save(a.save_clean)
    print(json.dumps(dict(paths=svg.count("<path"), kb=round(len(svg) / 1024),
                          secs=round(time.time() - t, 1), **params)))


if __name__ == "__main__":
    main()
