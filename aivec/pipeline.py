"""AI-assisted raster -> SVG vectorizer.

Pipeline (see README for why each step exists):
  1. AI de-grain:  box-downscale by `pre_down` (averages film grain away) -> Real-ESRGAN
                   anime model x4 (re-synthesises crisp line art) -> resample to `work_scale`.
  2. Flatten:      mean-shift filtering collapses residual painterly texture into flat colour.
  3. Trace:        vtracer 1.0 hierarchical (stacked) clustering + spline fitting.
  4. Output:       SVG at the input's pixel size (viewBox keeps the working resolution).
"""
import argparse, json, re, time
import numpy as np, cv2, vtracer
from PIL import Image

PRESETS = {
    # tuned on anime / illustration with heavy grain
    "illustration": dict(pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
                         filter_speckle=12, layer_difference=10, simplify=1.5),
    # fewer, larger shapes; loses tiny strokes (mouth lines etc.)
    "flat": dict(pre_down=2.0, work_scale=1.5, ms_sp=10, ms_sr=16,
                 filter_speckle=24, layer_difference=12, simplify=1.5),
    # keep more small detail, bigger file
    "detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                   filter_speckle=8, layer_difference=16, simplify=1.0),
}


def vectorize(img: Image.Image, weights=None, pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
              filter_speckle=12, layer_difference=10, simplify=1.5, color_precision=6, ai=True):
    W, H = img.size
    if ai:
        from .denoise import ai_clean
        work = ai_clean(img, weights, pre_down=pre_down, out_scale=work_scale)
    else:
        work = img.convert("RGB").resize((round(W * work_scale), round(H * work_scale)), Image.LANCZOS)
    rgb = np.asarray(work.convert("RGB"))
    if ms_sp > 0:
        rgb = cv2.pyrMeanShiftFiltering(rgb, ms_sp, ms_sr)
    h, w = rgb.shape[:2]
    cfg = vtracer.Config(filter_speckle=filter_speckle, layer_difference=layer_difference,
                         color_precision=color_precision, simplify=simplify)
    svg = cfg.convert_pixels(np.dstack([rgb, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    svg = re.sub(r"<svg\b([^>]*)>", lambda m: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m.group(1))
                 + f' width="{W}" height="{H}" viewBox="0 0 {w} {h}">', svg, count=1)
    return svg, work


def main():
    ap = argparse.ArgumentParser(description="AI-assisted raster -> SVG vectorizer")
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--preset", choices=PRESETS, default="illustration")
    ap.add_argument("--weights", default="weights/realesr-animevideov3.pth")
    ap.add_argument("--no-ai", action="store_true", help="skip the Real-ESRGAN clean step")
    ap.add_argument("--save-clean", default=None, help="also save the AI-cleaned raster here")
    for k, v in PRESETS["illustration"].items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    a = ap.parse_args()
    params = dict(PRESETS[a.preset])
    for k in params:
        v = getattr(a, k)
        if v is not None:
            params[k] = v
    t = time.time()
    svg, work = vectorize(Image.open(a.input), a.weights, ai=not a.no_ai, **params)
    open(a.output, "w").write(svg)
    if a.save_clean:
        work.save(a.save_clean)
    print(json.dumps(dict(paths=svg.count("<path"), kb=round(len(svg) / 1024),
                          secs=round(time.time() - t, 1), **params)))


if __name__ == "__main__":
    main()
