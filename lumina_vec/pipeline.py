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
import argparse, json, math, os, re, time
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
                           filter_speckle=4, layer_difference=16, simplify=1.0, ink=False, **_BANDS),
    # ^ clean painterly art: its own lines are already clean; redrawing them made eyes/lashes heavier
    # clean cel/flat-colour art (hard-edged fills): no gradient smoothing (it put halos around
    # highlights), cutout tracing (shared edges, no slivers), fewer colours
    "cel": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10, filter_speckle=4, layer_difference=16,
                simplify=1.0, ink=True, **{**_BANDS, "bands": 48, "blur_sigma": 0.0},
                detail_overlay=False, hierarchical_mode="cutout"),
    # fewer, larger shapes; loses tiny strokes (mouth lines etc.)
    "flat": dict(pre_down=2.0, work_scale=1.5, ms_sp=10, ms_sr=16,
                 filter_speckle=24, layer_difference=12, simplify=1.5, **_OFF),
    # keep more small detail, bigger file
    "detail": dict(pre_down=2.0, work_scale=1.0, ms_sp=6, ms_sr=10,
                   filter_speckle=8, layer_difference=16, simplify=1.0, **_OFF),
}


GRAIN_THRESHOLD = 4.0          # mean |img - median3(img)| on native-resolution grey
MAX_WORK_PX = {"shading": 14e6, "shading-detail": 6e6, "cel": 6e6}
FLAT_THRESHOLD = 0.18          # share of pixels in gentle gradients; below = flat cel art


def grain_level(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"))
    return float(np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16)).mean())


def add_gap_fillers(svg):
    """Display-only seam cover for shared-edge (cutout) output: before each fill path, a
    same-colour 2px non-scaling stroke in Lumina's exact `qq3d-gap-filler` layout, which
    Lumina strips on import (core/vector_engine.py fast filter), so it never reaches the print."""
    def rep(m):
        tag = m.group(0)
        d = re.search(r'\sd="([^"]*)"', tag); f = re.search(r'\sfill="(#[0-9a-fA-F]{6})"', tag)
        if not d or not f:
            return tag
        gf = (f'<path class="qq3d-gap-filler" d="{d.group(1)}" fill="none" stroke="{f.group(1)}" '
              'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>')
        tr = re.search(r'\stransform="([^"]*)"', tag)
        if tr:
            gf = f'<g transform="{tr.group(1)}">{gf}</g>'
        return gf + "\n" + tag
    return re.sub(r"<path\b[^>]*/>", rep, svg)


def gradient_share(img: Image.Image) -> float:
    """Share of pixels in gentle colour gradients (shading), measured at ~1 MP after a median
    filter (ignores grain). Flat cel art ~0.1, shaded illustrations 0.3-0.5."""
    W, H = img.size; s = min(1.0, (1e6 / (W * H)) ** 0.5)
    a = np.asarray(img.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BOX))
    lab = cv2.cvtColor(cv2.medianBlur(a, 5).astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    g = np.sqrt((cv2.Sobel(lab, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab, cv2.CV_32F, 0, 1) ** 2).sum(-1))
    return float(((g >= 2) & (g < 12)).mean())


def auto_params(img: Image.Image):
    """Pick preset from measured grain, and shrink the working size of very large images.
    Calibrated on 3 images: heavy-grain flat anime (8.8), painterly (1.9), clean cel (0.2)."""
    grain = grain_level(img)
    if grain >= GRAIN_THRESHOLD:
        name = "shading"
    else:
        name = "cel" if gradient_share(img) < FLAT_THRESHOLD else "shading-detail"
    params = dict(PRESETS[name])
    W, H = img.size
    cap = (MAX_WORK_PX[name] / (W * H)) ** 0.5
    if params["work_scale"] > cap:
        params["work_scale"] = round(max(0.25, cap), 2)
    return name, grain, params


FINE_MAX_PX = 6e6              # fine mode: working-size cap (memory / time)


def fine_params(img: Image.Image, params: dict) -> dict:
    """精细模式 (tested on 5 illustrations, Nov 2026): line layer v3 on; for images that are not
    grainy, no 2x downscale before the AI clean (small images lost hair strands and wallpaper
    lines there), a 2x working size (measured knee: 1x->2x completes lines, beyond 2x nothing
    but more nodes), and thin faint structures protected from gradient smoothing. Grainy
    images keep their preset's downscale and smoothing (grain would pass as 'thin structure')."""
    p = dict(params)
    p.update(ink=True, line_style="v3")
    if grain_level(img) < GRAIN_THRESHOLD:
        W, H = img.size
        p.update(pre_down=1.0, thin_thr=0.015,
                 work_scale=round(min(2.0, (FINE_MAX_PX / (W * H)) ** 0.5), 2))
    return p


def vectorize(img: Image.Image, weights=None, pre_down=2.0, work_scale=1.5, ms_sp=8, ms_sr=14,
              filter_speckle=12, layer_difference=10, simplify=1.5, color_precision=6, ai=True,
              clean=None,
              bands=0, vote_sigma=3.0, l_weight=1.3, c_weight=2.0, blur_sigma=12.0, edge_thr=0.4, detail_overlay=True, ink=False, progress=None,
              print_mm=0.0, line_mm=0.42, lines="anime", weights_dir=None, hierarchical_mode=None, contour_lines=True,
              tracer="contour", label_sigma=1.0, thin_thr=0.0, line_style="v3",
              palette_n=0, palette_de=2.5, region_color=False):
    W, H = img.size
    # stage cost model (seconds per megapixel, CPU), calibrated on two test images; the AI step
    # scales with its (downscaled) input, everything else with the working resolution
    mp_in = W * H / pre_down ** 2 / 1e6
    mp_w = W * H * work_scale ** 2 / 1e6
    stages = [("ai", "AI 去颗粒", 12.5 * mp_in if (ai and clean is None) else 0.2),
              ("flat", "压平", 1.0 * mp_w),
              ("ink", "线条", 4.0 * mp_w if (bands and ink) else 0),
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
    ink_cover = None
    bright = None
    # print mode: smallest feature one `line_mm` extrusion can lay down, in working pixels
    min_px = line_mm * max(w, h) / print_mm if print_mm else 0.0
    hierarchical = hierarchical_mode or "stacked"
    if min_px:
        filter_speckle = max(filter_speckle, int(math.ceil(min_px)))
        # stacked shapes are fitted independently and leave slivers where their curves disagree
        # (thousands of sub-extrusion islands); a cutout mosaic shares every boundary
        hierarchical = "cutout"
        detail_overlay = False      # printable details are already in the cleaned label map
    if bands and ink:
        from .ink import ink_mask, fill_under, trace_ink, trace_lines
        report("ink")
        wdir = weights_dir or (os.path.dirname(weights) if weights else "weights")
        v3 = (line_style in ("v3", "stroke") and lines and not min_px
              and all(os.path.exists(os.path.join(wdir, f)) for f in ("netG.pth", "erika.pth")))
        if v3:
            # line layer v3 (lumina_vec/linework.py): manga_line + lineart_anime say WHERE, the
            # original's ink says HOW WIDE (capped near the centreline: dark fills stay fills),
            # one outline per line network traced at 2x, each piece coloured from the original.
            # Print mode keeps the older path below (minimum stroke width / length handling).
            from .linework import line_field, trace_field, color_paths
            geo = clean if grain_level(img) >= GRAIN_THRESHOLD else \
                np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS))
            report("ink", 0.05, "AI 识别线条")
            if line_style == "stroke":
                # v4: strokes (lumina_vec/strokes2.py) - continuous, tapered, one colour each
                from .lines import line_map
                from .strokes2 import stroke_layer
                dl = min(max(h, w), 1536)
                M = line_map(clean, "manga", detect_long=dl, weights_dir=wdir)
                A = line_map(clean, "anime", detect_long=dl, weights_dir=wdir)
                report("ink", 0.6, "重建笔画")
                ink_paths, im = stroke_layer(geo, M, A)
            else:
                fld, _ = line_field(geo, clean, weights_dir=wdir)
                report("ink", 0.7, "描摹线条")
                ink_paths = color_paths(trace_field(fld), geo)
                # colour fill is repaired ONLY under the drawn lines: texture the line layer did
                # not take (knit, wallpaper lines) stays in the colour layer
                im = fld > 0.5
        elif lines and os.path.exists(os.path.join(wdir, "netG.pth")):
            # AI line model decides WHERE lines are; the original gives their width; strokes are
            # redrawn from centreline + smoothed width -> even, smooth lines
            from .lines import line_map, stroke_mask, redraw_strokes
            report("ink", 0.05, "AI 识别线条")
            lm = line_map(clean, "anime", detect_long=min(3072, max(clean.shape[:2])), weights_dir=wdir)
            report("ink", 0.6, "重绘线条")
            # print mode: a stroke shorter than a few extrusion widths prints as a blob
            im = stroke_mask(clean, lm, min_len=max(6, int(round(4 * min_px))) if min_px else 6,
                             rel_thr=0.06 if not contour_lines else 0.035)
            rd = redraw_strokes(im, min_radius=min_px / 2 if min_px else 0.0)
            # lines never paint over highlights: pixels clearly lighter than their surroundings
            from .color import srgb_to_oklab as _ok
            _L = _ok(clean)[..., 0].astype(np.float32)
            # highlights = SMALL blobs lighter than their surroundings (white top-hat). A median
            # comparison also flagged ordinary fill next to every dark line, leaving slivers
            _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            bright = cv2.morphologyEx(_L, cv2.MORPH_TOPHAT, _k) > 0.10
            bright = cv2.dilate(bright.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if blur_sigma <= 0:     # flat cel art: hard fills already keep highlights; protecting
                bright[:] = False   # its white outlines/rims only leaves slivers beside lines
            im &= ~bright
            rd &= ~cv2.resize(bright.astype(np.uint8), (rd.shape[1], rd.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            report("ink", 0.8, "描摹线条")
            if contour_lines:
                # geometry from CONTINUOUS values of the original (its anti-aliasing places the
                # edge sub-pixel); the AI-cleaned image only where the original is grainy
                from .ink import trace_lines_contour
                from .contour import ink_field
                geo = clean if grain_level(img) >= GRAIN_THRESHOLD else \
                    np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS))
                # WHERE: the line model's own map, gated low (0.15): its thin faint outlines peak
                # at ~0.5 and a mask built from local darkness missed them (outlines merge with
                # dark shading into blobs). HOW WIDE / HOW DARK: the original's ink coverage
                # (no highlight mask here: coverage comes from darkness, so highlights get none,
                # while the dilated top-hat mask also ate every line beside white hair)
                # Print mode keeps the conservative stroke mask: model-gated hairlines add
                # thousands of sub-extrusion islands
                gate = im if min_px else lm > 0.15
                fld = ink_field(geo, gate)
                fld[bright] = 0
                im = gate & (fld > 0.35)
                if min_px:      # print mode: never thinner than one extrusion line
                    rd_cov = cv2.resize(rd.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                    fld = np.maximum(fld, rd_cov)
                ink_paths = trace_lines_contour(geo, fld, im)
                im = im | (fld > 0.3)
            else:
                ink_paths = trace_lines(clean, im, rd)
                im = im | (cv2.resize(rd.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0)
        else:                       # no line model: colour rule (dark, neutral, thin)
            im = ink_mask(clean)
            ink_paths = trace_ink(clean, im)
        sm_fill = blur_sigma > 0
        rgb, clean = fill_under(rgb, im, bright, sm_fill), fill_under(clean, im, bright, sm_fill)
        ink_cover = im
    if bands:
        from .bands import posterize
        # contours come from the (pre-mean-shift) cleaned image: mean-shift plateaus are blocky
        rgb, palette = posterize(rgb, k=bands, vote_sigma=vote_sigma, l_weight=l_weight, c_weight=c_weight,
                                 edge_thr=edge_thr, blur_sigma=blur_sigma, smooth_src=clean,
                                 progress=lambda f, s: report("bands", f, s), min_px=min_px,
                                 cover=ink_cover, thin_thr=thin_thr)
        color_precision, layer_difference = 8, 1   # keep every band as its own layer
    # print mode: the cutout mosaic is still more printable (its stacked shapes leave sub-0.7 px
    # slivers where two fitted curves trace the same edge); contour tracer for display output
    if bands and tracer == "contour" and not min_px:
        # own tracer: mode-filtered label map -> stacked sub-pixel contours -> cubic Beziers.
        # Each visible boundary belongs to exactly one shape, so no seams/slivers (no cutout needed)
        from .contour import smooth_labels, drop_slivers, region_paths
        report("trace", 0.0, "平滑色块边界")
        if label_sigma > 0:
            rgb = smooth_labels(rgb, sigma=label_sigma)
        rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
        extra = []
        if detail_overlay:
            from .overlay import trace_details
            extra = trace_details(rgb, min_px=min_px)
            detail_overlay = False          # added below together with the lines
        if region_color:
            # colour pipeline: shapes from the bands, colour per region from a 50 % copy of the
            # (line-free) cleaned image, similar neighbours merged, one global palette for all
            from .palette import region_pipeline
            report("trace", 0.05, "区域取色")
            n_det = len(extra)
            rgb, recol = region_pipeline(rgb, clean, extra + list(ink_paths), n=palette_n or 128,
                                         barrier=ink_cover if ink_cover is not None and ink_cover.shape == rgb.shape[:2] else None)
            rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
            extra, ink_paths = recol[:n_det], recol[n_det:]
        elif palette_n:
            # one global palette for every layer; same-colour neighbours then merge into one
            # region in region_paths (connected components of equal colour)
            from .palette import unify
            report("trace", 0.05, "统一调色板")
            n_det = len(extra)
            rgb, recol, _ = unify(rgb, extra + list(ink_paths), n=palette_n, min_de=palette_de)
            rgb = drop_slivers(rgb, keep_dark=0.08 if not ink else 0.0)
            extra, ink_paths = recol[:n_det], recol[n_det:]
        items = region_paths(rgb, min_area=max(6, int(min_px * min_px)) if min_px else 6,
                             progress=lambda f: report("trace", 0.1 + 0.9 * f, f"{f * 100:.0f}%"))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n'
               + "\n".join([p for _, p in items] + extra) + "\n</svg>")
        hierarchical = "stacked"
    else:
        report("trace", 0.0, "这一步无法细分，请稍候")
        cfg = vtracer.Config(filter_speckle=filter_speckle, layer_difference=layer_difference,
                             color_precision=color_precision, simplify=simplify, palette=palette,
                             hierarchical=hierarchical)
        svg = cfg.convert_pixels(np.dstack([rgb, np.full((h, w), 255, np.uint8)]).tobytes(), w, h)
    report("detail")
    if hierarchical == "cutout":
        svg = add_gap_fillers(svg)       # colour layer only; lines/details go on top afterwards
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
    ap.add_argument("--print-mm", type=float, default=0.0,
                    help="print mode: printed length of this image's long side in mm (crop first if you only "
                         "print part of it). Removes colour features narrower than one extrusion line")
    ap.add_argument("--line-mm", type=float, default=0.42, help="extrusion line width in mm (default 0.42)")
    ap.add_argument("--lines", choices=["anime", "none"], default=None,
                    help="AI line model for the line layer (default anime if weights/netG.pth exists)")
    ap.add_argument("--tracer", choices=["contour", "vtracer"], default="contour",
                    help="colour-layer tracer for band presets (default: own sub-pixel contour tracer)")
    ap.add_argument("--fine", action="store_true",
                    help="精细模式: line layer v3 + no AI downscale + 2x work size + thin-line protection")
    ap.add_argument("--line-style", choices=["stroke", "v3", "contour"], default=None,
                    help="line layer: v3 (manga_line + lineart_anime, default) or contour (older)")
    ap.add_argument("--thin-thr", type=float, default=None,
                    help="protect faint thin structures from gradient smoothing (0 = off, fine mode 0.015)")
    ap.add_argument("--palette", type=int, default=None,
                    help="experimental: one global palette of N colours for all layers (0 = off)")
    ap.add_argument("--region-color", type=int, choices=[0, 1], default=None,
                    help="experimental colour pipeline: per-region colour from a 50%% image, merge, palette")
    ap.add_argument("--ink", type=int, choices=[0, 1], default=None,
                    help="separate smoothed outline layer (default: on for the grainy `shading` preset)")
    a = ap.parse_args()
    src = Image.open(a.input)
    if a.preset == "auto":
        chosen, grain, params = auto_params(src)
        print(json.dumps(dict(auto_preset=chosen, grain=round(grain, 2), work_scale=params["work_scale"])))
    else:
        params = dict(PRESETS[a.preset])
    if a.fine:
        params = fine_params(src, params)
    if a.line_style:
        params["line_style"] = a.line_style
    if a.region_color is not None:
        params["region_color"] = bool(a.region_color)
    if a.palette is not None:
        params["palette_n"] = a.palette
    if a.thin_thr is not None:
        params["thin_thr"] = a.thin_thr
    if a.ink is not None:
        params["ink"] = bool(a.ink)
    if a.lines:
        params["lines"] = None if a.lines == "none" else a.lines
    if a.print_mm:
        params["print_mm"], params["line_mm"] = a.print_mm, a.line_mm
        if a.bands is None and params.get("bands"):
            params["bands"] = 56            # fewer colours -> fewer filament recipes -> fewer splits per layer
    for k in params:
        if k in ("ink", "print_mm", "line_mm", "lines", "line_style", "thin_thr", "palette_n", "region_color"):
            continue
        v = getattr(a, k, None)
        if v is not None:
            params[k] = v
    t = time.time()
    svg, work = vectorize(src, a.weights, ai=not a.no_ai, tracer=a.tracer,
                          clean=Image.open(a.clean) if a.clean else None, **params)
    open(a.output, "w").write(svg)
    if a.save_clean:
        work.save(a.save_clean)
    print(json.dumps(dict(paths=svg.count("<path"), kb=round(len(svg) / 1024),
                          secs=round(time.time() - t, 1), **params)))


if __name__ == "__main__":
    main()
