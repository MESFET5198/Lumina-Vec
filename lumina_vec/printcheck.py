"""Printability check for an SVG before slicing.

    python -m lumina_vec.printcheck out.svg --print-mm 263 [--line-mm 0.42]

Renders the SVG at its viewBox resolution (crisp edges) and reports, at the given print size:
colours, separate colour regions, regions smaller than one extrusion dot (each one becomes a tiny
loop + retraction in the slicer) and the share of area in features narrower than one line.
"""
import argparse, json, math, os, re, subprocess, tempfile
import numpy as np, cv2
from PIL import Image
from skimage.measure import label

JS = """const {chromium}=require('playwright');(async()=>{const [s,p,w,h]=process.argv.slice(2);
const b=await chromium.launch();const pg=await b.newPage({viewport:{width:+w,height:+h}});
await pg.goto('file://'+s);await pg.screenshot({path:p});await b.close();})();"""


def render_flat(svg_path, max_side=4000):
    svg = open(svg_path, encoding="utf8").read()
    vb = re.search(r'viewBox="([^"]+)"', svg)
    if vb:
        _, _, W, H = (float(v) for v in vb.group(1).replace(",", " ").split())
    else:
        W = float(re.search(r'\swidth="([\d.]+)', svg).group(1)); H = float(re.search(r'\sheight="([\d.]+)', svg).group(1))
    s = min(1.0, max_side / max(W, H)); w, h = round(W * s), round(H * s)
    svg = re.sub(r"<svg\b([^>]*)>", lambda m: "<svg" + re.sub(r'\s(width|height)="[^"]*"', "", m.group(1))
                 + f' width="{w}" height="{h}" shape-rendering="crispEdges">', svg, count=1)
    d = tempfile.mkdtemp(); sp, pp, js = (os.path.join(d, x) for x in ("a.svg", "a.png", "r.js"))
    open(sp, "w", encoding="utf8").write(svg); open(js, "w").write(JS)
    subprocess.run(["node", js, sp, pp, str(w), str(h)], check=True)
    return np.asarray(Image.open(pp).convert("RGB"))


def check(svg_path, print_mm, line_mm=0.42):
    a = render_flat(svg_path).astype(np.int64)
    key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    px_mm = max(key.shape) / print_mm
    line_px = line_mm * px_mm
    reg = label(key + 1, connectivity=1, background=0)
    sizes = np.bincount(reg.ravel())[1:]
    dot = line_px * line_px
    d = max(2, int(math.ceil(line_px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    keep = np.zeros(key.shape, bool)
    for c in np.unique(key):
        keep |= cv2.morphologyEx((key == c).astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    return dict(colours=int(len(np.unique(key))), regions=int(len(sizes)),
                regions_below_one_dot=int((sizes < dot).sum()),
                regions_below_1mm2=int((sizes < px_mm * px_mm).sum()),
                thin_area_pct=round(float((~keep).mean() * 100), 2),
                px_per_mm=round(px_mm, 2))


def main():
    ap = argparse.ArgumentParser(description="SVG printability check")
    ap.add_argument("svg", nargs="+"); ap.add_argument("--print-mm", type=float, required=True)
    ap.add_argument("--line-mm", type=float, default=0.42)
    a = ap.parse_args()
    for s in a.svg:
        print(os.path.basename(s), json.dumps(check(s, a.print_mm, a.line_mm), ensure_ascii=False))


if __name__ == "__main__":
    main()
