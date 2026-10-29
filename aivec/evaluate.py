"""Render SVGs with Chromium and score them against a de-grained (Gaussian) copy of the original.
Metrics: mean OKLab dE (x100), SSIM (gray), edge-F1 (Canny on reference vs render, 2px tolerance)."""
import sys, json, os, subprocess, tempfile, numpy as np, cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from .color import srgb_to_oklab

JS = """const {chromium}=require('playwright');(async()=>{const [svg,png,w,h]=process.argv.slice(2);
const b=await chromium.launch();const p=await b.newPage({viewport:{width:+w,height:+h}});
await p.goto('file://'+svg);await p.screenshot({path:png});await b.close();})();"""


def render(svg, png, w, h):
    js = os.path.join(tempfile.gettempdir(), "aivec_render.js")
    open(js, "w").write(JS)
    # make the svg fill the viewport exactly
    s = open(svg).read()
    import re
    s = re.sub(r'<svg\b([^>]*)>', lambda m: '<svg' + re.sub(r'\s(width|height)="[^"]*"', '', m.group(1))
               + f' width="{w}" height="{h}" preserveAspectRatio="none">', s, count=1)
    tmp = svg + ".fit.svg"; open(tmp, "w").write(s)
    subprocess.run(["node", js, os.path.abspath(tmp), os.path.abspath(png), str(w), str(h)], check=True)
    os.remove(tmp)
    return np.asarray(Image.open(png).convert("RGB"))


def score(ref, img):
    de = np.sqrt(((srgb_to_oklab(ref) - srgb_to_oklab(img)) ** 2).sum(-1)).mean() * 100
    g1, g2 = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ss = ssim(g1, g2, data_range=255)
    e1, e2 = cv2.Canny(g1, 40, 100) > 0, cv2.Canny(g2, 40, 100) > 0
    k = np.ones((5, 5), np.uint8)
    d1, d2 = cv2.dilate(e1.astype(np.uint8), k) > 0, cv2.dilate(e2.astype(np.uint8), k) > 0
    p = (e2 & d1).sum() / max(1, e2.sum()); r = (e1 & d2).sum() / max(1, e1.sum())
    return dict(dE=round(float(de), 2), ssim=round(float(ss), 4), edgeF1=round(float(2 * p * r / (p + r + 1e-9)), 4))


def main():
    orig = Image.open(sys.argv[1]).convert("RGB"); W, H = orig.size
    ref = cv2.GaussianBlur(np.asarray(orig), (0, 0), 2.0)
    res = {}
    for svg in sys.argv[2:]:
        img = render(svg, svg + ".png", W, H)
        r = score(ref, img); s = open(svg).read()
        r.update(paths=s.count("<path"), kb=round(len(s) / 1024))
        res[os.path.basename(svg)] = r
        print(os.path.basename(svg), json.dumps(r))
    return res


if __name__ == "__main__":
    main()
