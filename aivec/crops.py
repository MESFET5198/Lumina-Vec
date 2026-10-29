"""Side-by-side crops: original | each rendered SVG."""
import sys, numpy as np
from PIL import Image, ImageDraw
REGIONS = {"face": (700, 650, 1300, 1150), "eye": (1000, 830, 1250, 1000), "leg": (300, 2350, 1100, 2950),
           "stamp": (1420, 750, 1560, 1100), "hand": (1100, 1950, 1600, 2300), "lens": (600, 520, 1000, 780)}
def main(out_prefix, orig, *pngs, labels=None):
    ims = [Image.open(orig).convert("RGB")] + [Image.open(p).convert("RGB") for p in pngs]
    names = ["original"] + [p.split("/")[-1].replace(".svg.png", "") for p in pngs]
    for k, box in REGIONS.items():
        cs = [im.crop(box) for im in ims]
        w, h = cs[0].size; sc = max(1, 600 // w) if w < 300 else 1
        cs = [c.resize((w * sc, h * sc), Image.NEAREST) for c in cs]; w, h = cs[0].size
        o = Image.new("RGB", (len(cs) * (w + 8), h + 26), "white"); d = ImageDraw.Draw(o)
        for i, (c, n) in enumerate(zip(cs, names)):
            o.paste(c, (i * (w + 8), 26)); d.text((i * (w + 8) + 4, 6), n, fill="black")
        o.save(f"{out_prefix}_{k}.png")
if __name__ == "__main__":
    main(*sys.argv[1:])
