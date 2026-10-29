"""AI pre-processing: Real-ESRGAN (realesr-animevideov3, SRVGGNetCompact) on CPU.

Trick: downscale by 1/k first (averages grain away), then let the anime SR model
re-synthesise crisp edges at 4x, then resample to the target working size.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.up = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.up(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def load_model(path):
    m = SRVGGNetCompact()
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd.get("params", sd))
    return m.eval()


@torch.no_grad()
def sr4x(model, rgb, tile=256, pad=12):
    x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * 4, w * 4)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            o = model(x[:, :, ya:yb, xa:xb])
            ty, tx = (y0 - ya) * 4, (x0 - xa) * 4
            th, tw = (min(h, y0 + tile) - y0) * 4, (min(w, x0 + tile) - x0) * 4
            out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = o[:, :, ty:ty + th, tx:tx + tw]
    return (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def ai_clean(img: Image.Image, weights, pre_down=2.0, out_scale=1.0):
    """Return a de-grained image at out_scale x the input size."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rgb = img.convert("RGB")
    W, H = rgb.size
    small = rgb.resize((round(W / pre_down), round(H / pre_down)), Image.BOX)
    big = sr4x(load_model(weights), np.asarray(small))
    return Image.fromarray(big).resize((round(W * out_scale), round(H * out_scale)), Image.LANCZOS)
