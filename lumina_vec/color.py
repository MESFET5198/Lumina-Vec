"""Colour helpers."""
import numpy as np


def srgb_to_oklab(rgb):
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
    lms = np.cbrt(c @ M1.T)
    return lms @ M2.T


def flat_sample(rgb_flat, n, power=0.4, seed=0):
    """Sample pixels with probability ~ count(colour bin)^(power-1), so rare but distinct
    hues (an iris, a highlight) are represented instead of drowned by large flat areas."""
    rng = np.random.default_rng(seed)
    q = rgb_flat.astype(np.int32) >> 2
    key = (q[:, 0] << 12) | (q[:, 1] << 6) | q[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    p = cnt[inv].astype(np.float64) ** (power - 1.0)
    p /= p.sum()
    return rng.choice(len(rgb_flat), min(n, len(rgb_flat)), replace=False, p=p)
