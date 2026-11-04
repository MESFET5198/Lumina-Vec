# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MESFET5198 - Lumina-Vec (https://github.com/MESFET5198/Lumina-Vec)
"""Minimal rasteriser for our own path data (M/L/H/V/C/Z, absolute): id maps for colour sampling."""
import re
import numpy as np
import cv2

_TOK = re.compile(r"[MLHVCZmlhvcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def path_polys(d, n_bez=8):
    toks = _TOK.findall(d)
    polys, cur, pt, start, i, cmd = [], [], np.zeros(2), np.zeros(2), 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t.upper(); i += 1
            if cmd == "Z":
                if cur:
                    polys.append(np.array(cur)); cur = []
                pt = start.copy()
                continue
            continue
        if cmd == "M":
            if cur:
                polys.append(np.array(cur))
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2
            start = pt.copy(); cur = [pt]; cmd = "L"
        elif cmd == "L":
            pt = np.array([float(toks[i]), float(toks[i + 1])]); i += 2; cur.append(pt)
        elif cmd == "H":
            pt = np.array([float(toks[i]), pt[1]]); i += 1; cur.append(pt)
        elif cmd == "V":
            pt = np.array([pt[0], float(toks[i])]); i += 1; cur.append(pt)
        elif cmd == "C":
            c = np.array([float(v) for v in toks[i:i + 6]]).reshape(3, 2); i += 6
            t_ = np.linspace(0, 1, n_bez + 1)[1:, None]
            b = ((1 - t_) ** 3) * pt + 3 * ((1 - t_) ** 2) * t_ * c[0] + 3 * (1 - t_) * t_ ** 2 * c[1] + t_ ** 3 * c[2]
            cur.extend(list(b)); pt = c[2]
        else:
            i += 1
    if cur:
        polys.append(np.array(cur))
    return polys


def rasterize_ids(ds, W, H, ss=2):
    """int32 map (H*ss x W*ss): 1-based index of the last path covering each pixel (even-odd)."""
    lab = np.zeros((H * ss, W * ss), np.int32)
    for k, d in enumerate(ds, start=1):
        polys = [np.round(p * ss * 16).astype(np.int32) for p in path_polys(d) if len(p) >= 3]
        if not polys:
            continue
        allp = np.vstack(polys) >> 4
        x0, y0 = np.maximum(allp.min(0) - 1, 0); x1, y1 = allp.max(0) + 2
        x1, y1 = min(x1, W * ss), min(y1, H * ss)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0]) * 16
        cv2.fillPoly(m, [p - off for p in polys], 1, lineType=cv2.LINE_8, shift=4)
        lab[y0:y1, x0:x1][m > 0] = k
    return lab
