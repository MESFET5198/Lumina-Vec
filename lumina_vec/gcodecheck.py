"""Compare sliced G-code: how much of the print is tiny islands, travel and retraction.

    python -m lumina_vec.gcodecheck before.gcode after.gcode
"""
import json, math, os, re, sys

G = re.compile(r"([XYE])(-?[\d.]+)")


def analyse(path):
    x = y = 0.0; typ = ""; cur = 0.0
    ext = trav = 0.0; loops = tiny = retr = 0; est = None
    for line in open(path, errors="ignore"):
        if line.startswith(";TYPE:"):
            typ = line[6:].strip(); continue
        if "estimated printing time" in line and est is None:
            est = line.split("=")[-1].strip()
        if not line.startswith(("G1 ", "G0 ")):
            continue
        d = dict(G.findall(line.split(";")[0]))
        nx, ny, e = float(d.get("X", x)), float(d.get("Y", y)), float(d.get("E", 0))
        L = math.hypot(nx - x, ny - y)
        if e < 0:
            retr += 1
        if e > 0 and L > 0:
            ext += L
            if typ == "Outer wall":
                cur += L
        elif L > 0:
            trav += L
            if cur > 0:
                loops += 1; tiny += cur < 2.0; cur = 0.0
        x, y = nx, ny
    return {"estimated_time": est, "extrude_m": round(ext / 1000, 1), "travel_m": round(trav / 1000, 1),
            "outer_wall_loops": loops, "loops_under_2mm": tiny, "retractions": retr}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(os.path.basename(p), json.dumps(analyse(p), ensure_ascii=False))
