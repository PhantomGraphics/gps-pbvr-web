"""Writes tests/fixtures/synthetic_sh1.ply: a small 3DGS-style scene (binary little-endian, SH degree 1,
Y-down like COLMAP data). Deterministic; used by the browser viewer check. No real data."""
import math, random, struct, pathlib

random.seed(7)
C0 = 0.28209479177387814


def logit(p):
    return math.log(p / (1 - p))


def dc(rgb):
    return [(c - 0.5) / C0 for c in rgb]


rows = []


def splat(pos, scale, rgb, opacity, q=(1, 0, 0, 0), rest=None):
    rest = rest or [0.0] * 9
    rows.append(list(pos) + [0, 0, 0] + dc(rgb) + rest + [logit(opacity)]
                + [math.log(s) for s in scale] + list(q))


# ground disk (y = +0.5 is "down"), checker-ish colouring
for _ in range(220):
    r = 2.2 * math.sqrt(random.random())
    a = random.random() * 2 * math.pi
    x, z = r * math.cos(a), r * math.sin(a)
    c = (0.75, 0.75, 0.7) if (int(math.floor(x * 2)) + int(math.floor(z * 2))) % 2 else (0.35, 0.4, 0.5)
    splat((x, 0.5, z), (0.22, 0.02, 0.22), c, 0.9)

# three coloured blobs standing on it (up is -y)
for cx, cz, col in [(-0.8, 0.0, (0.9, 0.25, 0.2)), (0.6, -0.5, (0.2, 0.7, 0.3)), (0.3, 0.8, (0.25, 0.4, 0.95))]:
    for _ in range(60):
        u, v, w = random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)
        n = math.sqrt(u * u + v * v + w * w) or 1
        p = (cx + 0.35 * u / n, 0.5 - 0.35 - 0.35 * v / n, cz + 0.35 * w / n)
        q = [random.gauss(0, 1) for _ in range(4)]
        qn = math.sqrt(sum(t * t for t in q))
        # band-1 coefficients so SH>0 data is exercised (ignored by the phase-2 renderer)
        splat(p, (0.14, 0.09, 0.06), col, 0.8, tuple(t / qn for t in q), [0.2 * random.gauss(0, 1) for _ in range(9)])

names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(9)]
         + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(rows)
       + "".join(f"property float {n}\n" for n in names) + "end_header\n")
out = pathlib.Path(__file__).parent / "fixtures" / "synthetic_sh1.ply"
with open(out, "wb") as f:
    f.write(hdr.encode("ascii"))
    for r in rows:
        f.write(struct.pack("<%df" % len(names), *r))
print(out, len(rows), "gaussians", out.stat().st_size, "bytes")
