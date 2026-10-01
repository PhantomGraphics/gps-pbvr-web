"""Validates the trajectory ZIP written by the viewer (used by tools-cdp-session.mjs)."""
import hashlib, json, sys, zipfile

z = zipfile.ZipFile(sys.argv[1])
assert z.testzip() is None, 'corrupt entry (CRC)'
names = z.namelist()
print(names)
pngs = [n for n in names if n.endswith('.png')]
assert len(pngs) == 3 and 'stats.csv' in names and 'session.json' in names, 'unexpected entries'
assert all(z.read(n)[:8] == b'\x89PNG\r\n\x1a\n' for n in pngs), 'not PNG'
h = [hashlib.md5(z.read(n)).hexdigest() for n in sorted(pngs)]
# a full 360-degree orbit: the middle frame differs, the first and last are the same view with the same seed
assert h[0] != h[1] and h[0] == h[2], 'unexpected frame relationship'
assert len(z.read('stats.csv').decode().strip().split('\n')) == 4, 'stats rows'
json.loads(z.read('session.json'))
print('zip ok')
