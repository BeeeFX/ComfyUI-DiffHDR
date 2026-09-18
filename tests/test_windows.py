"""Sliding-window scheduling and blending checks. No GPU, no weights."""

import os
import sys

import torch

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)

from diffhdr_runtime import windows  # noqa: E402

fails = []


def ck(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


print("coverage: every frame is produced by at least one window")
for total in (1, 17, 33, 34, 80, 81, 200):
    for ws, st in ((33, 16), (33, 33), (33, 8), (17, 9)):
        wins = windows.plan(total, ws, st)
        covered = set()
        for w in wins:
            covered |= set(range(w["start"], w["end"]))
        ck(f"total={total} ws={ws} stride={st}", covered == set(range(total)),
           f"({len(wins)} windows)")

print("boundary flags")
w = windows.plan(80, 33, 16)
ck("first window flagged", w[0]["is_first"] and not w[0]["is_last"])
ck("last window flagged", w[-1]["is_last"])
ck("no interior window is flagged", not any(x["is_first"] or x["is_last"] for x in w[1:-1]))

print("padding only on the final short window")
w = windows.plan(40, 33, 16)
ck("short tail padded to window_size", w[-1]["pad"] == 33 - w[-1]["real"], f"pad={w[-1]['pad']}")
ck("full windows unpadded", all(x["pad"] == 0 for x in w if x["real"] == 33))

print("ramp weights sum to 1 across an overlap (no exposure pulse at the seam)")
border = 33 - 16
a = windows.ramp(33, True, False, border)
b = windows.ramp(33, False, False, border)
overlap = a[-border:] + b[:border]
ck("overlapping ramps sum to 1", torch.allclose(overlap, torch.ones(border), atol=1e-6),
   f"max dev={(overlap - 1).abs().max():.2e}")

print("blend of a constant sequence is that constant (the real exposure test)")
total, ws, st = 80, 33, 16
acc = windows.Accumulator(total, ws, st)
for win in windows.plan(total, ws, st):
    n = win["real"] + win["pad"]
    acc.add(win, torch.full((n, 4, 4, 3), 7.5))
out = acc.result()
ck("constant preserved everywhere", torch.allclose(out, torch.full_like(out, 7.5), atol=1e-5),
   f"min={out.min():.4f} max={out.max():.4f}")

print("blend preserves a temporal ramp (no drift or smear)")
acc = windows.Accumulator(total, ws, st)
src = torch.arange(total, dtype=torch.float32).view(total, 1, 1, 1).expand(total, 4, 4, 3).contiguous()
for win in windows.plan(total, ws, st):
    chunk = src[win["start"]:win["end"]].clone()
    if win["pad"]:
        chunk = torch.cat([chunk, chunk[-1:].repeat(win["pad"], 1, 1, 1)])
    acc.add(win, chunk)
out = acc.result()
ck("linear ramp reconstructed", torch.allclose(out, src, atol=1e-4),
   f"max err={(out - src).abs().max():.2e}")

print("invalid configurations are rejected")
for bad, msg in (((80, 33, 40), "stride > window"), ((80, 0, 8), "zero window"), ((80, 33, 0), "zero stride")):
    try:
        windows.plan(*bad)
        ck(msg, False, "was accepted")
    except ValueError:
        ck(msg, True)

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all window checks passed")
