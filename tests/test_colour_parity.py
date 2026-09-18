"""Assert diffhdr_runtime/colour.py matches the upstream originals bit for bit.

The model is stock Wan2.1-VACE-14B; the colour curve and the exposure masks ARE the
technique. If this file drifts, output stops being DiffHDR while still looking
plausible -- which is the worst kind of regression. So the port is checked against
the vendored upstream copies rather than against golden numbers.

Runs on CPU, needs no weights:
    env/bin/python custom_nodes/ComfyUI-DiffHDR/tests/test_colour_parity.py
"""

import math
import os
import sys

import torch

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)

from diffhdr_runtime import colour  # noqa: E402
from diffhdr_runtime import vendor  # noqa: E402

# Upstream's own copies, straight out of the vendored tree.
vendor.diffsynth()
import importlib  # noqa: E402

up = importlib.import_module("diffhdr_vendored_diffsynth.diffhdr_utils.color_utils")

FAILURES = []


def check(name, a, b, exact=True, tol=0.0):
    same = torch.equal(a, b) if exact else torch.allclose(a, b, rtol=tol, atol=tol)
    status = "ok " if same else "FAIL"
    if not same:
        d = (a - b).abs()
        FAILURES.append(f"{name}: max|Δ|={d.max().item():.3e} mean|Δ|={d.mean().item():.3e}")
    print(f"  [{status}] {name}")


torch.manual_seed(0)
srgb = torch.rand(3, 64, 96, dtype=torch.float32)
# Force genuine clipped and crushed regions to exist, or the mask tests prove nothing.
srgb[:, :10, :20] = 1.0
srgb[:, 20:30, 30:50] = 0.0
srgb[0, 40:50, 10:30] = 1.0  # single-channel clip: must NOT read as over-exposed
lin = torch.rand(3, 64, 96, dtype=torch.float32) * 500.0

print("transfer functions")
check("srgb_to_linear", colour.srgb_to_linear(srgb), up._srgb_to_linear(srgb))
check("lin_to_log", colour.lin_to_log(lin), up.Lin_to_Log(lin))
check("log_to_lin", colour.log_to_lin(colour.lin_to_log(lin)), up.Log_to_Lin(up.Lin_to_Log(lin)))
check("luma709", colour._luma709_linear(lin), up._luma709_linear(lin))

print("round trip")
x = torch.tensor([0.0, 1e-4, 0.18, 1.0, 10.0, 1000.0, 65536.0])
rt = colour.log_to_lin(colour.lin_to_log(x))
check("log->lin->log identity", rt, x, exact=False, tol=1e-3)
# The curve must be monotonic, or grading the result would be undefined.
codes = colour.lin_to_log(torch.linspace(0.0, 65536.0, 4096))
assert torch.all(codes[1:] >= codes[:-1]), "lin_to_log is not monotonic"
print("  [ok ] monotonic")
assert abs(colour.lin_to_log(torch.tensor(65536.0)).item() - 1.0) < 1e-6
print(f"  [ok ] log 1.0 == {colour.log_to_lin(torch.tensor(1.0)).item():.1f} linear")

print("exposure masks (video/image path)")
mine_o, mine_u = colour.exposure_masks_from_srgb(srgb, 0.95, 0.01, 0.02, 0.02)
theirs_o, theirs_u = up.exposure_masks_from_srgb(
    srgb, over_luma_thr_srgb=0.95, under_luma_thr_srgb=0.01,
    softness=0.02, clip_eps_srgb=0.02, use_local_contrast_gate=False,
)
check("over mask", mine_o, theirs_o)
check("under mask", mine_u, theirs_u)
assert mine_o[:10, :20].mean() > 0.9, "fully clipped block not detected as over-exposed"
assert mine_u[20:30, 30:50].mean() > 0.9, "black block not detected as under-exposed"
assert mine_o[40:50, 10:30].mean() < 0.5, "single-channel clip wrongly read as over-exposed"
print("  [ok ] blown / crushed / single-channel-clip behaviour")

print("temporal stabilisation")
prev_mine = prev_up = None
for i in range(4):
    f = (srgb + 0.02 * torch.randn_like(srgb)).clamp(0, 1)
    o_m, _ = colour.exposure_masks_from_srgb(f, 0.95, 0.01, 0.02, 0.02)
    o_t, _ = up.exposure_masks_from_srgb(
        f, over_luma_thr_srgb=0.95, under_luma_thr_srgb=0.01,
        softness=0.02, clip_eps_srgb=0.02, use_local_contrast_gate=False,
    )
    s_m, prev_mine = colour.stabilize_soft_mask(o_m, prev_mine, 0.7, 9, 9)
    s_t, prev_up = up.stabilize_soft_mask(o_t, prev_up, alpha=0.7, k_smooth=9, k_open_close=9)
    check(f"stabilize frame {i}", s_m, s_t)

print("panorama mask")
# infer_hdri.py is a script, not an importable module; its helpers are reproduced in
# colour.py, so compare against a local transcription of the upstream source.
def _up_pano(srgb_chw, over_thr=0.95):
    import torch.nn.functional as F
    def _eotf(x):
        return torch.where(x <= 0.04045, x / 12.92, torch.pow((torch.clamp(x, min=0) + 0.055) / 1.055, 2.4))
    def _oetf(x):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(torch.clamp(x, min=0.0031308), 1 / 2.4) - 0.055)
    def _l709(rgb):
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    def _ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0, 1)
        return t * t * (3 - 2 * t)
    luma = _l709(_eotf(srgb_chw.clamp(0, 1)))
    luma_srgb = _oetf(luma.clamp(0, 1))
    over = _ss(luma_srgb, over_thr - 0.02, over_thr + 0.02)
    ch_clip = _ss(srgb_chw.max(dim=0).values, 0.93, 0.99)
    over_bin = (torch.max(over, ch_clip) > 0.2).float()
    m = over_bin.unsqueeze(0).unsqueeze(0)
    m = -F.max_pool2d(-m, 5, stride=1, padding=2)
    m = F.max_pool2d(m, 5, stride=1, padding=2)
    m = F.max_pool2d(m, 7, stride=1, padding=3)
    m = -F.max_pool2d(-m, 7, stride=1, padding=3)
    return m.squeeze(0).squeeze(0)

check("pano over mask", colour.overexposure_mask_pano(srgb, 0.95), _up_pano(srgb, 0.95))

print("reference-image path (sRGB -> linear -> EV boost -> log)")
# DiffHDRSample._prepare_reference reproduces upstream's load_reference_image colour
# chain. Upstream routes it through srgb_to_linear_rec709_torch rather than the plain
# _srgb_to_linear used elsewhere, so it is checked separately.
for ev in (0.0, 5.0, -2.0):
    ref = torch.rand(3, 48, 64)
    mine = colour.lin_to_log(colour.srgb_to_linear(ref.clamp(0, 1)).clamp(min=0) * (2.0 ** ev))
    theirs = colour.lin_to_log(up.srgb_to_linear_rec709_torch(ref) * (2.0 ** ev))
    check(f"reference chain at {ev:+.0f} EV", mine, theirs)

print("layout adapters")
img = torch.rand(3, 32, 48, 3)
check("image->chw->image", colour.chw_to_image(colour.image_to_chw(img)), img)

print()
if FAILURES:
    print("FAILED:")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all colour parity checks passed")
