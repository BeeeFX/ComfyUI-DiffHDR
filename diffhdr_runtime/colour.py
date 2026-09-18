"""DiffHDR's colour maths and exposure-mask detection, ported to ComfyUI IMAGE layout.

This module is the whole of DiffHDR's "HDR", and it is the part most worth getting
exactly right: the model itself is stock Wan2.1-VACE-14B. What makes the output high
dynamic range is that the plate is handed to VACE on a log curve, that the blown and
crushed regions are marked as the inpaint mask, and that the result is taken back to
linear afterwards without a display clamp.

Ported rather than called through the vendored tree on purpose. Upstream's copies live
in ``utils/color_utils.py`` (and a second, subtly different set in ``infer_hdri.py``),
reached by an absolute ``from utils...`` import that cannot survive inside ComfyUI --
see ``DiffHDR/VENDORED.md``. Every function below is op-for-op identical to its
upstream original so results match the reference implementation bit for bit;
``tests/test_colour_parity.py`` asserts that against the vendored copies and fails the
build if this file ever drifts.

Layout note: ComfyUI IMAGE is ``(B, H, W, C)`` float. Upstream works per frame in
``(C, H, W)``. The adapters at the bottom convert; the maths in the middle keeps
upstream's own layout so the two can be compared directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Upstream's defaults. max_val 65536 is also the ceiling Lin_to_Log clamps to, so a
# log code of 1.0 corresponds to exactly 65536.0 linear -- 16 stops above diffuse
# white, which is why clamping the log signal at 1.0 costs nothing in practice.
LOG_MAX_VAL = 65536.0
LOG_GAMMA = 2.2


# --------------------------------------------------------------------------- #
# Transfer functions (utils/color_utils.py)
# --------------------------------------------------------------------------- #

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF. Upstream ``_srgb_to_linear``."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """sRGB OETF. Upstream ``_srgb_oetf`` (infer_hdri.py)."""
    a = 0.055
    t = 0.0031308
    return torch.where(x <= t, 12.92 * x, (1 + a) * torch.pow(torch.clamp(x, min=t), 1 / 2.4) - a)


def lin_to_log(im: torch.Tensor, max_val: float = LOG_MAX_VAL, gamma: float = LOG_GAMMA) -> torch.Tensor:
    """Linear -> DiffHDR log. Upstream ``Lin_to_Log``.

    Not a standard camera log curve: a plain mu-law-ish compression that maps
    [0, max_val] onto [0, 1], with a final 1/gamma power to spend more code values
    on the shadows. The model was trained on this exact curve, so it is not
    interchangeable with LogC/ACEScct.
    """
    im = torch.clamp(im, max=max_val)
    im = torch.log(gamma * im + 1.0) / math.log(gamma * max_val + 1.0)
    return torch.pow(im, 1.0 / gamma)


def log_to_lin(im: torch.Tensor, max_val: float = LOG_MAX_VAL, gamma: float = LOG_GAMMA) -> torch.Tensor:
    """DiffHDR log -> linear. Upstream ``Log_to_Lin``. Exact inverse of :func:`lin_to_log`."""
    im = torch.pow(im, gamma)
    im = torch.exp(im * math.log(gamma * max_val + 1.0)) - 1.0
    return im / gamma


def _luma709_linear(rgb_lin: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma of a linear [3,H,W] frame. Upstream ``_luma709_linear``."""
    return 0.2126 * rgb_lin[0] + 0.7152 * rgb_lin[1] + 0.0722 * rgb_lin[2]


def _smoothstep(x: torch.Tensor, edge0, edge1) -> torch.Tensor:
    """Upstream ``_smoothstep`` (color_utils variant -- no epsilon on the denominator)."""
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _smoothstep_eps(x: torch.Tensor, e0, e1) -> torch.Tensor:
    """Upstream ``_smoothstep`` (infer_hdri variant -- carries a 1e-8 guard)."""
    t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- #
# Exposure masks -- video/image path (utils/color_utils.py)
# --------------------------------------------------------------------------- #

def exposure_masks_from_srgb(
    rgb_srgb_chw: torch.Tensor,
    over_luma_thr_srgb: float = 0.95,
    under_luma_thr_srgb: float = 0.01,
    softness: float = 0.02,
    clip_eps_srgb: float = 0.01,
):
    """Soft over/under-exposure masks from one sRGB frame. Upstream ``exposure_masks_from_srgb``.

    Args:
        rgb_srgb_chw: ``[3,H,W]`` float32 sRGB in [0,1].
    Returns:
        ``(over, under)``, each ``[H,W]`` float in [0,1].

    Both masks are a 0.7/0.3 blend of a soft luma threshold and a hard channel test,
    which is what stops a merely bright-but-detailed highlight from being marked as
    blown. The optional local-contrast gate upstream exposes is omitted: it is off in
    every one of upstream's own entry points.
    """
    device = rgb_srgb_chw.device
    rgb_lin = srgb_to_linear(rgb_srgb_chw.clamp(0, 1))
    Y = _luma709_linear(rgb_lin)

    # Over: soft luma threshold, thresholds linearised so the comparison happens in
    # the same domain as Y.
    over_l = _smoothstep(
        Y,
        srgb_to_linear(torch.tensor(over_luma_thr_srgb - softness, device=device)),
        srgb_to_linear(torch.tensor(over_luma_thr_srgb + softness, device=device)),
    )
    # ...plus "at least two channels are pinned at white", the actual signature of clipping.
    near1 = (rgb_srgb_chw >= (1.0 - clip_eps_srgb)).float()
    ch_clip = (near1.sum(dim=0) >= 2).float()
    over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)

    # Under: inverse soft threshold, plus "all channels low" to avoid catching
    # saturated dark colour rather than genuine crush.
    under = 1.0 - _smoothstep(
        Y,
        srgb_to_linear(torch.tensor(max(0, under_luma_thr_srgb - softness), device=device)),
        srgb_to_linear(torch.tensor(under_luma_thr_srgb + softness, device=device)),
    )
    all_low = (rgb_srgb_chw <= under_luma_thr_srgb).all(dim=0).float()
    under = (0.7 * under + 0.3 * all_low).clamp(0, 1)

    def _morph_smooth(m):
        m4 = m.unsqueeze(0).unsqueeze(0)
        m4 = F.max_pool2d(m4, 3, stride=1, padding=1)      # dilate
        m4 = -F.max_pool2d(-m4, 3, stride=1, padding=1)    # erode
        return m4.squeeze(0).squeeze(0)

    return _morph_smooth(over).clamp(0, 1), _morph_smooth(under).clamp(0, 1)


def stabilize_soft_mask(
    soft: torch.Tensor,
    prev_ema: torch.Tensor | None,
    alpha: float = 0.7,
    k_smooth: int = 9,
    k_open_close: int = 9,
):
    """Temporal EMA + spatial clean-up for one frame's mask. Upstream ``stabilize_soft_mask``.

    Returns ``(stabilised, ema_state)``; feed ``ema_state`` back in as ``prev_ema`` on
    the next frame. Without this the mask boils frame to frame and the model is asked
    to invent a slightly different region each time, which reads as flicker in the
    recovered highlights.

    Note the EMA state is captured *before* the spatial filtering, so blur and
    morphology do not compound across the sequence.
    """
    ema = soft if prev_ema is None else (alpha * soft + (1 - alpha) * prev_ema)
    ema_state = ema.clone()

    if k_smooth > 1:
        p = k_smooth // 2
        ema = F.avg_pool2d(ema[None, None], k_smooth, 1, p)[0, 0]

    def erode(b, k):
        return (-F.max_pool2d(-b[None, None], k, 1, k // 2))[0, 0]

    def dilate(b, k):
        return F.max_pool2d(b[None, None], k, 1, k // 2)[0, 0]

    if k_open_close > 1:
        ema = dilate(erode(ema, k_open_close), k_open_close)   # open: kill specks
        ema = erode(dilate(ema, k_open_close), k_open_close)   # close: fill pinholes

    return ema.clamp(0, 1), ema_state


# --------------------------------------------------------------------------- #
# Exposure mask -- panorama path (infer_hdri.py)
# --------------------------------------------------------------------------- #

def overexposure_mask_pano(srgb_chw: torch.Tensor, over_thr: float = 0.95, no_morph: bool = False) -> torch.Tensor:
    """Over-exposure mask for HDRI panoramas. Upstream ``make_overexposure_mask``.

    Deliberately a different function from :func:`exposure_masks_from_srgb`, because
    upstream's pano entry point uses different maths and the Pano LoRA was trained
    with it: the threshold is applied to *re-encoded* sRGB luma, the channel test is a
    soft ``max`` rather than a weighted sum, and the morphology is a 5px open followed
    by a 7px close instead of a single 3px pass. Panoramas also get no under-exposure
    mask at all -- for an HDRI it is the sun and sky clipping that matter.

    Returns a binary ``[H,W]`` float mask.
    """
    luma = _luma_709_chw(srgb_to_linear(srgb_chw.clamp(0, 1)))
    luma_srgb = linear_to_srgb(luma.clamp(0, 1))
    over = _smoothstep_eps(luma_srgb, over_thr - 0.02, over_thr + 0.02)
    ch_max = srgb_chw.max(dim=0).values
    ch_clip = _smoothstep_eps(ch_max, 0.93, 0.99)
    over = torch.max(over, ch_clip)
    over_bin = (over > 0.2).float()

    if not no_morph:
        m = over_bin.unsqueeze(0).unsqueeze(0)
        m = -F.max_pool2d(-m, 5, stride=1, padding=2)   # open
        m = F.max_pool2d(m, 5, stride=1, padding=2)
        m = F.max_pool2d(m, 7, stride=1, padding=3)     # close
        m = -F.max_pool2d(-m, 7, stride=1, padding=3)
        over_bin = m.squeeze(0).squeeze(0)

    return over_bin


def _luma_709_chw(rgb: torch.Tensor) -> torch.Tensor:
    """Upstream ``_luma_709`` (infer_hdri.py) -- same weights as the color_utils one."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# --------------------------------------------------------------------------- #
# ComfyUI adapters
# --------------------------------------------------------------------------- #

def image_to_chw(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE ``(B,H,W,C)`` -> ``(B,C,H,W)``, dropping any alpha."""
    return image[..., :3].permute(0, 3, 1, 2).contiguous()


def chw_to_image(video: torch.Tensor) -> torch.Tensor:
    """``(B,C,H,W)`` -> ComfyUI IMAGE ``(B,H,W,C)``."""
    return video.permute(0, 2, 3, 1).contiguous()
