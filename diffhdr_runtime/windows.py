"""Sliding-window scheduling and temporal blending for clips longer than one window.

The model is trained at a fixed window (33 frames). Longer clips are run as
overlapping windows and cross-faded, which is what ``infer_long_video.py`` does.

Two details from upstream that are easy to get wrong and both visible in the result:

* the fade happens in **linear** space, after Log->Lin, not on the log signal. Blending
  two log-encoded frames and then decoding gives a different (darker) mid-point;
* the ramp is ``(i + 0.5) / border_width``, not ``i / border_width``. The half-sample
  offset keeps the two overlapping ramps summing to 1 everywhere, so a blended frame
  has the same exposure as an unblended one. Without it the seam reads as a pulse.

Upstream streams through a ring buffer to keep memory at O(window). Inside ComfyUI the
whole sequence is already a resident IMAGE tensor, so this accumulates over the full
length instead -- same arithmetic, far less machinery.
"""

from __future__ import annotations

import torch


def plan(total_frames: int, window_size: int, stride: int) -> list[dict]:
    """Window layout for ``total_frames``.

    Each entry gives the source slice, how many frames were repeated to pad a short
    final window up to ``window_size`` (the model degrades if run below its trained
    length), and whether the window sits at a sequence boundary -- boundary edges are
    not faded, or the first and last frames would fade to black.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("window_stride must be positive")
    if stride > window_size:
        raise ValueError(
            f"window_stride ({stride}) cannot exceed window_size ({window_size}) -- "
            "that would leave gaps between windows."
        )

    windows: list[dict] = []
    for start in range(0, total_frames, stride):
        # Skip a window the previous one already covered to the end.
        if start - stride >= 0 and start - stride + window_size >= total_frames:
            break
        end = min(start + window_size, total_frames)
        real = end - start
        windows.append(
            {
                "start": start,
                "end": end,
                "real": real,
                "pad": max(0, window_size - real),
                "is_first": start == 0,
                "is_last": end >= total_frames,
            }
        )
    return windows


def ramp(length: int, is_first: bool, is_last: bool, border_width: int,
         device=None, dtype=torch.float32) -> torch.Tensor:
    """Per-frame blend weights for one window. Upstream ``build_1d_mask``."""
    x = torch.ones((length,), device=device, dtype=dtype)
    if border_width <= 0:
        return x
    shift = 0.5  # half-sample offset: see module docstring
    if not is_first:
        x[:border_width] = (torch.arange(border_width, device=device, dtype=dtype) + shift) / border_width
    if not is_last:
        x[-border_width:] = torch.flip(
            (torch.arange(border_width, device=device, dtype=dtype) + shift) / border_width,
            dims=(0,),
        )
    return x


class Accumulator:
    """Weighted sum of linear-space window outputs, normalised on ``result()``."""

    def __init__(self, total_frames: int, window_size: int, stride: int):
        self.total = total_frames
        self.border = window_size - stride
        self.value: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None

    def add(self, window: dict, frames_linear: torch.Tensor) -> None:
        """``frames_linear``: ``(L,H,W,C)`` linear frames for this window, padding included."""
        real = window["real"]
        frames_linear = frames_linear[:real]  # drop the repeated padding frames
        if self.value is None:
            h, w, c = frames_linear.shape[1:]
            self.value = torch.zeros((self.total, h, w, c), dtype=torch.float32)
            self.weight = torch.zeros((self.total,), dtype=torch.float32)

        w1d = ramp(real, window["is_first"], window["is_last"], self.border,
                   device=frames_linear.device, dtype=torch.float32)
        s, e = window["start"], window["start"] + real
        self.value[s:e] += (frames_linear.float() * w1d.view(real, 1, 1, 1)).cpu()
        self.weight[s:e] += w1d.cpu()

    def result(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("ComfyUI-DiffHDR: no windows were accumulated.")
        return self.value / self.weight.view(-1, 1, 1, 1).clamp(min=1e-8)
