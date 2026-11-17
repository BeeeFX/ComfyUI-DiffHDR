"""Where the weights live after diffsynth's VRAM management wraps them. No GPU, no weights.

Runs the vendored ``enable_vram_management`` and ``load_models_to_device`` for real, on
tiny instances of the real Wan DiT and VACE classes, and checks each model is resident
in its OWN dtype.

That was not true as vendored: the VACE section read a ``dtype`` variable left over
from the section before it -- the VAE's, which this pack loads in fp32 -- so VACE was
upcast to fp32 on every onload. For Wan2.1-VACE-14B that is 11.4 GiB instead of 2.8 GiB
at fp8, and it is why ``vram_mode=full_gpu`` could not fit a 24 GB card even at fp8
(24.7 GiB of weights; 16.2 GiB fixed). Numerically the upcast was lossless, so the fix
changes memory only -- see DiffHDR/VENDORED.md, item 5. This suite catches a re-vendor
that drops the fix.
"""

import importlib
import os
import sys

import torch

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)

from diffhdr_runtime import vendor  # noqa: E402

vendor.diffsynth()
wvn = importlib.import_module("diffhdr_vendored_diffsynth.pipelines.wan_video_new")
dit_mod = importlib.import_module("diffhdr_vendored_diffsynth.models.wan_video_dit")
vace_mod = importlib.import_module("diffhdr_vendored_diffsynth.models.wan_video_vace")

fails = []


def ck(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


def pipe_with(weight_dtype):
    """A pipeline shaped like the one runtime._build makes: DiT + VACE at the load dtype,
    VAE in fp32. Tiny dims; the wrapping logic does not depend on size."""
    pipe = wvn.WanVideoPipeline(device="cpu", torch_dtype=torch.bfloat16, vae_dtype=torch.float32)
    pipe.dit = dit_mod.WanModel(dim=256, in_dim=16, ffn_dim=512, out_dim=16, text_dim=64,
                                freq_dim=32, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                                num_layers=4, has_image_input=False).to(weight_dtype)
    pipe.vace = vace_mod.VaceWanModel(vace_layers=(0, 2), vace_in_dim=96, dim=256,
                                      num_heads=2, ffn_dim=512).to(weight_dtype)
    # Stands in for the fp32 Wan VAE: all that matters here is that it is fp32 and
    # wrapped just before VACE.
    pipe.vae = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1)).float()
    pipe.audio_encoder = None
    return pipe


def linear_dtypes(model):
    return {m.weight.dtype for m in model.modules() if isinstance(m, torch.nn.Linear)}


for weight_dtype in (torch.float8_e4m3fn, torch.bfloat16):
    name = str(weight_dtype).replace("torch.", "")
    for mode in ("full_gpu", "streaming"):
        print(f"{name}, {mode}")
        pipe = pipe_with(weight_dtype)
        if mode == "full_gpu":  # exactly as runtime._build calls it
            pipe.enable_vram_management(num_persistent_param_in_dit=10**12, vae_dtype=torch.float32)
        else:
            pipe.enable_vram_management(vram_limit=20.0, vae_dtype=torch.float32)
        pipe.load_models_to_device(["dit", "vace"])  # what __call__ does before denoising
        ck("DiT resident in its own dtype", linear_dtypes(pipe.dit) == {weight_dtype},
           str(linear_dtypes(pipe.dit)))
        ck("VACE resident in its own dtype, not the VAE's fp32",
           linear_dtypes(pipe.vace) == {weight_dtype}, str(linear_dtypes(pipe.vace)))
        pipe.load_models_to_device([])  # and back again
        ck("VACE offloaded in its own dtype", linear_dtypes(pipe.vace) == {weight_dtype})

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all VRAM dtype checks passed")
