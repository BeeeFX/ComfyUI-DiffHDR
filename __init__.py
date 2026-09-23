"""ComfyUI-DiffHDR -- LDR to scene-linear HDR with Eyeline Labs' DiffHDR.

DiffHDR treats LDR->HDR as generative radiance inpainting: it runs Wan2.1-VACE-14B
with a small LoRA, conditioned on the plate converted to a log curve, with the blown
and crushed regions marked as the VACE inpaint mask, and takes the result back to
linear without a display clamp.

What this pack is, in one line: a thin wrapper around the upstream pipeline that lets
you point it at the Wan weights you already have, at a precision your card can hold,
and get a plain IMAGE back out.

Three things worth knowing before reading the code:

* The upstream tree is VENDORED under ./DiffHDR rather than cloned on first run, and
  is loaded under a private module name so it never claims the global ``diffsynth``
  name. Two of its imports had to be rewritten to survive inside ComfyUI at all --
  see DiffHDR/VENDORED.md.
* Checkpoints are validated from their safetensors header before loading. Several
  plausibly-named Wan files do NOT work (the split "VACE module" releases, the
  non-encoder umt5 build, fp8_scaled repacks); the pre-flight names the problem in
  under a second instead of failing obscurely minutes into a 34 GB read.
* The LoRA is merged into the VACE weights and the match count is checked. If it
  merges into nothing you get ordinary Wan2.1-VACE video -- plausible, and not HDR --
  so that case is a hard error, not a warning.

Output is an unclamped scene-linear IMAGE. This pack writes no files: wire it into
whatever EXR writer your pipeline uses.

Changelog
---------
2026-09-16: Initial version. Vendored upstream at cfc1b48. Nodes: DiffHDR_LoadModel,
            DiffHDR_Encode, DiffHDR_Sample, DiffHDR_FreeMemory. Covers all four
            upstream entry points (video, single image, long video via overlapping
            windows, HDRI panorama).
2026-09-23: 0.2.0. Fixes: example workflows had every Sample value after `seed`
            shifted by one widget (sigma_shift=33, tea_cache=1.0) and a non-UUID id;
            a released pipeline (Free Memory, or a second loader) now rebuilds on
            demand instead of failing every run while the loader stayed cached; Free
            Memory takes an `images` pass-through so it can run after Sample;
            loader dropdowns no longer list files twice (unet/clip aliases); the
            tokenizer is found via extra_model_paths.yaml and symlinks; a still is
            capped at window_size (720p stills OOM'd on 24 GB); a batch of panoramas
            runs one frame each instead of as a clip; resize_mode=none generates at
            the frames' size. Added: Encode `mask` input (add/subtract/replace) and
            final `mask` output; seed defaults to fixed; one progress bar per job;
            ComfyUI's models are unloaded before sampling; antialiased downscale;
            Encode works a frame at a time (same result, far lower peak RAM).
            Vendored fix: diffsynth's VRAM management gave VACE the VAE's dtype, so
            VACE sat in fp32 (11.4 GiB instead of 2.8 at fp8) -- the reason
            full_gpu + fp8 needed 24.7 GiB of weights; now 16.2. Output
            bit-identical. Sample logs tokens and peak VRAM per pass.
"""

import logging

logging.getLogger("ComfyUI-DiffHDR").setLevel(logging.INFO)

# Node class keys are a permanent API -- they are what saved workflows store as
# `class_type`. Renaming one silently breaks every graph that used it, so these are
# chosen once and kept.
from .diffhdr_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

DIFFHDR_VERSION = "0.2.0"

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
