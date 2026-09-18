"""ComfyUI nodes for DiffHDR (Eyeline Labs) -- LDR video/stills -> scene-linear HDR.

Three nodes, and the split follows what an artist actually adjusts:

    DiffHDR Load Model   which weights, at what precision, within what VRAM budget
    DiffHDR Encode       what the model is told to reinvent (the exposure masks)
    DiffHDR Sample       how hard it works at it

Encode is separate from Sample because the masks are the real creative control: they
decide which pixels are treated as missing radiance. Getting them wrong is the usual
cause of a disappointing result, and you cannot tune what you cannot see, so they come
out of the node as MASKs.

Output is an unclamped scene-linear IMAGE. Nothing here writes files -- wire it into
whatever your pipeline already uses (in this suite, the EXR-sequence format on
Bat_VideoCombine). Expect the node thumbnail to look blown out: ComfyUI previews clamp
to [0,1] and this data deliberately goes well above it.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils

from .diffhdr_runtime import colour, model_scan
from .diffhdr_runtime.windows import Accumulator, plan

logger = logging.getLogger("ComfyUI-DiffHDR")

# Upstream's negative prompt, verbatim. Inert at the default cfg_scale of 1.0 (no
# classifier-free guidance pass), kept so that raising cfg reproduces the reference
# behaviour rather than something subtly different.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Wan2.1 constraints: spatial multiple of 16, temporal 4k+1.
# Kept in step with runtime.DTYPES by tests/test_nodes.py -- listed here so
# INPUT_TYPES does not have to import the torch-heavy runtime module.
DIT_DTYPES = ["bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2"]

SPATIAL_MULTIPLE = 16
TRAINED_HEIGHT, TRAINED_WIDTH, TRAINED_FRAMES = 720, 1280, 33
_TRAINED_TOKENS = 9 * (TRAINED_HEIGHT // 16) * (TRAINED_WIDTH // 16)  # 32400


def _round_frames(n: int) -> int:
    """Round up to the next 4k+1, which is the only temporal length Wan2.1 accepts."""
    n = max(1, int(n))
    return n if n % 4 == 1 else ((n + 3) // 4) * 4 + 1


def _frames_for_still(height: int, width: int) -> int:
    """Frame count for a single still, matching upstream ``infer_image.compute_num_frames``.

    A still is fed to a video model by replicating it. Upstream picks the replication
    count so the DiT sees roughly the token count it was trained on (32400) -- at
    720x1280 that is 33 frames, and at higher resolutions proportionally fewer. Running
    a still at fewer tokens than training is what makes single-image output look weaker
    than video output.
    """
    spatial = (height // 16) * (width // 16)
    if spatial <= 0:
        return TRAINED_FRAMES
    return max(1, (max(1, int(np.ceil(_TRAINED_TOKENS / spatial))) - 1) * 4 + 1)


def _crop_and_resize(image: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    """Conform a (B,H,W,C) IMAGE to height x width. ``mode`` is crop_and_resize/stretch/none."""
    if mode == "none":
        return image
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    x = image.permute(0, 3, 1, 2)  # -> B,C,H,W for interpolate
    if mode == "crop_and_resize":
        # Centre-crop to the target aspect first, so nothing is squashed.
        if h / w < height / width:
            cw = max(1, int(h / height * width))
            left = (w - cw) // 2
            x = x[:, :, :, left:left + cw]
        else:
            ch = max(1, int(w / width * height))
            top = (h - ch) // 2
            x = x[:, :, top:top + ch, :]
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _mask_to_pil(mask_hw: torch.Tensor) -> Image.Image:
    """[H,W] float -> 3-channel uint8 PIL, the form diffsynth's VACE unit expects.

    The uint8 round trip is upstream's, not ours: the mask it consumes is already
    binarised, so nothing is lost.
    """
    arr = mask_hw.detach().clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2), mode="RGB")


class DiffHDRLoadModel:
    """Load Wan2.1-VACE-14B plus a DiffHDR LoRA into a cached pipeline."""

    DESCRIPTION = (
        "Loads the Wan2.1-VACE-14B backbone, T5 encoder, VAE and a DiffHDR LoRA into "
        "one cached pipeline. Checkpoints are validated from their safetensors header "
        "before loading, so a wrong file fails in a second with an explanation rather "
        "than minutes into a 34 GB read."
    )
    CATEGORY = "DiffHDR"
    RETURN_TYPES = ("DIFFHDR_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        # Evaluated lazily by ComfyUI. Deliberately not done at import: scanning model
        # folders on an NFS asset library would stall every ComfyUI start.
        model_scan.register_model_folder()
        dits = model_scan.list_candidates("dit") or ["<none found>"]
        t5s = model_scan.list_candidates("text_encoder") or ["<none found>"]
        vaes = model_scan.list_candidates("vae") or ["<none found>"]
        loras = model_scan.list_candidates("lora") or ["<none found>"]
        toks = [model_scan.AUTO] + model_scan.find_tokenizers()
        return {
            "required": {
                "wan_vace_model": (dits, {"tooltip":
                    "Wan2.1-VACE-14B. Must be the MERGED checkpoint (~1331 tensors), e.g. "
                    "wan2.1_vace_14B_fp16.safetensors. The split 'VACE module' files "
                    "published for ComfyUI-WanVideoWrapper will be rejected -- they hold "
                    "only the VACE blocks and expect a separate base at runtime."}),
                "text_encoder": (t5s, {"tooltip":
                    "umt5-xxl text encoder. Use the encoder-only build "
                    "(umt5-xxl-enc-bf16.safetensors). The full umt5_xxl_fp16 file has a "
                    "different key set and is rejected."}),
                "vae": (vaes, {"tooltip":
                    "Wan 2.1 VAE. Both the ComfyUI and Kijai builds work. Always run in "
                    "fp32 regardless of dit_dtype: it is small, runs once, and carries "
                    "the log signal back out."}),
                "lora_name": (loras, {"tooltip":
                    "DiffHDR.safetensors for video and stills, DiffHDR_Pano.safetensors "
                    "for equirectangular HDRI. This LoRA is what makes the output HDR; "
                    "if it fails to attach the node errors rather than quietly emitting "
                    "ordinary Wan video."}),
                "tokenizer_dir": (toks, {"tooltip":
                    "google/umt5-xxl tokenizer directory (~5 MB). 'auto' searches your "
                    "models folder -- it ships inside the official Wan repos, so it is "
                    "usually already present."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip":
                    "Upstream uses 1.0. The LoRA is merged into the weights, so changing "
                    "this rebuilds the pipeline. Below ~0.8 the highlight reconstruction "
                    "weakens towards plain Wan behaviour."}),
                "speed_lora": ([model_scan.NONE] + loras, {"tooltip":
                    "Optional step-distillation LoRA for the DiT, e.g. lightx2v "
                    "cfg_step_distill. These patch blocks.* while DiffHDR patches "
                    "vace_blocks.*, so they compose rather than conflict.\n\n"
                    "The win is steps: ~10 down to ~4. The CFG half of a 'cfg_step' "
                    "distill buys nothing here, because DiffHDR already runs at "
                    "cfg_scale 1.0.\n\n"
                    "It is a real change to the model, not a free speedup -- check the "
                    "highlight reconstruction still looks right on your material before "
                    "adopting it. Drop 'steps' on the Sample node when you enable it."}),
                "speed_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Strength for speed_lora. 1.0 is normal; lower it if the "
                               "distillation is flattening the recovered highlights."}),
                "dit_dtype": (DIT_DTYPES,
                    {"default": "bf16", "tooltip":
                    "Load precision for the DiT. bf16 is the reference (~34 GB); "
                    "fp8_e4m3fn roughly halves that (~17 GB) by casting on load and is "
                    "the practical choice on a 24-32 GB card. Note this casts the fp16 "
                    "file -- pre-quantised '_fp8_scaled' repacks carry extra scale "
                    "tensors that diffsynth cannot identify, and are rejected."}),
                "vram_mode": (["auto", "full_gpu", "streaming"], {"default": "auto", "tooltip":
                    "auto: upstream's default -- budget from total VRAM and stream DiT "
                    "blocks from system RAM.\n"
                    "full_gpu: keep everything resident. Fastest, needs the whole DiT in "
                    "VRAM.\n"
                    "streaming: same as auto but with an explicit budget from "
                    "vram_limit_gb, for sharing a card."}),
                "vram_limit_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5,
                    "tooltip": "VRAM budget in GB for vram_mode=streaming. 0 means auto."}),
            }
        }

    def load(self, wan_vace_model, text_encoder, vae, lora_name, tokenizer_dir,
             lora_strength, speed_lora, speed_lora_strength, dit_dtype, vram_mode,
             vram_limit_gb):
        from .diffhdr_runtime import runtime  # deferred: pulls in the 88 KB pipeline module

        handle = runtime.load_pipeline(
            wan_vace_model=wan_vace_model, text_encoder=text_encoder, vae=vae,
            tokenizer_dir=tokenizer_dir, lora_name=lora_name,
            lora_strength=lora_strength, speed_lora=speed_lora,
            speed_lora_strength=speed_lora_strength, dit_dtype=dit_dtype,
            vram_mode=vram_mode, vram_limit_gb=vram_limit_gb,
        )
        return (handle,)


class DiffHDREncode:
    """Convert an sRGB plate to DiffHDR's log space and detect the clipped regions."""

    DESCRIPTION = (
        "Prepares the VACE conditioning: converts sRGB to DiffHDR's log curve and "
        "builds the over/under-exposure masks that tell the model which pixels to "
        "treat as missing radiance. The masks are also returned so you can see and "
        "tune what is being reinvented."
    )
    CATEGORY = "DiffHDR"
    RETURN_TYPES = ("DIFFHDR_COND", "MASK", "MASK")
    RETURN_NAMES = ("conditioning", "over_mask", "under_mask")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip":
                    "sRGB LDR frames in [0,1]. One image for a still or a panorama, a "
                    "batch for video."}),
                "mode": (["video", "panorama"], {"default": "video", "tooltip":
                    "video: over- and under-exposure masks, temporally stabilised. Use "
                    "for footage and stills.\n"
                    "panorama: over-exposure only, with the different thresholds and "
                    "morphology the Pano LoRA was trained against. Pair with "
                    "DiffHDR_Pano.safetensors."}),
                "width": ("INT", {"default": TRAINED_WIDTH, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "Working width, multiple of 16. The model was trained at "
                               "1280x720; panoramas at 2048x1024."}),
                "height": ("INT", {"default": TRAINED_HEIGHT, "min": 16, "max": 8192, "step": 16,
                    "tooltip": "Working height, multiple of 16."}),
                "resize_mode": (["crop_and_resize", "stretch", "none"], {"default": "crop_and_resize",
                    "tooltip":
                    "How to reach width x height. crop_and_resize centre-crops to the "
                    "target aspect first (upstream's default). 'none' passes frames "
                    "through untouched -- they must already be multiples of 16."}),
                "over_threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip":
                    "sRGB luma above which a pixel counts as blown. Lower catches more "
                    "highlight, at the risk of inventing detail where real detail "
                    "survives; raise it if bright-but-intact areas are being rebuilt."}),
                "under_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip":
                    "sRGB luma below which a pixel counts as crushed. Only used when "
                    "use_under_exposure_mask is on."}),
                "use_under_exposure_mask": ("BOOLEAN", {"default": False, "tooltip":
                    "Also reconstruct crushed shadows. Off by default, as upstream: it "
                    "adds shadow detail but is the more speculative half of the model, "
                    "and it replaces the conditioning in those regions with mid-grey."}),
                "softness": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Width of the soft band either side of the thresholds."}),
                "temporal_stabilise": ("BOOLEAN", {"default": True, "tooltip":
                    "Smooth the masks across frames (EMA + morphology). Leave on for "
                    "footage: without it the mask boils frame to frame and the model is "
                    "asked to invent a slightly different region each time, which reads "
                    "as flicker in the recovered highlights. No effect on a single frame."}),
            }
        }

    def encode(self, images, mode, width, height, resize_mode, over_threshold,
               under_threshold, use_under_exposure_mask, softness, temporal_stabilise):
        if width % SPATIAL_MULTIPLE or height % SPATIAL_MULTIPLE:
            raise ValueError(
                f"ComfyUI-DiffHDR: width and height must be multiples of {SPATIAL_MULTIPLE} "
                f"(got {width}x{height}). Wan2.1 patches the latent 16 pixels at a time."
            )

        frames = _crop_and_resize(images, height, width, resize_mode)
        if resize_mode == "none" and (frames.shape[1] % SPATIAL_MULTIPLE or frames.shape[2] % SPATIAL_MULTIPLE):
            raise ValueError(
                f"ComfyUI-DiffHDR: resize_mode='none' but the incoming frames are "
                f"{frames.shape[2]}x{frames.shape[1]}, which is not a multiple of "
                f"{SPATIAL_MULTIPLE}."
            )

        srgb = colour.image_to_chw(frames).float()  # (T,3,H,W), sRGB in [0,1]
        t = srgb.shape[0]

        combined, overs, unders = [], [], []
        prev_over = prev_under = None
        conditioning = srgb.clone()

        for i in range(t):
            f = srgb[i]
            if mode == "panorama":
                # Different maths on purpose -- the Pano LoRA was trained with it.
                over = colour.overexposure_mask_pano(f, over_threshold)
                under = torch.zeros_like(over)
            else:
                over, under = colour.exposure_masks_from_srgb(
                    f, over_threshold, under_threshold, softness, clip_eps_srgb=0.02
                )
                if temporal_stabilise:
                    over, prev_over = colour.stabilize_soft_mask(over, prev_over, 0.7, 9, 9)
                    under, prev_under = colour.stabilize_soft_mask(under, prev_under, 0.7, 9, 9)

            over_bin = (over > 0.2).float()
            under_bin = (under > 0.2).float()
            if use_under_exposure_mask and mode != "panorama":
                combined_bin = torch.logical_or(over_bin.bool(), under_bin.bool()).float()
                # Upstream replaces crushed pixels with mid-grey so the model is not
                # anchored to the (meaningless) clipped-black values it would otherwise see.
                conditioning[i][under_bin.unsqueeze(0).repeat(3, 1, 1) == 1.0] = 0.5
            else:
                combined_bin = over_bin

            combined.append(combined_bin)
            overs.append(over_bin)
            unders.append(under_bin)

        coverage = float(torch.stack(combined).mean())
        if coverage < 1e-5:
            logger.warning(
                "ComfyUI-DiffHDR: the exposure mask is empty -- no pixel in this plate "
                "reads as clipped at over_threshold=%.3f. The model will have nothing to "
                "reconstruct and the result will be close to a plain log->linear "
                "conversion. Lower over_threshold if the plate really does have blown "
                "highlights.", over_threshold,
            )
        logger.info("ComfyUI-DiffHDR: mask covers %.2f%% of pixels over %d frame(s)", coverage * 100.0, t)

        # sRGB -> linear -> log. This is the signal the model was trained on.
        log_video = colour.lin_to_log(colour.srgb_to_linear(conditioning))

        cond = {
            "log_video": log_video,                       # (T,3,H,W) float, log space
            "masks": [_mask_to_pil(m) for m in combined],  # what diffsynth's VACE unit wants
            "height": height,
            "width": width,
            "mode": mode,
            "frames": t,
        }
        over_stack = torch.stack(overs)    # (T,H,W) -- ComfyUI MASK layout
        under_stack = torch.stack(unders)
        return (cond, over_stack, under_stack)


class DiffHDRSample:
    """Run the diffusion pass and return scene-linear HDR."""

    DESCRIPTION = (
        "Runs DiffHDR and returns unclamped scene-linear frames. Handles stills, "
        "33-frame clips, longer clips (via overlapping windows) and panoramas from the "
        "same inputs. Wire the output into an EXR writer; the node preview clamps to "
        "[0,1] and will look blown."
    )
    CATEGORY = "DiffHDR"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIFFHDR_PIPELINE",),
                "conditioning": ("DIFFHDR_COND",),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "Upstream runs with an empty prompt and so should you, normally. The "
                    "task is radiance inpainting, not text-to-video; a prompt mostly "
                    "risks steering invented highlight content away from the plate."}),
                "negative_prompt": ("STRING", {"default": NEGATIVE_PROMPT, "multiline": True,
                    "tooltip":
                    "Upstream's default. Unused while cfg_scale is 1.0 (no guidance "
                    "pass) -- it only takes effect if you raise cfg_scale."}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200, "tooltip":
                    "50 reproduces the paper. 10 is upstream's own fast setting and is "
                    "usually indistinguishable for this task, at a fifth of the time."}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip":
                    "1.0 disables classifier-free guidance, which is what upstream uses "
                    "and halves the work. Above 1.0 doubles the cost per step and tends "
                    "to exaggerate the invented highlights."}),
                "seed": ("INT", {"default": 10, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Flow-match schedule shift. 5.0 is upstream's default for Wan2.1."}),
                "window_size": ("INT", {"default": TRAINED_FRAMES, "min": 1, "max": 257, "step": 4,
                    "tooltip":
                    "Frames per diffusion pass; rounded up to 4k+1. 33 is the trained "
                    "length. Clips longer than this are processed as overlapping windows "
                    "and cross-faded."}),
                "window_stride": ("INT", {"default": 16, "min": 1, "max": 256, "tooltip":
                    "Step between windows for long clips. The overlap "
                    "(window_size - stride) is the cross-fade length: wider overlap costs "
                    "more passes but hides the seams better."}),
                "use_prev_window_reference": ("BOOLEAN", {"default": True, "tooltip":
                    "Long clips only. Feed each window a frame from the previous "
                    "window's output as its reference, so successive windows invent "
                    "consistent highlight content instead of each inventing its own.\n\n"
                    "Upstream exposes this as an opt-in flag; it defaults to ON here "
                    "because without it the cross-fade blends two unrelated "
                    "reconstructions and the seam is visible as a shift in the recovered "
                    "highlights. Turn it off to match upstream's default exactly, or if "
                    "you are supplying your own reference_image.\n\n"
                    "Ignored when the clip fits in one window."}),
                "tea_cache": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip":
                    "Skip diffusion steps whose result is predicted to change little "
                    "(TeaCache). 0 disables it. Typical useful range 0.05-0.15; higher "
                    "skips more and degrades more.\n\n"
                    "Two caveats specific to this model. Upstream's coefficient table has "
                    "no Wan2.1-VACE-14B entry, so the T2V-14B curve is used -- same DiT, "
                    "but the calibration is borrowed. And on a skipped step the VACE "
                    "hints are computed and then discarded, which is exactly the signal "
                    "carrying the HDR reconstruction.\n\n"
                    "Measured numbers are in the README. Check peak radiance on your own "
                    "material, not just whether the image looks fine."}),
                "tiled_vae": ("BOOLEAN", {"default": True, "tooltip":
                    "Tile the VAE encode/decode to cut peak VRAM. On by default, as upstream."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip":
                    "Optional style reference for the recovered highlights. Boosted by "
                    "reference_ev stops and converted to log before use, matching upstream."}),
                "reference_ev": ("FLOAT", {"default": 5.0, "min": -10.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Exposure boost applied to the reference image. Upstream default 5.0."}),
            },
        }

    def sample(self, pipeline, conditioning, prompt, negative_prompt, steps, cfg_scale,
               seed, sigma_shift, window_size, window_stride, use_prev_window_reference,
               tea_cache, tiled_vae, reference_image=None, reference_ev=5.0):
        pipe = pipeline.pipe
        device = pipe.device
        height, width = conditioning["height"], conditioning["width"]
        log_video = conditioning["log_video"]
        masks = conditioning["masks"]
        total = conditioning["frames"]

        window_size = _round_frames(window_size)
        if window_stride > window_size:
            raise ValueError(
                f"ComfyUI-DiffHDR: window_stride ({window_stride}) cannot exceed "
                f"window_size ({window_size}) -- that would leave gaps between windows."
            )

        ref = None
        if reference_image is not None:
            ref = self._prepare_reference(reference_image, masks[0], height, width,
                                          reference_ev, device)

        # One frame in: a still (replicated to the trained token count, as upstream
        # does) or a panorama (a genuine single frame).
        if total == 1:
            n = 1 if conditioning["mode"] == "panorama" else _frames_for_still(height, width)
            logger.info("ComfyUI-DiffHDR: single frame, running as %d frame(s)", n)
            out = self._run(pipe, log_video.expand(n, -1, -1, -1), [masks[0]] * n, ref,
                            prompt, negative_prompt, steps, cfg_scale, seed, sigma_shift,
                            height, width, n, tiled_vae, tea_cache, label="still")
            return (self._to_linear(out[:1]),)

        windows = plan(total, window_size, window_stride)
        if len(windows) == 1:
            n = _round_frames(total)
            frames, mask_list = self._pad(log_video, masks, n)
            out = self._run(pipe, frames, mask_list, ref, prompt, negative_prompt, steps,
                            cfg_scale, seed, sigma_shift, height, width, n, tiled_vae,
                            tea_cache, label="clip")
            return (self._to_linear(out[:total]),)

        logger.info(
            "ComfyUI-DiffHDR: %d frames > window of %d -- %d overlapping windows "
            "(stride %d, %d-frame cross-fade)",
            total, window_size, len(windows), window_stride, window_size - window_stride,
        )
        acc = Accumulator(total, window_size, window_stride)
        window_ref = ref
        for i, w in enumerate(windows):
            comfy.model_management.throw_exception_if_processing_interrupted()
            chunk = log_video[w["start"]:w["end"]]
            chunk_masks = masks[w["start"]:w["end"]]
            if w["pad"]:
                chunk = torch.cat([chunk, chunk[-1:].repeat(w["pad"], 1, 1, 1)])
                chunk_masks = chunk_masks + [chunk_masks[-1]] * w["pad"]
            out_log = self._run(pipe, chunk, chunk_masks, window_ref, prompt,
                                negative_prompt, steps, cfg_scale, seed, sigma_shift,
                                height, width, window_size, tiled_vae, tea_cache,
                                label=f"window {i + 1}/{len(windows)}")
            acc.add(w, self._to_linear(out_log))

            if use_prev_window_reference:
                # Upstream carries a frame of this window's output forward as the next
                # window's reference, in LOG space -- the same space a reference is
                # consumed in, so no conversion happens. Index window_stride is the
                # frame that lines up with the start of the next window.
                real = w["real"]
                ref_idx = window_stride if real > window_stride else real - 1
                window_ref = out_log[ref_idx].permute(2, 0, 1).contiguous().to(pipe.device)
        return (acc.result(),)

    # -- internals ---------------------------------------------------------- #

    def _pad(self, log_video, masks, target):
        """Repeat the last frame up to ``target``; the model degrades below its window."""
        t = log_video.shape[0]
        if t >= target:
            return log_video[:target], masks[:target]
        pad = target - t
        return (
            torch.cat([log_video, log_video[-1:].repeat(pad, 1, 1, 1)]),
            masks + [masks[-1]] * pad,
        )

    def _prepare_reference(self, reference_image, first_mask, height, width, ev, device):
        """Match upstream's reference handling: mask out, EV-boost, then to log."""
        ref = _crop_and_resize(reference_image[:1], height, width, "crop_and_resize")
        ref = colour.image_to_chw(ref)[0].to(device).float()  # (3,H,W) sRGB

        m = torch.from_numpy(np.array(first_mask)).to(device).float() / 255.0
        if m.ndim == 3:
            m = m[..., 0]
        # Upstream pushes unmasked areas to white so the reference only speaks about
        # the regions actually being reconstructed.
        ref = ref * m.unsqueeze(0) + (1.0 - m.unsqueeze(0))

        ref = colour.srgb_to_linear(ref.clamp(0.0, 1.0)).clamp(min=0.0)
        ref = ref * (2.0 ** float(ev))
        return colour.lin_to_log(ref)

    def _run(self, pipe, log_frames, mask_list, ref, prompt, negative_prompt, steps,
             cfg_scale, seed, sigma_shift, height, width, num_frames, tiled_vae,
             tea_cache=0.0, label=""):
        """One diffusion pass. Returns (T,H,W,C) float in LOG space on the pipeline device."""
        comfy.model_management.throw_exception_if_processing_interrupted()
        pbar = comfy.utils.ProgressBar(steps)

        def progress(iterable):
            """Drive ComfyUI's progress bar from the sampler loop, and allow cancelling.

            The pipeline takes a ``progress_bar_cmd`` hook, so this reports real
            per-step progress instead of a heartbeat, and a Cancel in the UI takes
            effect within one step rather than at the end of the job.
            """
            for i, item in enumerate(iterable):
                comfy.model_management.throw_exception_if_processing_interrupted()
                pbar.update_absolute(i, steps)
                yield item
            pbar.update_absolute(steps, steps)

        t0 = time.perf_counter()
        logger.info("ComfyUI-DiffHDR: sampling %s -- %d frames at %dx%d, %d steps",
                    label, num_frames, width, height, steps)

        with torch.inference_mode():
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=height, width=width, num_frames=num_frames,
                cfg_scale=float(cfg_scale),
                num_inference_steps=int(steps),
                sigma_shift=float(sigma_shift),
                tiled=bool(tiled_vae),
                vace_video=log_frames.to(pipe.device),
                vace_video_mask=mask_list,
                vace_reference_image=ref,
                # The two flags that make this HDR rather than ordinary Wan VACE:
                # srgb_to_lg tells the VACE unit its conditioning is already a float
                # log signal, is_hdr stops the decoder quantising the result to 8-bit.
                srgb_to_lg=True,
                is_hdr=True,
                # Upstream's TeaCache table has no VACE-14B entry; T2V-14B is the same
                # DiT and the nearest calibration available.
                tea_cache_l1_thresh=(float(tea_cache) if tea_cache > 0 else None),
                tea_cache_model_id="Wan2.1-T2V-14B",
                progress_bar_cmd=progress,
            )

        out = torch.stack(video)                 # (T,H,W,C), log space, unquantised
        out = torch.clamp_min(out, 0.0).float()  # upstream clamps the floor only
        logger.info("ComfyUI-DiffHDR: %s done in %.1f s", label, time.perf_counter() - t0)
        return out

    @staticmethod
    def _to_linear(out_log: torch.Tensor) -> torch.Tensor:
        """Log -> scene-linear, on CPU. Kept separate from _run so the window loop can
        pass the *log* frames forward as the next window's reference, which is the space
        a reference is consumed in."""
        out = colour.log_to_lin(out_log).float().cpu()
        logger.info("ComfyUI-DiffHDR: linear range [%.4f, %.1f]",
                    float(out.min()), float(out.max()))
        return out


class DiffHDRFreeMemory:
    """Drop the cached pipeline to reclaim VRAM."""

    DESCRIPTION = (
        "Releases the cached DiffHDR pipeline. Only needed when you want the VRAM back "
        "without restarting ComfyUI -- the loader already keeps at most one resident."
    )
    CATEGORY = "DiffHDR"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "free"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("BOOLEAN", {"default": True})}}

    @classmethod
    def IS_CHANGED(cls, trigger):
        return float("nan")  # always run: the point is the side effect

    def free(self, trigger):
        if not trigger:
            return ("skipped",)
        from .diffhdr_runtime import runtime

        runtime.free()
        return ("DiffHDR pipeline released",)


NODE_CLASS_MAPPINGS = {
    "DiffHDR_LoadModel": DiffHDRLoadModel,
    "DiffHDR_Encode": DiffHDREncode,
    "DiffHDR_Sample": DiffHDRSample,
    "DiffHDR_FreeMemory": DiffHDRFreeMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffHDR_LoadModel": "DiffHDR Load Model",
    "DiffHDR_Encode": "DiffHDR Encode (LDR -> log + masks)",
    "DiffHDR_Sample": "DiffHDR Sample",
    "DiffHDR_FreeMemory": "DiffHDR Free Memory",
}
