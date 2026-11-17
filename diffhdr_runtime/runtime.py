"""Build, cache and sanity-check the DiffHDR pipeline.

One pipeline is ~34 GB of Wan2.1-VACE-14B at bf16 (~17 GB cast to fp8), so the rules
here are: never hold two, never rebuild when nothing changed, and never hand back a
pipeline that is quietly missing the LoRA.

That last one matters more than it sounds. Upstream merges the LoRA straight into the
VACE weights (``GeneralLoRALoader.load`` does ``weight += alpha * B@A``) and reports
the count on stdout; if zero modules match, nothing raises. You get plain
Wan2.1-VACE, which produces a perfectly plausible video that is not HDR. So the
merge is driven from here where the match count can be checked, rather than through
``pipe.load_lora``.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict

import torch

from . import model_scan, vendor

logger = logging.getLogger("ComfyUI-DiffHDR")

# One resident pipeline, ever. Evicted before the replacement is built so the two
# never overlap in VRAM. Values are ``(pipe, lora_merged, speed_merged)``; nothing else
# in the process holds a reference to the pipe, so eviction really frees it.
_CACHE_LOCK = threading.Lock()
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX = 1

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
VRAM_MODES = ("auto", "full_gpu", "streaming")


class DiffHDRPipeline:
    """Opaque handle passed down the graph on the DIFFHDR_PIPELINE wire.

    The handle holds the *recipe* for a pipeline -- resolved paths and settings -- and
    never the pipeline itself; ``pipe`` looks it up in the module cache and rebuilds it
    if it is gone. Both halves of that matter, because ComfyUI caches node outputs:

    * ComfyUI keeps its own reference to this handle, so if the handle held the
      pipeline, evicting it from our cache would free no memory at all;
    * ComfyUI will not re-run the Load Model node while its widgets are unchanged, so
      a handle that could only report "released" (after DiffHDR Free Memory, or after
      a second loader evicted it) would fail every run until the user touched a
      loader widget. Rebuilding on demand is what ComfyUI's own model management does
      for unloaded models, too.
    """

    def __init__(self, key, build_args: dict):
        self.key = key
        self._build_args = build_args
        self.lora_name = os.path.basename(build_args["lora_path"])
        speed_path = build_args.get("speed_path")
        self.speed_lora = os.path.basename(speed_path) if speed_path else None
        self.lora_modules_merged = 0
        self.speed_modules_merged = 0
        self._built = False

    @property
    def pipe(self):
        return _acquire(self)

    @property
    def released(self) -> bool:
        """True if the next ``pipe`` access will have to rebuild."""
        return self.key not in _CACHE

    def __repr__(self):
        state = "released" if self.released else f"merged={self.lora_modules_merged}"
        extra = f" speed={self.speed_lora!r}/{self.speed_modules_merged}" if self.speed_lora else ""
        return f"<DiffHDRPipeline lora={self.lora_name!r} {state}{extra}>"


def _read_lora(path: str) -> dict:
    from safetensors.torch import load_file

    if path.lower().endswith(".safetensors"):
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _lora_plan(lora: dict) -> dict[str, dict]:
    """Map ``module path -> what to apply to it`` for a Wan LoRA.

    Handles the three encodings actually found in the wild, because the two LoRAs this
    pack cares about use different ones and diffsynth's own loader only understands the
    first:

    * ``lora_A``/``lora_B`` (PEFT, optionally with a ``.default`` adapter segment) --
      DiffHDR's own LoRA.
    * ``lora_down``/``lora_up`` (+ optional ``.alpha``) -- the kohya/Civitai convention,
      which is what the lightx2v step-distill LoRAs use.
    * ``diff`` / ``diff_b`` -- plain additive deltas for weights and biases. These are
      not a LoRA at all, but lightx2v ships 647 of them alongside its 406 low-rank
      pairs, and dropping them would apply the distillation only partially.

    A leading ``diffusion_model.`` is stripped so the same file works whether it is
    merged into the whole DiT or a submodule.
    """
    plan: dict[str, dict] = {}

    def target(key: str, suffix: str) -> str:
        base = key[: -len(suffix)]
        return base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base

    for key in lora:
        if key.endswith(".lora_B.weight") or key.endswith(".lora_B.default.weight"):
            suffix = ".lora_B.default.weight" if ".default." in key else ".lora_B.weight"
            plan.setdefault(target(key, suffix), {})["ab"] = (key, key.replace(".lora_B.", ".lora_A."))
        elif key.endswith(".lora_up.weight"):
            plan.setdefault(target(key, ".lora_up.weight"), {})["ab"] = (
                key, key.replace(".lora_up.", ".lora_down."))
        elif key.endswith(".diff_b"):
            plan.setdefault(target(key, ".diff_b"), {})["diff_b"] = key
        elif key.endswith(".diff"):
            plan.setdefault(target(key, ".diff"), {})["diff"] = key
        elif key.endswith(".alpha"):
            plan.setdefault(target(key, ".alpha"), {})["alpha"] = key
    return plan


def _merge_lora(module, lora_path: str, alpha: float, what: str) -> int:
    """Merge a LoRA into ``module`` in place; return how many modules were updated.

    Merging (rather than patching) is what upstream does, so the arithmetic matches:
    fp32 ``up @ down``, added into the parameter's own dtype. Doing it here rather than
    through ``pipe.load_lora`` is what makes the match count observable -- upstream
    prints it and carries on, which means a LoRA that matched nothing looks like success.
    """
    _t0 = time.perf_counter()
    lora = _read_lora(lora_path)
    plan = _lora_plan(lora)

    # The weights sit on CPU at this point (they are loaded with offload_device="cpu"),
    # so a naive merge does every `up @ dn` as an fp32 CPU matmul. That is fine for
    # DiffHDR's 80 modules and slow for a step-distill LoRA's 600+, where the products
    # are 5120x5120 and larger. Do the product on the GPU when there is one and copy the
    # (small) result back; the weight itself never moves.
    mm_device = torch.device("cuda") if torch.cuda.is_available() else None

    if not plan:
        raise RuntimeError(
            f"ComfyUI-DiffHDR: {os.path.basename(lora_path)} contains no recognisable "
            "LoRA tensors (looked for lora_A/lora_B, lora_down/lora_up, diff/diff_b)."
        )

    matched = 0
    seen: set[str] = set()
    for name, mod in module.named_modules():
        entry = plan.get(name)
        if entry is None:
            continue
        sd = mod.state_dict()
        w = sd.get("weight")
        touched = False

        if "ab" in entry and w is not None:
            up_key, down_key = entry["ab"]
            if down_key in lora:
                dev = mm_device or w.device
                up = lora[up_key].to(device=dev, dtype=torch.float32)
                dn = lora[down_key].to(device=dev, dtype=torch.float32)
                scale = alpha
                if "alpha" in entry:
                    # kohya convention: the stored alpha rescales by alpha/rank.
                    scale = alpha * float(lora[entry["alpha"]]) / dn.shape[0]
                if up.ndim == 4:
                    up, dn = up.squeeze(3).squeeze(2), dn.squeeze(3).squeeze(2)
                    delta = scale * torch.mm(up, dn).unsqueeze(2).unsqueeze(3)
                else:
                    delta = scale * torch.mm(up, dn)
                delta = delta.to(device=w.device)
                sd["weight"] = (w.to(torch.float32) + delta.reshape(w.shape)).to(dtype=w.dtype)
                touched = True

        if "diff" in entry and w is not None:
            d = lora[entry["diff"]].to(device=w.device, dtype=torch.float32)
            base = sd["weight"] if touched else w
            sd["weight"] = (base.to(torch.float32) + alpha * d.reshape(w.shape)).to(dtype=w.dtype)
            touched = True

        if "diff_b" in entry and sd.get("bias") is not None:
            b = sd["bias"]
            d = lora[entry["diff_b"]].to(device=b.device, dtype=torch.float32)
            sd["bias"] = (b.to(torch.float32) + alpha * d.reshape(b.shape)).to(dtype=b.dtype)
            touched = True

        if touched:
            mod.load_state_dict(sd)
            seen.add(name)
            matched += 1

    expected = len(plan)
    if matched == 0:
        raise RuntimeError(
            f"ComfyUI-DiffHDR: {what} LoRA {os.path.basename(lora_path)} matched NONE of "
            f"the {expected} modules it targets. A LoRA that merges into nothing leaves "
            "you with the plain base model, which still produces plausible-looking "
            "output -- so this is a hard error rather than a warning. Check the file is "
            f"the right kind of LoRA for the {what} slot."
        )
    if matched < expected:
        logger.warning(
            "ComfyUI-DiffHDR: %s LoRA merged into %d of %d target modules -- the "
            "checkpoint and the LoRA may not be the same Wan variant.",
            what, matched, expected,
        )
    logger.info("ComfyUI-DiffHDR: merged %s LoRA into %d/%d modules in %.1f s",
                what, matched, expected, time.perf_counter() - _t0)
    return matched


def _install_prompt_cache(pipe) -> None:
    """Memoise the T5 prompt embedding across runs.

    DiffHDR is run with a constant prompt -- empty, by upstream's design, since the task
    is radiance inpainting rather than text-to-video -- yet the pipeline re-encodes it on
    every single job. Worse, the prompt unit calls ``load_models_to_device(["text_encoder"])``
    *before* doing anything else, so an 11 GB encoder is swapped onto the GPU to produce
    a tensor we already had.

    Wrapping the unit (rather than just ``prompter.encode_prompt``) is what lets the
    cache short-circuit the model swap as well as the forward pass. Embeddings are a few
    MB each, and are keyed on everything that can change the result.
    """
    for unit in getattr(pipe, "units", []):
        if type(unit).__name__ != "WanVideoUnit_PromptEmbedder":
            continue
        if getattr(unit, "_diffhdr_cached", False):
            return
        real = unit.process
        cache: dict = {}

        def process(pipe_, prompt, positive, _real=real, _cache=cache):
            try:
                key = (repr(prompt), bool(positive), str(pipe_.device), str(pipe_.torch_dtype))
            except Exception:  # noqa: BLE001 - never let caching break a run
                return _real(pipe_, prompt, positive)
            hit = _cache.get(key)
            if hit is None:
                hit = _real(pipe_, prompt, positive)
                # Bound it: a user editing the prompt repeatedly should not accumulate
                # embeddings forever.
                if len(_cache) >= 8:
                    _cache.pop(next(iter(_cache)))
                _cache[key] = hit
            else:
                logger.debug("ComfyUI-DiffHDR: prompt embedding cache hit")
            return hit

        unit.process = process
        unit._diffhdr_cached = True
        logger.info("ComfyUI-DiffHDR: prompt-embedding cache installed")
        return


def _evict_all():
    """Drop the cached pipeline and give the memory back. Caller holds _CACHE_LOCK."""
    while _CACHE:
        key, old = _CACHE.popitem(last=False)
        logger.info("ComfyUI-DiffHDR: evicting cached pipeline %s", key[0] if key else "")
        del old
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _acquire(handle: DiffHDRPipeline):
    """Return the pipeline for ``handle``, building it (and evicting any other) if needed."""
    with _CACHE_LOCK:
        entry = _CACHE.get(handle.key)
        if entry is None:
            if handle._built:
                logger.info(
                    "ComfyUI-DiffHDR: pipeline was released (DiffHDR Free Memory ran, or "
                    "another loader took its place) -- rebuilding it"
                )
            if len(_CACHE) >= _CACHE_MAX:
                # Evict first: building the replacement while the old one is resident
                # would need twice the memory, which is exactly what this cache exists
                # to avoid.
                _evict_all()
            entry = _build(**handle._build_args)
            _CACHE[handle.key] = entry
        else:
            _CACHE.move_to_end(handle.key)
        pipe, handle.lora_modules_merged, handle.speed_modules_merged = entry
        handle._built = True
        return pipe


def load_pipeline(
    *,
    wan_vace_model: str,
    text_encoder: str,
    vae: str,
    tokenizer_dir: str,
    lora_name: str,
    lora_strength: float,
    speed_lora: str,
    speed_lora_strength: float,
    dit_dtype: str,
    vram_mode: str,
    vram_limit_gb: float,
) -> DiffHDRPipeline:
    """Resolve, validate, build and cache the pipeline for one set of loader widgets."""
    dit_path = model_scan.resolve("dit", wan_vace_model)
    t5_path = model_scan.resolve("text_encoder", text_encoder)
    vae_path = model_scan.resolve("vae", vae)
    lora_path = model_scan.resolve("lora", lora_name)
    tok_path = model_scan.resolve_tokenizer(tokenizer_dir)
    speed_path = None if speed_lora in (None, "", model_scan.NONE) else model_scan.resolve("lora", speed_lora)

    # Header-only pre-flight, before anything large is read. Milliseconds, and it turns
    # a five-minute load that dies obscurely into an immediate, specific error.
    model_scan.validate(dit_path, "wan_video_vace", "wan_vace_model")
    model_scan.validate(t5_path, "wan_video_text_encoder", "text_encoder")
    model_scan.validate(vae_path, "wan_video_vae", "vae")

    if dit_dtype not in DTYPES:
        raise ValueError(f"ComfyUI-DiffHDR: unknown dit_dtype {dit_dtype!r}")
    if vram_mode not in VRAM_MODES:
        raise ValueError(f"ComfyUI-DiffHDR: unknown vram_mode {vram_mode!r}")

    key = (
        os.path.realpath(dit_path), os.path.realpath(t5_path), os.path.realpath(vae_path),
        os.path.realpath(tok_path), os.path.realpath(lora_path),
        round(float(lora_strength), 6), dit_dtype, vram_mode, round(float(vram_limit_gb), 3),
        os.path.realpath(speed_path) if speed_path else None,
        round(float(speed_lora_strength), 6) if speed_path else None,
    )
    handle = DiffHDRPipeline(key, dict(
        dit_path=dit_path, t5_path=t5_path, vae_path=vae_path, tok_path=tok_path,
        lora_path=lora_path, lora_strength=float(lora_strength), dit_dtype=dit_dtype,
        vram_mode=vram_mode, vram_limit_gb=float(vram_limit_gb),
        speed_path=speed_path, speed_strength=float(speed_lora_strength),
    ))

    if not handle.released:
        logger.info("ComfyUI-DiffHDR: reusing cached pipeline")
    # Build now rather than on first use, so the load cost lands on the Load Model
    # node where the user expects it, and a bad file fails there too.
    _acquire(handle)
    return handle


def _build(dit_path, t5_path, vae_path, tok_path, lora_path,
           lora_strength, dit_dtype, vram_mode, vram_limit_gb,
           speed_path=None, speed_strength=1.0):
    """Build one pipeline. Returns ``(pipe, lora_modules_merged, speed_modules_merged)``."""
    WanVideoPipeline, ModelConfig = vendor.pipeline_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16
    offload_dtype = DTYPES[dit_dtype]

    t0 = time.perf_counter()
    logger.info(
        "ComfyUI-DiffHDR: building pipeline (dit=%s dtype=%s vram_mode=%s)",
        os.path.basename(dit_path), dit_dtype, vram_mode,
    )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=compute_dtype,
        # The VAE stays fp32: it is small, it runs once per job, and it is what carries
        # the log signal back out. Halving it costs precision exactly where the
        # highlight recovery lives.
        vae_dtype=torch.float32,
        device=device,
        model_configs=[
            # offload_dtype is the *load* dtype (wan_video_new.py:552-556), which is how
            # the fp8 path works: the same fp16 file is cast on the way in. The
            # detection hash covers keys and shapes but not dtype, so it still validates.
            ModelConfig(path=dit_path, offload_device="cpu", offload_dtype=offload_dtype),
            ModelConfig(path=t5_path, offload_device="cpu"),
            ModelConfig(path=vae_path, offload_device="cpu", offload_dtype=torch.float32),
        ],
        tokenizer_config=ModelConfig(path=tok_path),
        # We always pass explicit local paths, so upstream's download/redirect logic
        # must stay out of the way (it would otherwise rewrite them to hub ids).
        redirect_common_files=False,
    )

    if getattr(pipe, "vace", None) is None:
        raise RuntimeError(
            "ComfyUI-DiffHDR: the checkpoint loaded but exposes no VACE module. "
            f"{os.path.basename(dit_path)} passed the header check yet produced a "
            "pipeline without vace_blocks -- this is the merged-checkpoint requirement; "
            "see the model table in the README."
        )

    matched = _merge_lora(pipe.vace, lora_path, lora_strength, "DiffHDR")

    # Step-distill LoRAs (lightx2v and friends) patch the DiT's own blocks.*, while
    # DiffHDR patches vace_blocks.* -- verified zero overlap, so the two compose rather
    # than fight. Merged into pipe.dit for the same reason DiffHDR goes into pipe.vace.
    speed_matched = 0
    if speed_path:
        speed_matched = _merge_lora(pipe.dit, speed_path, speed_strength, "speed")

    # Order is load-bearing: the LoRA merge writes into the raw module weights, and
    # enable_vram_management replaces those modules with offload-aware wrappers.
    # Merging afterwards would write into wrappers that may be parked on CPU.
    _install_prompt_cache(pipe)

    if vram_mode == "full_gpu":
        # Keep everything resident: fastest, and needs the whole DiT in VRAM.
        pipe.enable_vram_management(num_persistent_param_in_dit=10**12, vae_dtype=torch.float32)
    elif vram_mode == "streaming" and vram_limit_gb > 0:
        pipe.enable_vram_management(vram_limit=vram_limit_gb, vae_dtype=torch.float32)
    else:
        # Upstream's own default: budget from total VRAM and stream the DiT from CPU.
        pipe.enable_vram_management(vae_dtype=torch.float32)

    logger.info(
        "ComfyUI-DiffHDR: pipeline ready in %.1f s (DiffHDR LoRA -> %d modules%s)",
        time.perf_counter() - t0, matched,
        f", speed LoRA -> {speed_matched}" if speed_path else "",
    )
    return pipe, matched, speed_matched


def free():
    """Drop the cached pipeline (DiffHDR Free Memory). Handles rebuild it on next use."""
    with _CACHE_LOCK:
        _evict_all()
