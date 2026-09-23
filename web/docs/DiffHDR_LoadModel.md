# DiffHDR Load Model

Loads Wan2.1-VACE-14B, the umt5 text encoder, the Wan VAE and a DiffHDR LoRA into one
cached pipeline.

DiffHDR is **stock Wan2.1-VACE-14B plus a 58 MB LoRA** — so if you already run Wan in
ComfyUI, you almost certainly have the big files already. Nothing is downloaded
automatically.

## Choosing a checkpoint

Several plausibly-named Wan files do **not** work. The node reads each file's
safetensors header before loading anything and tells you which of these you hit, in
about a second, rather than failing minutes into a 34 GB read.

| Slot | Use | Not this |
|---|---|---|
| `wan_vace_model` | the **merged** Wan2.1-VACE-14B (~1331 tensors), e.g. `wan2.1_vace_14B_fp16.safetensors` | the split **VACE-module** files (~237 tensors) published for ComfyUI-WanVideoWrapper; plain T2V/I2V; the 1.3B model |
| `text_encoder` | the **encoder-only** `umt5-xxl-enc-bf16.safetensors` | `umt5_xxl_fp16.safetensors` — different key set |
| `vae` | `wan_2.1_vae.safetensors` or Kijai's `Wan2_1_VAE_bf16.safetensors` | — |
| `lora_name` | `DiffHDR.safetensors`, or `DiffHDR_Pano.safetensors` for panoramas | any other Wan LoRA |

The VACE-module case is the common mistake: those files hold only the VACE blocks and
expect a separate base model to be paired with them at runtime, which is a
WanVideoWrapper feature. The pipeline here needs the single merged checkpoint.

## Fitting it on your card

Two independent controls:

**`dit_dtype`** — the precision the DiT is loaded at.

* `bf16` — reference behaviour, ~34 GB.
* `fp8_e4m3fn` — roughly halves that to ~17 GB by casting on load. The usual choice on
  a 24–32 GB card.

This casts the ordinary fp16 file. Pre-quantised `*_fp8_scaled` repacks carry extra
per-tensor scale tensors that change the key set, so they cannot be identified and are
rejected — use the plain file and set this widget instead.

**`vram_mode`** — where the weights live during sampling.

* `auto` — upstream's default. Budgets from total VRAM and streams DiT blocks from
  system RAM. Works nearly everywhere.
* `full_gpu` — keep everything resident. Fastest; needs the whole DiT in VRAM.
* `streaming` — as `auto` but with an explicit budget from `vram_limit_gb`, for when
  you are sharing the card.

The VAE always runs in fp32 regardless. It is small, it runs once per job, and it is
what carries the log signal back out — halving it would cost precision exactly where
the highlight reconstruction lives.

## Speed LoRA (optional)

`speed_lora` takes a step-distillation LoRA for the DiT — the lightx2v
`cfg_step_distill` family and similar. The point is steps: roughly 10 down to 4.

This stacks rather than conflicts, and that is a structural fact rather than a hope:
DiffHDR patches `vace_blocks.*` (80 modules) and lightx2v patches `blocks.*` (607
modules), with **zero overlap**. They are merged into different submodules of the
pipeline (`pipe.vace` and `pipe.dit` respectively).

Two things to know:

* Only the *step* half of a "cfg_step" distill helps here. DiffHDR already runs at
  `cfg_scale` 1.0, so the CFG half is buying something you were not paying for.
* diffsynth's own LoRA loader cannot read these files at all — it keys off `.lora_B.`,
  and lightx2v ships `lora_up`/`lora_down` plus 647 `diff`/`diff_b` tensors. This pack
  handles all three encodings; if it ever matches nothing it raises rather than
  continuing quietly.

### Measured, and read the last column

5 frames at 832×480 on an A5000, upstream's demo still, fp8:

| | time | speedup | peak radiance | log PSNR vs baseline |
|---|---|---|---|---|
| baseline, 10 steps | 147 s | 1.0× | **19.9** | — |
| lightx2v, 6 steps | 72 s | 2.0× | 10.2 | 45.4 dB |
| lightx2v, 4 steps | 61 s | 2.4× | 8.5 | 40.2 dB |

The PSNR looks reassuring and is misleading. It is computed in log space, and this log
curve spends very few code values on the top of its range — one photographic stop is
only ~0.05 of log code up there. So a log-space mean error of 0.0064 is, at the
highlight end, **1.2 stops of lost peak radiance**.

That is not a detail for this tool. Recovering the top end is the entire reason DiffHDR
exists; a 2.4× speedup that halves your peak is a bad trade for a finished HDR plate,
and a fine one for a look-dev turnaround. Judge it on peak radiance and on your own
material, not on PSNR.

Which is why this widget defaults to `none`. Lower `steps` on the Sample node when you
do enable it — leaving it at 10 wastes most of the benefit.

## Caching

One pipeline is held at a time, and it is released *before* a replacement is built, so
two 14 B models are never resident at once. Changing any widget rebuilds it. That
includes `lora_strength`, because the LoRA is merged into the weights rather than
applied as a patch.

Two Load Model nodes in one graph (say, the video and the Pano LoRA) therefore take
turns: each Sample rebuilds its own pipeline if the other loader has taken the slot.
That is correct but costs a full load each time, so prefer one loader per graph. The
same automatic rebuild happens after DiffHDR Free Memory.

## If the LoRA fails to attach

This is a hard error, not a warning, and deliberately so: without the LoRA you have
plain Wan2.1-VACE, which produces a perfectly plausible video that is not an HDR
reconstruction. That is the kind of failure you would ship without noticing.
