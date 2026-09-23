# ComfyUI-DiffHDR

ComfyUI nodes for [DiffHDR](https://github.com/Eyeline-Labs/DiffHDR) (Eyeline Labs) —
reconstructing scene-linear HDR from 8-bit LDR video and stills.

DiffHDR treats LDR→HDR as *generative radiance inpainting*. The plate is converted to a
log curve and handed to Wan2.1-VACE-14B as control video; the blown and crushed regions
are marked as the VACE inpaint mask; the model synthesises plausible radiance there; the
result comes back to linear without a display clamp. A small LoRA is what teaches it the
task.

Output is a plain unclamped linear `IMAGE`. This pack writes no files, so it composes
with whatever EXR writer you already use.

```
  Load Image / Video ──▶ DiffHDR Encode ──▶ DiffHDR Sample ──▶ linear HDR IMAGE
                                │                  ▲                  │
                                ▼                  │                  ▼
                       over / under MASK   DiffHDR Load Model    your EXR writer
```

**Requires:** a CUDA GPU, Wan2.1-VACE-14B (which you may already have), and a 58 MB
LoRA. Tested on a 24 GB RTX A5000; see [what fits](#what-fits-it-is-the-token-count-not-the-resolution)
before planning a job.

---

## The short version

DiffHDR is **stock Wan2.1-VACE-14B plus a 58 MB LoRA.** If you already run Wan in
ComfyUI, the only thing you need to download is the LoRA. The upstream README's 75 GB
figure is for a fresh checkout in diffsynth's own layout; it is not required here.

---

## Nodes

| Display name | Category | What it does |
|---|---|---|
| **DiffHDR Load Model** | `DiffHDR` | Loads the Wan2.1-VACE-14B backbone, umt5 encoder, VAE and a DiffHDR LoRA into one cached pipeline. Validates every checkpoint from its safetensors header *before* loading, because several plausibly-named Wan files do not work and the resulting failure is otherwise obscure and minutes deep. Handles precision (`bf16`/`fp8`) and VRAM strategy. Errors — rather than warns — if the LoRA merges into nothing, since that silently yields ordinary Wan video. |
| **DiffHDR Encode** | `DiffHDR` | Conforms the plate, detects over/under-exposed regions, and converts sRGB → linear → DiffHDR's log curve. A separate node because the masks are the real creative control: they decide what gets reinvented. They come out as `MASK` so you can see them, and an optional `mask` input lets you add to, subtract from or replace the detection. |
| **DiffHDR Sample** | `DiffHDR` | Runs the diffusion pass and returns unclamped scene-linear frames. Covers all four upstream entry points — stills, 33-frame clips, long clips via overlapping windows, and HDRI panoramas — choosing between them from what you feed it rather than from a mode widget. Real per-step progress and working cancellation. |
| **DiffHDR Free Memory** | `DiffHDR` | Releases the cached pipeline. Rarely needed; the loader already holds at most one. Wire Sample's output through it so it runs after sampling; the next run reloads automatically. |

Each node has a help page in the ComfyUI node-help panel (`web/docs/`).

---

## Installing

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-DiffHDR
pip install -r ComfyUI-DiffHDR/requirements.txt
```

Do **not** install upstream DiffHDR's `requirements.txt` — it pins python 3.10,
torch 2.6.0 and `numpy==1.26.4`, and most of it belongs to training and evaluation code
this pack does not vendor.

The upstream pipeline is vendored under `DiffHDR/` (see `DiffHDR/VENDORED.md`), so there
is no runtime clone step.

### Weights

Nothing is downloaded automatically. Put files anywhere ComfyUI already looks — the
node scans `diffusion_models`, `text_encoders`, `vae`, `loras` and `unet`, plus a
`models/diffhdr/` folder it creates for convenience.

| Role | Get | Notes |
|---|---|---|
| **LoRA** | [`ZhengmingYu/DiffHDR`](https://huggingface.co/ZhengmingYu/DiffHDR) → `DiffHDR.safetensors`, `DiffHDR_Pano.safetensors` | 58 MB each. The only mandatory download. |
| **Wan2.1-VACE-14B** | [`Wan-AI/Wan2.1-VACE-14B`](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B), or the repack [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) → `split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors` | Must be the **merged** checkpoint (~1331 tensors). |
| **Text encoder** | [`Comfy-Org/Wan_2.1_ComfyUI_repackaged`](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) → `umt5-xxl-enc-bf16.safetensors` | Must be the **encoder-only** build. |
| **VAE** | Same repack → `wan_2.1_vae.safetensors`, or [`Kijai/WanVideo_comfy`](https://huggingface.co/Kijai/WanVideo_comfy) → `Wan2_1_VAE_bf16.safetensors` | Either works. |
| **Tokenizer** | ships inside the Wan repos as `google/umt5-xxl/` (~5 MB) | Auto-detected. `hf download Wan-AI/Wan2.1-VACE-14B --include 'google/umt5-xxl/*'` if you lack it. |

#### Files that look right but are not

These are all **rejected by the header pre-flight**, in about a second, with a message
naming the problem. Listed because each is an easy mistake:

| File | Why not |
|---|---|
| `Wan2_1-VACE_module_14B_*.safetensors` (Kijai) | The **split VACE-module** release (~237 tensors), built for ComfyUI-WanVideoWrapper, which pairs it with a separate base at runtime. diffsynth needs the merged checkpoint. |
| `umt5_xxl_fp16.safetensors` | Not the encoder-only build; different key set. |
| `wan2.1_vace_1.3B_*` | 1.3B is not in upstream's detection table and DiffHDR was not trained against it. |
| `*_fp8_scaled.safetensors` | Carries extra per-tensor scale tensors, which change the key set. Use the plain fp16 file with `dit_dtype=fp8_e4m3fn` instead — same VRAM saving, via a path diffsynth understands. |
| GGUF quantisations | Not supported by diffsynth. |

### Going faster: step-distillation LoRAs

`speed_lora` on the loader takes a lightx2v-style `cfg_step_distill` LoRA. It stacks
cleanly with DiffHDR's — structurally, not by luck: DiffHDR patches `vace_blocks.*` (80
modules), lightx2v patches `blocks.*` (607), **zero overlap**, and they are merged into
different submodules.

(diffsynth's own loader reads none of these files — it keys off `.lora_B.`, while
lightx2v ships `lora_up`/`lora_down` plus 647 `diff`/`diff_b` tensors. This pack handles
all three encodings, and raises rather than continuing quietly if a LoRA matches
nothing.)

Measured, 5 frames at 832×480 on an A5000, upstream's demo still, fp8:

| | time | speedup | peak radiance | log PSNR |
|---|---|---|---|---|
| baseline, 10 steps | 147 s | 1.0× | **19.9** | — |
| lightx2v, 6 steps | 72 s | 2.0× | 10.2 | 45.4 dB |
| lightx2v, 4 steps | 61 s | 2.4× | 8.5 | 40.2 dB |

**Read the peak-radiance column, not the PSNR.** The PSNR is computed in log space, and
this curve spends very few code values at the top — one stop is ~0.05 of log code there.
So a log-space mean error of 0.0064 amounts to **1.2 stops of lost peak** at the
highlight end, which is precisely the range the model exists to reconstruct.

So: excellent for look-dev turnaround, questionable for a finished plate. It defaults to
`none` for that reason. Only the *step* half of a "cfg_step" distill helps anyway —
DiffHDR already runs at `cfg_scale` 1.0.

### Fitting it on your card

| `dit_dtype` | DiT + VACE weights | Notes |
|---|---|---|
| `bf16` | 32.3 GiB | Reference behaviour. |
| `fp8_e4m3fn` | 16.2 GiB | Casts the fp16 file on load. The practical choice on 24–32 GB. |
| `fp16` | 32.3 GiB | No saving: computation is bf16 regardless, so this is bit-identical to `bf16` except for a sub-ULP difference on LoRA-merged layers, plus a cast per layer per step. Use `bf16`. |

(Parameter counts from the model definitions: DiT 14.29 B, VACE 3.05 B.)

`vram_mode` is separate: `auto` (keep blocks on the GPU until it is nearly full, stream
the rest from system RAM every step — works nearly everywhere), `full_gpu` (all
resident, never streams), `streaming` (explicit budget via `vram_limit_gb`). Blocks
`auto` keeps on the GPU are upcast to bf16 whatever `dit_dtype` says; `full_gpu` keeps
them in the loaded dtype and casts each layer to bf16 as it runs.

**`full_gpu` + fp8 on a 24 GB card — expected to fit since 0.2.0, not yet measured.**
Before 0.2.0 it could not fit at all: a bug in the vendored diffsynth held the
3 B-parameter VACE branch in fp32 (11.4 GiB), putting fp8 at 24.7 GiB of weights. That is
fixed (`DiffHDR/VENDORED.md`, item 5) with bit-identical output, leaving 16.2 GiB of
weights and ~7 GiB for activations. It could also be much faster than `auto`: the timings
below work out to ~14 s per extra step at 3,120 tokens (147 s at 10 steps, 61 s at 4),
where the arithmetic alone should take ~2 s on an A5000 — so most of each `auto` step is
moving weights, which `full_gpu` never does. The Sample node logs peak VRAM and time per
pass, so one run with `vram_mode = full_gpu` settles it.

The VAE always runs fp32. It is small, runs once, and carries the log signal back out.

#### What fits: it is the token count, not the resolution

The DiT sees `((frames-1)//4 + 1) x (height/16) x (width/16)` tokens, and that single
number decides whether a job runs. Measured on a 24 GB A5000, fp8 + `auto`, 4 steps:

| frames | resolution | tokens | | peak VRAM |
|---:|---|---:|---|---:|
| 5 | 832×480 | 3,120 | ok | 16.7 GiB |
| 17 | 832×480 | 7,800 | ok | 16.4 GiB |
| 5 | 1280×720 | 7,200 | ok | 16.7 GiB |
| **9** | **1280×720** | **10,800** | **ok** | **17.8 GiB** |
| 13 | 1280×720 | 14,400 | OOM | — |
| 33 | 832×480 | 14,040 | OOM | — |
| 17 | 1280×720 | 18,000 | OOM | — |
| 33 | 1280×720 | 32,400 | OOM | — |

Two things follow.

**The trained configuration — 33 frames at 1280×720, 32,400 tokens — does not fit on a
24 GB card**, with or without fp8, streaming, tiled VAE or a speed LoRA. Weights are not
the constraint (the floor is ~16.5 GiB of resident fp8 DiT and barely moves); attention
activations are. Budget roughly 11,000 tokens on 24 GB, and note the ceiling is a cliff,
not a slope — 10,800 works, 14,400 does not.

**The fix is `window_size`, which you already have.** It bounds tokens per pass, and the
windows are blended as usual, so a clip of any length runs at any resolution if you
shrink it. At 1280×720:

| `window_size` | tokens/pass | fits 24 GB |
|---:|---:|---|
| 33 | 32,400 | no |
| 17 | 18,000 | no |
| 13 | 14,400 | no |
| **9** | **10,800** | **yes** |

So full-resolution 720p work on a 24 GB card means `window_size = 9`, at the cost of
more passes. On a 48 GB or 80 GB card, raise it back up.

**Measured**, RTX A5000 (24 GB), `fp8_e4m3fn` + `auto`, 4 steps:

| | |
|---|---|
| pipeline load (cold, from NFS) | 66 s |
| 5 frames @ 832×480 | 106 s |
| 13 frames @ 576×320, 2 windows | 94–134 s |
| single panorama @ 512×256 | 44 s |

fp8 is safe for this model, which is worth stating because it is not obvious: the LoRA
is merged into weights that have *already* been cast, so a small enough delta could be
rounded straight back out while still reporting a successful merge. Measured against the
real weights, the DiffHDR delta is 3–7 % of each tensor's norm and **survives the fp8
round trip at ~104 %** (the excess is fp8 noise) — comfortably above the quantisation
step, so nothing is lost.

---

## Using it

```
Load Image / Load Video ──▶ DiffHDR Encode ──▶ DiffHDR Sample ──▶ (linear IMAGE)
                                  │                  ▲
                                  └─ over/under MASK │
        DiffHDR Load Model ───────────────────────────┘
```

Start with `steps = 10` and `cfg_scale = 1.0` — upstream's own fast settings, and
usually indistinguishable from the paper's 50 steps for this task. Leave the prompt
empty.

The node only really has one dial worth turning: **`over_threshold` on Encode**. Lower
it if genuinely clipped highlights are being left alone; raise it if bright-but-intact
areas are being rebuilt with invented detail. Look at the `mask` output while you
do it (the example workflows already preview it), and connect your own `MASK` to Encode
for the cases detection gets wrong — `add` for missed clipping, `subtract` to protect
bright detail that is intact.

### Four jobs, no mode widget

| You feed it | It does |
|---|---|
| one image | replicates to the trained token count (33 frames at 1280×720, fewer at higher res), capped at `window_size`, and returns frame 0 — upstream's `infer_image.py` behaviour |
| ≤ `window_size` frames | one pass |
| > `window_size` frames | overlapping windows, cross-faded in linear space (see below) |
| Encode `mode=panorama` + `DiffHDR_Pano` LoRA | each image as a single genuine frame at 2048×1024 — a batch is processed one panorama at a time |

### Long clips

The cross-fade weights are exact — `test_windows.py` shows they reconstruct a constant
or a ramp perfectly — so a visible seam is never the blend arithmetic. It is two windows
independently inventing *different* highlight content where they overlap, with the fade
then dissolving between two answers.

`use_prev_window_reference` (on by default) is upstream's fix: each window gets a frame
of the previous window's output as its reference. Measured on a 13-frame clip across two
windows, worst-case frame-to-frame luminance step in the overlap:

| | seam ratio (max Δ ÷ median Δ) | worst overlap Δ |
|---|---|---|
| off (upstream's default) | 5.38× | 0.0399 |
| **on (default here)** | **2.76×** | **0.0138** |

Upstream ships this as an opt-in flag. It defaults to on here because the artefact it
removes is the main thing that makes long-clip output look stitched. Turn it off to
match upstream exactly.

### Writing EXR

Wire the `IMAGE` output into whatever EXR writer you already use — this pack
deliberately ships none, so it composes with what you have.

If you need one, [ComfyUI-BAT-NodePack](https://github.com/BeeeFX/ComfyUI-BAT-NodePack)
has **Video Combine** with an `image/exr-sequence` format that writes through
OpenImageIO with the full compression set (including DWAA) and no clamp. ffmpeg's EXR
encoder only offers none/rle/zip, which is why a generic video saver is a poor fit for
scene-linear output.

Two things to expect:

* **The node preview looks blown out.** ComfyUI clamps previews to `[0,1]` and this data
  deliberately goes above it. That is the preview, not the data.
* **The values are scene-linear with Rec.709 primaries.** A log code of 1.0 is exactly
  65536.0 linear. Chain an OCIO node to get to ACEScg or a display transform.

---

## Optimisation status

What this pack already does:

| | |
|---|---|
| **fp8 weights** | `dit_dtype=fp8_e4m3fn` halves the weights (32.3 → 16.2 GiB). Measured safe for this LoRA (see above). |
| **DiT block streaming** | `vram_mode`. Weights are not the binding constraint at 24 GB; activations are. |
| **Tiled VAE** | On by default. |
| **Prompt-embedding cache** | DiffHDR runs a constant (empty) prompt, but the pipeline re-encoded it every job *and* swapped the 11 GB T5 onto the GPU to do so. Now cached at the unit level, so both are skipped after the first run. |
| **Step-distill LoRA** | `speed_lora`, 2.0–2.4× — with the peak-radiance caveat above. |

### Things that were evaluated and rejected

**SageAttention — not worth it on Ampere.** Benchmarked against SDPA at the exact shapes
Wan2.1-14B uses (40 heads, head_dim 128, bf16):

| seq_len | SDPA | Sage | speedup | rel L2 error |
|---:|---:|---:|---:|---:|
| 3,120 | 3.34 ms | 3.42 ms | 0.98× | 0.014 |
| 7,200 | 16.3 ms | 15.5 ms | 1.05× | 0.014 |
| 10,800 | 37.0 ms | 35.3 ms | 1.05× | 0.014 |
| 32,400 | 325 ms | 286 ms | 1.14× | 0.014 |

At the sizes that actually fit on a 24 GB card this is **1.05×**, on attention only, for
1.4 % relative error. PyPI ships only SageAttention 1.0.6, the pure-Triton build;
the 2–5× figures belong to SageAttention 2's CUDA kernels, which need sm_89+ (Ada or
Hopper) and are source-only. An A5000 is sm_86. Worth revisiting on newer cards; not
implemented here, and the test package was uninstalled afterwards.

**torch.compile — not implemented, and not measured end to end.** It is *not*
structurally incompatible with diffsynth's offload wrappers, which was the first
assumption here and was wrong: dynamo specialises on the offload state and guards it
(2 graphs per module, bounded) rather than breaking, and it handles varying sequence
lengths dynamically. The reasons for leaving it are weaker than a measurement: a 14 B
transformer at 10<sup>4</sup> tokens is dominated by large matmuls that are already near
peak, so compile's usual wins (fusing elementwise work, cutting launch overhead) have
little to bite on; and warmup across 40 blocks is minutes. `full_gpu`, the mode with
the fewest moving parts to compile, did not fit on 24 GB until the VACE fix in 0.2.0;
worth revisiting once it has been measured there.

**TeaCache — implemented, off by default, and underwhelming here.** It is exposed as
`tea_cache` on the Sample node. Measured, 5 frames at 832×480, 10 steps:

| `tea_cache` | speedup | peak radiance | stops lost | log MAE |
|---:|---:|---:|---:|---:|
| 0 (off) | 1.00× | 19.9 | — | — |
| 0.05 | 1.02× | 19.9 | 0.00 | 0.0000 |
| 0.10 | 1.01× | 19.9 | 0.00 | 0.0000 |
| 0.15 | 1.19× | 16.7 | 0.25 | 0.0012 |

At 0.05 and 0.10 the output is **bit-identical to baseline** — it skipped no steps at
all. That is the borrowed calibration showing: upstream's coefficient table has no
Wan2.1-VACE-14B entry, so the T2V-14B polynomial is used, and it maps this model's
observed step-to-step distances onto values that almost never reach the threshold. Only
at 0.15 does it engage, for a modest 1.19×.

Per stop of lost highlight it is a worse trade than a step-distill LoRA (1.19× for 0.25
stops, versus 2.03× for 0.96). Kept because it is written, tested and costs nothing when
off, but it is not recommended as-is. The fix is to fit TeaCache coefficients against
VACE-14B rather than borrow T2V's — a calibration run, not a code change.

**FlashAttention-2** — not installed, and worth a try if you want: diffsynth picks it up
automatically when present (`pip install flash_attn`), no code change. Expect modest
gains on sm_86, since SDPA already dispatches to efficient kernels.

And one structural limit worth knowing: diffsynth's own `sliding_window_size` is
**broken for VACE** — it slices only `latents` and `y`, leaving `vace_context` at full
length, and the VACE block then does `new_zeros(1, x.shape[1] - u.size(1), ...)` with a
negative dimension. Verified: `RuntimeError: Trying to create tensor with negative
dimension -21600`. That is why long clips are windowed at the output level here (one
full pass per window) rather than inside the denoise loop. Fixing it upstream would mean
slicing `vace_context` on the latent-time axis in step with the latents.

## Verifying

Six suites, none needing a GPU or weights:

```bash
cd ComfyUI
env/bin/python custom_nodes/ComfyUI-DiffHDR/tests/run_all.py
```

`test_colour_parity.py` is the one that matters. Since the model is stock Wan, the log
curve and the exposure masks *are* the technique — so that suite asserts this pack's
port is **bitwise identical** to the vendored upstream originals, function by function.
If it ever fails, the output has stopped being DiffHDR while still looking plausible.

`test_model_scan.py` checks the header pre-flight against your actual asset library:
that the right files are accepted, the known-bad ones rejected, and that the hash
implementation still agrees with the vendored original. It also checks that the loader
dropdowns list each file once and that the tokenizer is found through
`extra_model_paths.yaml` folders and symlinks.

`test_nodes.py` drives Encode and Sample (with a stand-in pipeline) and checks the
shipped example workflows line up widget-for-widget with what the frontend builds —
the same layout in the classic and Nodes 2.0 renderers. `test_runtime.py` checks that
the one-pipeline cache really frees memory and that a released pipeline is rebuilt on
demand. Two suites need a ComfyUI install and are skipped from a bare clone.

Sampling itself needs weights and a GPU, so it is not in the suite. It has been verified
by hand on an RTX A5000: a 5-frame clip, a 13-frame clip across two overlapping windows,
and a single-frame panorama with the Pano LoRA — checking output shape, finiteness, real
above-1.0 headroom, and that no brightness step appears at a window seam.

---

## How it wraps upstream

Three decisions worth knowing if you are reading the code:

**Vendored, not cloned.** `DiffHDR/` is committed rather than fetched on first run.
A runtime clone does not survive a deploy that mirrors directories (`rsync --delete` and
friends): the tree gets wiped while leaving enough behind that a naive "is it present?"
check still passes, and the loader then fails with a confusing ImportError. Vendoring
makes an install self-contained, and `DiffHDR/VENDORED.md` records exactly what was
changed and how to re-vendor at a newer commit.

**Loaded under a private module name.** The vendored tree is imported as
`diffhdr_vendored_diffsynth`, never as the global `diffsynth`. `sys.path` is only
consulted for names not already in `sys.modules`, so if any other custom node ever ships
a `diffsynth`, whichever imported first would win — silently. Every self-import inside
the vendored tree is relative, which is what makes the rename safe.

**Two upstream imports had to be rewritten**, itemised in `DiffHDR/VENDORED.md`. Upstream
reaches for a *top-level* `utils` package, which works from its repo root but cannot work
inside ComfyUI — ComfyUI ships its own `ComfyUI/utils/`. Honouring that spelling would
either fail or shadow ComfyUI core for every other node in the process. Separately, three
eager `modelscope` imports were deferred, so a large download-only dependency is not
required for functionality this pack never uses.

**One memory bug was fixed**, also itemised there: diffsynth's VRAM management gave the
VACE branch the VAE's dtype (fp32 here) instead of its own. One line; outputs are
bit-identical, the saving is 8.5 GiB at fp8.

No model, sampling or colour logic in the vendored tree has been changed.

---

## Status

Working and measured, but young — first public release. The numbers throughout this
README were taken on a single 24 GB RTX A5000; if you run it on other hardware I would
be glad to hear what you get — particularly `full_gpu` at fp8 on a 24 GB card, which
should fit since 0.2.0 but has not been measured, and 48 GB+ cards where
`torch.compile` is worth retesting.

Known rough edges, all documented above rather than hidden: TeaCache barely engages
because upstream's coefficient table has no VACE entry; SageAttention is not worth it on
Ampere; and the trained 33-frame 720p configuration does not fit on 24 GB (use
`window_size`).

## Credits

- **[DiffHDR](https://github.com/Eyeline-Labs/DiffHDR)** by Eyeline Labs — the model,
  the method, and the pipeline vendored under `DiffHDR/`. This pack is a wrapper; the
  research is theirs.
- **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** — the diffusion
  framework DiffHDR's pipeline is derived from.
- **[Wan2.1](https://github.com/Wan-Video/Wan2.1)** (Alibaba) — the VACE-14B base model.

## Licence

Apache-2.0, matching upstream. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Model weights are **not** redistributed here and carry their own terms — DiffHDR's LoRA
and Alibaba's Wan2.1 licence both apply to what you produce. Both are Apache-2.0 at the
time of writing, but check before relying on that for commercial work.
