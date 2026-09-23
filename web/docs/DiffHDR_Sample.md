# DiffHDR Sample

Runs the diffusion pass and returns **unclamped scene-linear** frames.

## The preview will look wrong

ComfyUI clamps node previews to [0,1]. This output deliberately goes far above that, so
the thumbnail will look blown out. That is the preview, not the data. To see it
properly, put it through a view transform, or write it to EXR and open it in a viewer
that handles scene-linear.

The values are scene-linear with **Rec.709 primaries**. A log code of 1.0 corresponds to
65536.0 linear, so there is plenty of headroom above diffuse white.

## Output

A plain IMAGE. This node writes no files — wire it into whatever saver you already use.
If you need one, ComfyUI-BAT-NodePack has **Video Combine** with an
`image/exr-sequence` format that writes through OpenImageIO with the full compression
set (including DWAA) and no clamp.

## Four jobs, one node

You do not choose a mode; it follows from what you feed it.

* **Single image** — one frame in. It is replicated to match the token count the model
  was trained on (33 frames at 1280×720, fewer at higher resolutions) and frame 0 comes
  back. This is what upstream's `infer_image.py` does, and it is why a still costs about
  as much as a clip. The replication is capped at `window_size`, so the setting that
  makes clips fit your card makes stills fit too.
* **Clip** — up to `window_size` frames, one pass.
* **Long clip** — more than `window_size` frames: overlapping windows, cross-faded.
* **Panorama** — `mode=panorama` on the Encode node. Each frame runs as one genuine
  frame; a batch of panoramas is processed one at a time, never as a clip.

## Settings

**`steps`** — 50 reproduces the paper; **10 is upstream's own fast setting** and is
usually indistinguishable for this task at a fifth of the time. Start at 10.

**`cfg_scale`** — leave at 1.0. That disables classifier-free guidance, which is what
upstream uses and halves the work per step. Above 1.0 costs double and tends to
exaggerate the invented highlights.

**`seed`** — defaults to *fixed*, not randomize. A seed that changes after every run
would make ComfyUI re-run this node — the most expensive one in the graph — every time
you queue, even if you only changed something downstream, and would give a different
reconstruction of the same shot each time.

**`prompt`** — leave empty, normally. The task is radiance inpainting, not
text-to-video; a prompt mostly risks steering invented highlight content away from the
plate. `negative_prompt` does nothing while `cfg_scale` is 1.0.

**`tea_cache`** skips steps whose result is predicted to change little. Measured on this
model it is **disappointing**, and the widget defaults to 0 (off) for that reason:
0.05 and 0.10 skip nothing at all (output bit-identical to baseline), and 0.15 buys only
1.19× for a quarter-stop of peak radiance.

The cause is calibration, not the technique. Upstream's coefficient table has no
Wan2.1-VACE-14B entry, so the T2V-14B curve is borrowed, and it under-reports this
model's step-to-step change. Fitting coefficients against VACE would likely make the
low thresholds useful. Until then, a step-distill LoRA is the better speed lever.

**`window_size` is also the VRAM control.** The DiT sees
`((window_size-1)//4 + 1) x (height/16) x (width/16)` tokens, and that number is what
decides whether a job fits. On a 24 GB card the ceiling is around 11,000 tokens — which
means 1280×720 needs `window_size = 9` (10,800 tokens), because 13 frames (14,400) OOMs.
The trained 33 frames at 1280×720 is 32,400 tokens and does not fit on 24 GB at all.
Lowering `window_size` costs passes, not quality, since the windows are blended anyway.

**`window_size` / `window_stride`** — only matter for clips longer than `window_size`.
33 is the trained length; going beyond it degrades quality rather than helping. The
overlap (`window_size − window_stride`) is the cross-fade length: a wider overlap costs
more passes but hides the seams better. The blend happens in linear space, after the log
decode, which is what keeps exposure continuous across a seam.

**`use_prev_window_reference`** — long clips only. Each window is handed a frame of the
previous window's output as its reference, so successive windows invent *consistent*
highlight content.

This matters more than it sounds. The cross-fade weights are exact — they reconstruct a
constant or a ramp perfectly — so a visible seam is never the blend arithmetic. It is
two windows independently inventing different content in the region where they overlap,
and the fade then dissolving between two different answers. Carrying a reference forward
is upstream's fix for it.

Upstream exposes this as an opt-in flag; it defaults to **on** here, because the failure
it prevents is the main artefact of long-clip mode. Measured on a 13-frame clip across
two windows, the worst frame-to-frame luminance step in the overlap fell from 0.0399 to
0.0138, and the seam ratio (max step ÷ median step) from 5.38× to 2.76×. Turn it off to match upstream's
default exactly, or if you are supplying your own `reference_image`.

**`reference_image`** (optional) — a style reference for the recovered highlights. It is
masked to the reconstructed regions, boosted by `reference_ev` stops and converted to
log, matching upstream. On long clips it seeds the first window; after that
`use_prev_window_reference` takes over, if enabled.

## Progress and cancelling

The progress bar tracks real sampler steps across the whole job — every window or
panorama — and Cancel takes effect within one step rather than at the end of the job.

## Memory

Before sampling, the node asks ComfyUI to move its own models off the GPU (they are
reloaded on demand). ComfyUI does not manage this pipeline, so it would not otherwise
make room for it, and DiffHDR's `auto` budget is measured from free VRAM.

If the pipeline was released — by DiffHDR Free Memory, or because another Load Model
node took the single slot — it is rebuilt automatically here.
