# DiffHDR Encode (LDR → log + masks)

Builds the conditioning DiffHDR samples from: the plate on a log curve, plus the masks
marking which pixels the model should treat as missing radiance.

This is a separate node because **the masks are the real creative control**. They decide
what gets reinvented, and they are the usual reason a result disappoints. They come out
as MASK outputs so you can look at them, and you can feed in your own.

## Outputs

* **`conditioning`** — for DiffHDR Sample.
* **`over_mask`** / **`under_mask`** — what the detector found.
* **`mask`** — what the model will actually reconstruct: the detection (over, plus under
  if enabled), combined with your own mask if one is connected. Preview this one
  (MaskToImage → Preview Image).

## What it does

1. Conforms the frames to `width` × `height` (both must be multiples of 16).
2. Detects over- and under-exposed regions.
3. Optionally stabilises those masks across time.
4. Converts sRGB → linear Rec.709 → DiffHDR's log curve.

The log curve is not LogC or ACEScct. It is a custom compression mapping 0–65536 linear
onto 0–1, and the model was trained on it specifically, so it is not interchangeable
with a camera log.

## Tuning the masks

**`over_threshold`** (default 0.95) is the one worth touching. It is the sRGB luma above
which a pixel counts as blown.

* Lower it if genuinely clipped highlights are being left alone.
* Raise it if bright-but-intact areas are being rebuilt — you will see invented detail
  replace real detail.

Detection is not a plain threshold: it is a 0.7/0.3 blend of a soft luma threshold and a
"at least two channels pinned at white" test. That second half is what stops a bright,
detailed, single-channel-saturated area reading as blown.

If the mask comes out empty the node warns, because the result will then be close to a
plain log→linear conversion with nothing reconstructed.

**`use_under_exposure_mask`** (default off, as upstream) also rebuilds crushed shadows.
It adds detail but is the more speculative half of the model, and in those regions it
replaces the conditioning with mid-grey — the model is told to invent, not to refine.

**Your own mask** (optional `mask` input). Detection gets most plates right, but not
all, so a MASK connected here is combined with it per `mask_mode`:

* **add** (default) — also reconstruct these regions: clipping the detector missed.
* **subtract** — protect these regions: bright but intact detail such as signage or
  speculars you want kept exactly as shot.
* **replace** — ignore the detection and use only this mask.

One mask applies to every frame, or supply one per frame. It is conformed with the same
`resize_mode` as the images and binarised at 0.5, since the model was trained on binary
masks. The mid-grey fill for crushed shadows only applies where the final mask is on, so
subtracting a crushed region really does keep it as shot.

**`temporal_stabilise`** (default on) smooths the masks frame to frame. Leave it on for
footage: without it the mask boils, the model is asked to invent a slightly different
region every frame, and that reads as flicker in the recovered highlights. It does
nothing to a single frame.

## Modes

* **video** — over and under masks, temporally stabilised. Use for footage *and* stills.
* **panorama** — over-exposure only, with different thresholds and morphology. This is
  not a simplification: upstream's HDRI path genuinely uses different maths, and the
  Pano LoRA was trained against it. Pair with `DiffHDR_Pano.safetensors`.

## Resolution

The model was trained at 1280×720 (panoramas at 2048×1024). Straying far from that
changes the token count the DiT sees and generally costs quality. `crop_and_resize`
centre-crops to the target aspect before scaling, so nothing is squashed. Downscaling is
antialiased, so a 4K plate taken to 720p does not alias — aliasing in a blown highlight
is what makes a mask speckle.

With `resize_mode = none` the frames keep their own size (which must be a multiple of
16) and `width`/`height` are ignored; Sample generates at the frames' size.
