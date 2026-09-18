# Vendored upstream: DiffHDR

| | |
|---|---|
| Upstream | https://github.com/Eyeline-Labs/DiffHDR |
| Pinned commit | `cfc1b48bdf42e07e75ef3e360d2d1a6dd27403d0` |
| License | Apache-2.0 (see `./LICENSE`) |
| Vendored on | 2026-09-16 |

This directory is a **redistributed copy** of part of the upstream repository. It is
vendored and committed rather than cloned on first run: this suite deploys with
`rsync --delete`, which wipes a runtime clone while leaving enough behind that a
"is it present?" check still passes, after which the loader fails with a confusing
`ImportError`. Vendoring makes a deploy self-contained.

The upstream tree is never imported as the global name `diffsynth`. It is loaded under
a private module name by `../diffhdr_runtime/vendor.py` — see that file for why.

---

## Modifications to the upstream copy

Apache-2.0 §4(b) requires these be stated. They are the minimum needed to make the
package import inside ComfyUI; **no model, sampling or colour logic has been changed.**

1. **Removed directories** (not needed for inference, and each pulled in dependencies
   this pack would otherwise have to require):
   - `diffsynth/trainers/` — training only; pulled in `accelerate`.
   - `diffsynth/data/` — `VideoData` / EXR frame datasets; pulled in `imageio`,
     `torchvision`, `pandas`, `pyexr`. Frames reach this pack as ComfyUI IMAGE
     tensors, so none of it is reachable.
   - Everything outside `diffsynth/` and `utils/`: `eval/`, `scripts/`, `assets/`,
     `train_hdr.py`, and `demo/` (94 MB of sample media).

2. **`diffsynth/__init__.py`** — dropped `from .data import *` to match (1).

3. **Relocated `utils/` → `diffsynth/diffhdr_utils/`**, keeping only `color_utils.py`
   and `token_utils.py`, and rewrote the three import sites accordingly:
   - `diffsynth/pipelines/wan_video_new.py` (2 imports)
   - `diffsynth/models/wan_video_colour_transoform.py` (1 import)

   **This one is not cosmetic.** Upstream reaches for a *top-level* `utils` package
   (`from utils.color_utils import ...`), which works when you run its scripts from the
   repo root. Inside ComfyUI it cannot: ComfyUI ships its own `ComfyUI/utils/` package,
   which is already on `sys.path`. Honouring upstream's spelling would either resolve
   to ComfyUI's `utils` and fail, or — if we forced ours to win — shadow ComfyUI core
   for every other node in the process.

4. **Deferred `modelscope` imports.** Upstream imports `snapshot_download` at module
   scope in three places, but every call site is inside `download_if_necessary()` and is
   only reached when a model path is `None`. This pack always resolves explicit local
   paths through `folder_paths`, so the import is moved to the call:
   - `diffsynth/pipelines/wan_video_new.py`, `diffsynth/utils/__init__.py` — inlined at
     the call site.
   - `diffsynth/models/downloader.py` — routed via
     `diffsynth/diffhdr_utils/lazy_modelscope.py` (this module is imported at package
     load by `model_manager`, so the import had to be deferred rather than moved).

   Without this, `modelscope` — a large package — would be a hard install requirement
   for functionality this pack never uses.

No other file has been edited.

---

## Re-vendoring at a newer commit

```bash
git clone --filter=blob:none https://github.com/Eyeline-Labs/DiffHDR.git /tmp/diffhdr-new
git -C /tmp/diffhdr-new checkout <new-commit>

rm -rf DiffHDR/diffsynth
cp -r /tmp/diffhdr-new/diffsynth DiffHDR/diffsynth
cp /tmp/diffhdr-new/LICENSE DiffHDR/LICENSE
rm -rf DiffHDR/diffsynth/trainers DiffHDR/diffsynth/data

mkdir -p DiffHDR/diffsynth/diffhdr_utils
cp /tmp/diffhdr-new/utils/color_utils.py /tmp/diffhdr-new/utils/token_utils.py \
   DiffHDR/diffsynth/diffhdr_utils/
# re-create diffhdr_utils/__init__.py and lazy_modelscope.py (keep the existing ones)

# then re-apply edits 2-4 above, and:
sed -i 's/^from utils\.color_utils import/from ..diffhdr_utils.color_utils import/' \
    DiffHDR/diffsynth/pipelines/wan_video_new.py \
    DiffHDR/diffsynth/models/wan_video_colour_transoform.py
sed -i 's/^from utils\.token_utils import/from ..diffhdr_utils.token_utils import/' \
    DiffHDR/diffsynth/pipelines/wan_video_new.py
```

Then bump the commit above and run the test suite. `tests/test_colour_parity.py`
compares `diffhdr_runtime/colour.py` against the vendored `color_utils.py`, so it will
catch any upstream change to the colour maths; `tests/test_model_scan.py` checks the
checkpoint-detection table is still in step.

Verify afterwards that these all still hold:

```bash
grep -rn '^from utils\.\|^import utils' DiffHDR/diffsynth/        # must be empty
grep -rn '^from modelscope\|^import modelscope' DiffHDR/diffsynth/ # must be empty
grep -rn 'from diffsynth\|import diffsynth' DiffHDR/diffsynth/     # must be empty
```

That last one matters: the private-name import in `vendor.py` only works because every
self-import inside the tree is relative.
