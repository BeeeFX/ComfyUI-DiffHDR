"""Find the Wan2.1 pieces DiffHDR needs, and reject the wrong ones before loading them.

DiffHDR runs stock Wan2.1-VACE-14B, so most studios already have the weights -- the
problem is that several files with plausible names do NOT work, and the failure
otherwise arrives minutes into a 34 GB load as an obscure ``NoneType`` from deep
inside diffsynth.

diffsynth identifies a checkpoint by MD5-ing its ``key:shape`` pairs against a table
in ``configs/model_config.py``. That hash is computable from the safetensors *header*
alone -- a few KB off the front of the file -- so this module runs the same check in
milliseconds, before anything is loaded, and names the actual problem when it fails.

Verified against the local asset library:

    wan2.1_vace_14B_fp16.safetensors        -> dit + vace   OK
    umt5-xxl-enc-bf16.safetensors           -> text encoder OK
    wan_2.1_vae.safetensors                 -> vae          OK
    Wan2_1_VAE_bf16.safetensors (Kijai)     -> vae          OK
    Wan2_1-VACE_module_14B_*.safetensors    -> REJECTED (VACE module only, 237 keys)
    umt5_xxl_fp16.safetensors               -> REJECTED (243 keys, not the encoder-only build)
    wan2.1_vace_1.3B_fp16.safetensors       -> REJECTED (no entry in upstream's table)

The hash covers keys and shapes but not dtype, which is what makes the fp8 path in
``runtime.py`` possible: the same fp16 file can be cast on load and still validate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct

try:
    import folder_paths
except ImportError:  # pragma: no cover - only when imported outside a ComfyUI install
    # Importing this module must not require ComfyUI. Everything that genuinely needs
    # the model library raises a clear error at call time instead, which keeps the pack
    # inspectable and its non-ComfyUI tests runnable from a bare clone.
    folder_paths = None

from . import vendor

logger = logging.getLogger("ComfyUI-DiffHDR")

AUTO = "auto"
NONE = "none"

# Model folders scanned for each role, in preference order. "diffhdr" is ours; the
# rest are ComfyUI's standard folders, so a user who already runs Wan needs no new
# downloads and no new directories.
_ROLE_FOLDERS = {
    "dit": ("diffhdr", "diffusion_models", "unet", "checkpoints"),
    "text_encoder": ("diffhdr", "text_encoders", "clip"),
    "vae": ("diffhdr", "vae"),
    "lora": ("diffhdr", "loras"),
}

_EXTS = (".safetensors", ".pth", ".pt", ".ckpt", ".bin")


def _models_dir() -> str:
    if folder_paths is None:
        raise RuntimeError(
            "ComfyUI-DiffHDR: ComfyUI's folder_paths is not importable, so model "
            "directories cannot be resolved. This pack is a ComfyUI node pack -- run it "
            "from inside a ComfyUI install."
        )
    return folder_paths.models_dir


def register_model_folder() -> str:
    """Register ``models/diffhdr`` so users can keep DiffHDR weights together if they want.

    Additive only: the standard ComfyUI folders are still scanned, because the whole
    point is that an existing Wan install works untouched.
    """
    path = os.path.join(_models_dir(), "diffhdr")
    try:
        os.makedirs(path, exist_ok=True)
        folder_paths.add_model_folder_path("diffhdr", path)
    except Exception as exc:  # noqa: BLE001 - a read-only models dir must not kill the pack
        logger.warning("ComfyUI-DiffHDR: could not register %s (%s)", path, exc)
    return path


# Registered at import, not from INPUT_TYPES: it is only a makedirs plus a dict entry,
# and doing it lazily meant `resolve()` could not find anything in models/diffhdr when a
# caller reached the node without INPUT_TYPES having been evaluated first. The expensive
# part -- actually scanning the folders -- stays lazy, in list_candidates().
if folder_paths is not None:
    register_model_folder()


def list_candidates(role: str) -> list[str]:
    """Filenames offered for a role, as ``<folder>/<name>`` so duplicate basenames stay distinct.

    Called from INPUT_TYPES, which ComfyUI evaluates lazily -- deliberately not at
    import time, because stat-ing several model folders on an NFS asset library is a
    measurable ComfyUI startup stall for every user whether or not they open this node.
    """
    if folder_paths is None:
        return []
    # Current ComfyUI maps the legacy names "unet" -> "diffusion_models" and
    # "clip" -> "text_encoders", so scanning both spellings (kept for older installs)
    # would list every file twice under two labels. Dedupe on the canonical folder
    # name rather than on each file's real path: that would be a stat per file on
    # every /object_info, which on an NFS LoRA library is a visible stall.
    canonical = getattr(folder_paths, "map_legacy", lambda f: f)
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for folder in _ROLE_FOLDERS.get(role, ()):  # preference order
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:
            continue
        for n in names:
            if not n.lower().endswith(_EXTS):
                continue
            key = (canonical(folder), n)
            if key not in seen:
                seen.add(key)
                out.append(f"{folder}/{n}")
    return out


def resolve(role: str, label: str) -> str:
    """Turn a ``<folder>/<name>`` widget value back into an absolute path."""
    if not label or label == AUTO:
        raise ValueError(f"ComfyUI-DiffHDR: no {role} selected.")
    folder, _, name = label.partition("/")
    path = None
    if name:
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:
            path = None
    if path is None:  # tolerate a bare filename typed or loaded from an older workflow
        for f in _ROLE_FOLDERS.get(role, ()):
            try:
                path = folder_paths.get_full_path(f, label)
            except Exception:
                path = None
            if path:
                break
    if path is None or not os.path.exists(path):
        raise ValueError(
            f"ComfyUI-DiffHDR: {role} file {label!r} not found. It may have been moved "
            "or removed since this workflow was saved."
        )
    return path


# --------------------------------------------------------------------------- #
# Header-only identification
# --------------------------------------------------------------------------- #

def read_key_shapes(path: str) -> dict[str, list[int]] | None:
    """Read ``{key: shape}`` from a safetensors header without loading any tensor data.

    Returns ``None`` for non-safetensors files (``.pth``/``.ckpt``), which cannot be
    inspected cheaply -- those skip validation and are identified by diffsynth on load.
    """
    if not path.lower().endswith(".safetensors"):
        return None
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n <= 0 or n > 512 * 1024 * 1024:
                return None
            header = json.loads(fh.read(n))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ComfyUI-DiffHDR: could not read header of %s (%s)", path, exc)
        return None
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__" and isinstance(v, dict) and "shape" in v}


def _hash(key_shapes: dict[str, list[int]], with_shape: bool = True) -> str:
    """Reimplementation of diffsynth ``hash_state_dict_keys`` over header data.

    Mirrors ``convert_state_dict_keys_to_single_str``: for each tensor emit
    ``"key:d0_d1_..."`` then ``"key"``, sort, join with commas, MD5. Kept in step with
    the vendored original by ``tests/test_model_scan.py``.
    """
    keys: list[str] = []
    for key, shape in key_shapes.items():
        if with_shape:
            keys.append(key + ":" + "_".join(map(str, shape)))
        keys.append(key)
    keys.sort()
    return hashlib.md5(",".join(keys).encode("UTF-8")).hexdigest()


def _known_hashes() -> dict[str, tuple[str, ...]]:
    """``{shape_hash: (model_name, ...)}`` from upstream's detection table."""
    table = {}
    for keys_hash, keys_hash_with_shape, model_names, _classes, _res in vendor.model_loader_configs():
        for h in (keys_hash, keys_hash_with_shape):
            if h:
                table[h] = tuple(model_names)
    return table


def identify(path: str) -> tuple[str, ...] | None:
    """What diffsynth will call this file, or ``None`` if it is unrecognised/uninspectable."""
    key_shapes = read_key_shapes(path)
    if key_shapes is None:
        return None
    known = _known_hashes()
    for h in (_hash(key_shapes, True), _hash(key_shapes, False)):
        if h in known:
            return known[h]
    return None


def _diagnose(path: str, key_shapes: dict[str, list[int]]) -> str:
    """Explain *why* a file was rejected, in terms of what the user should get instead."""
    keys = set(key_shapes)
    has_vace = any(k.startswith("vace_blocks.") for k in keys)
    has_dit = any(k.startswith("blocks.") for k in keys)
    name = os.path.basename(path)

    if any(k.endswith((".scale_weight", ".scale_input")) for k in keys):
        return (
            f"{name} looks like an fp8-*scaled* repack: it carries per-tensor scale "
            "tensors that change the key set, so diffsynth cannot identify it. Use the "
            "plain fp16/bf16 Wan2.1-VACE-14B file and set dit_dtype=fp8_e4m3fn on this "
            "node instead -- that gives you the same VRAM saving through a path "
            "diffsynth understands."
        )
    if has_vace and not has_dit:
        return (
            f"{name} is a VACE *module* checkpoint ({len(keys)} tensors) -- the split "
            "release intended for ComfyUI-WanVideoWrapper, which pairs it with a "
            "separate T2V base at runtime. diffsynth needs the single merged "
            "Wan2.1-VACE-14B checkpoint (~1331 tensors), e.g. "
            "wan2.1_vace_14B_fp16.safetensors."
        )
    if has_dit and not has_vace:
        return (
            f"{name} is a plain Wan2.1 DiT ({len(keys)} tensors) with no VACE blocks. "
            "DiffHDR conditions through VACE, so it needs a VACE checkpoint -- "
            "Wan2.1-VACE-14B, not T2V/I2V."
        )
    return (
        f"{name} ({len(keys)} tensors) does not match any checkpoint diffsynth knows. "
        "If this is a Wan variant DiffHDR was not built against (1.3B, or a GGUF "
        "quantisation), it will not work here."
    )


def validate(path: str, expect: str, role: str) -> None:
    """Raise a useful error if ``path`` is not the kind of model ``role`` needs.

    ``expect`` is a diffsynth model name (``wan_video_vace``, ``wan_video_text_encoder``,
    ``wan_video_vae``). Non-safetensors files are passed through unvalidated.
    """
    key_shapes = read_key_shapes(path)
    if key_shapes is None:
        logger.info(
            "ComfyUI-DiffHDR: %s is not a safetensors file, skipping pre-flight; "
            "diffsynth will identify it on load.", os.path.basename(path)
        )
        return

    names = identify(path)
    if names is None:
        raise RuntimeError(f"ComfyUI-DiffHDR: {role} rejected. {_diagnose(path, key_shapes)}")
    if expect not in names:
        # Recognised, but the wrong kind. The structural diagnosis is more actionable
        # than the model names alone -- "no VACE blocks" tells the user what to go and
        # download, "is a wan_video_dit" does not.
        raise RuntimeError(
            f"ComfyUI-DiffHDR: {role} rejected. {_diagnose(path, key_shapes)} "
            f"(diffsynth identifies it as {'/'.join(names)}; this slot needs {expect}.)"
        )
    logger.info("ComfyUI-DiffHDR: %s -> %s OK", os.path.basename(path), "+".join(names))


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

_TOKENIZER_MARKERS = ("spiece.model", "tokenizer.json", "tokenizer_config.json")

# Folders whose registered paths are searched for a tokenizer as well as models/
# itself. Registered paths are where extra_model_paths.yaml points ComfyUI, which on a
# studio install is usually a shared library outside models/ altogether.
_TOKENIZER_FOLDERS = ("diffhdr", "text_encoders", "diffusion_models", "checkpoints")


def _tokenizer_roots() -> list[str]:
    roots = [_models_dir()]
    for folder in _TOKENIZER_FOLDERS:
        try:
            roots.extend(folder_paths.get_folder_paths(folder))
        except Exception:
            continue
    return roots


def find_tokenizers() -> list[str]:
    """Locate ``google/umt5-xxl`` tokenizer directories already present on disk.

    The tokenizer is ~5 MB and ships inside the official Wan repos, so on any machine
    that has ever staged a full Wan checkout it is already on disk -- worth finding
    rather than downloading again.

    Symlinks are followed (model folders are very often symlinks into a shared
    library), with each real directory visited once so a link cycle cannot loop.
    """
    found: list[str] = []
    seen: set[str] = set()
    visited: set[str] = set()
    for root in _tokenizer_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            # Depth guard: the asset library is large and on NFS.
            if real in visited or dirpath[len(root):].count(os.sep) > 4:
                dirnames[:] = []
                continue
            visited.add(real)
            if any(m in filenames for m in _TOKENIZER_MARKERS) and "umt5" in dirpath.lower():
                if real not in seen:
                    seen.add(real)
                    found.append(dirpath)
                dirnames[:] = []
    return found


def resolve_tokenizer(value: str) -> str:
    """Resolve the tokenizer widget: an explicit path, or ``auto`` -> first one found."""
    if value and value != AUTO:
        if os.path.isdir(value):
            return value
        raise ValueError(f"ComfyUI-DiffHDR: tokenizer directory {value!r} does not exist.")
    found = find_tokenizers()
    if not found:
        raise RuntimeError(
            "ComfyUI-DiffHDR: no umt5-xxl tokenizer found under your models directory. "
            "It ships inside the official Wan repos; fetch just the tokenizer with:\n"
            "  hf download Wan-AI/Wan2.1-VACE-14B --include 'google/umt5-xxl/*' "
            "--local-dir <models>/diffhdr/Wan2.1-VACE-14B\n"
            "then set tokenizer_dir to that google/umt5-xxl folder."
        )
    logger.info("ComfyUI-DiffHDR: tokenizer auto-detected at %s", found[0])
    return found[0]
