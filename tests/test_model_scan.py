"""Check the header-only pre-flight agrees with diffsynth, and rejects the right files.

Needs no GPU and loads no weights -- it reads safetensors headers only.
    cd ComfyUI && env/bin/python custom_nodes/ComfyUI-DiffHDR/tests/test_model_scan.py
"""

import os
import sys

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.path.dirname(os.path.dirname(_PACK))
sys.path.insert(0, _COMFY)
sys.path.insert(0, _PACK)

import folder_paths  # noqa: E402

from diffhdr_runtime import model_scan  # noqa: E402

M = folder_paths.models_dir
failures = []


def expect(rel, want, note=""):
    path = os.path.join(M, rel)
    if not os.path.exists(path):
        print(f"  [skip] {rel} (not installed)")
        return
    got = model_scan.identify(path)
    ok = (got == want) if want is None else (want is not None and got is not None and want in got)
    print(f"  [{'ok ' if ok else 'FAIL'}] {os.path.basename(rel)} -> {got} {note}")
    if not ok:
        failures.append(f"{rel}: got {got}, wanted {want}")


print("files that must be ACCEPTED")
expect("diffusion_models/WAN21/wan2.1_vace_14B_fp16.safetensors", "wan_video_vace")
expect("text_encoders/umt5-xxl-enc-bf16.safetensors", "wan_video_text_encoder")
expect("vae/wan_2.1_vae.safetensors", "wan_video_vae")
expect("vae/Wan2_1_VAE_bf16.safetensors", "wan_video_vae", "(Kijai)")

print("files that must be REJECTED")
expect("diffusion_models/WAN21/Wan2_1-VACE_module_14B_bf16.safetensors", None, "(VACE module only)")
expect("diffusion_models/WAN21/Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors", None, "(VACE module only)")
expect("text_encoders/umt5_xxl_fp16.safetensors", None, "(not the encoder-only build)")
expect("diffusion_models/WAN21/wan2.1_vace_1.3B_fp16.safetensors", None, "(1.3B, unsupported)")

print("rejection messages name the real problem")
for rel, needle in [
    ("diffusion_models/WAN21/Wan2_1-VACE_module_14B_bf16.safetensors", "VACE *module*"),
    ("diffusion_models/WAN21/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", "no VACE blocks"),
]:
    path = os.path.join(M, rel)
    if not os.path.exists(path):
        print(f"  [skip] {rel}")
        continue
    try:
        model_scan.validate(path, "wan_video_vace", "wan_vace_model")
        print(f"  [FAIL] {os.path.basename(rel)} was accepted")
        failures.append(f"{rel} accepted")
    except RuntimeError as exc:
        ok = needle in str(exc)
        print(f"  [{'ok ' if ok else 'FAIL'}] {os.path.basename(rel)}: {str(exc)[:90]}...")
        if not ok:
            failures.append(f"{rel}: message missing {needle!r}")

print("hash implementation matches the vendored original")
import shutil  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from diffhdr_runtime import vendor  # noqa: E402

vendor.diffsynth()
import importlib  # noqa: E402

up = importlib.import_module("diffhdr_vendored_diffsynth.models.utils")
tmp = tempfile.mkdtemp(prefix="diffhdr_scan_")
# A synthetic file always, so parity is checked on every install -- not only on
# machines that happen to have one particular VAE in one particular place.
synth = os.path.join(tmp, "synthetic.safetensors")
save_file({"blocks.0.self_attn.q.weight": torch.zeros(4, 8), "head.bias": torch.zeros(3),
           "patch_embedding.weight": torch.zeros(2, 3, 1, 2, 2)}, synth)
probes = [synth] + [p for p in [os.path.join(M, "vae/wan_2.1_vae.safetensors")] if os.path.exists(p)]
for probe in probes:
    ks = model_scan.read_key_shapes(probe)
    fake_sd = {k: torch.empty(tuple(s), dtype=torch.uint8, device="meta") for k, s in ks.items()}
    for with_shape in (True, False):
        mine = model_scan._hash(ks, with_shape)
        theirs = up.hash_state_dict_keys(fake_sd, with_shape=with_shape)
        ok = mine == theirs
        print(f"  [{'ok ' if ok else 'FAIL'}] {os.path.basename(probe)} with_shape={with_shape}: {mine}")
        if not ok:
            failures.append(f"hash mismatch with_shape={with_shape}: {mine} vs {theirs}")

print("dropdowns list each file once")
# Current ComfyUI maps the legacy folder names unet -> diffusion_models and
# clip -> text_encoders; both spellings are scanned, and must not list a file twice.
dm = os.path.join(tmp, "dm")
os.makedirs(dm)
save_file({"x": torch.zeros(1)}, os.path.join(dm, "diffhdr_scan_probe.safetensors"))
folder_paths.add_model_folder_path("diffusion_models", dm)
labels = [c for c in model_scan.list_candidates("dit") if c.endswith("diffhdr_scan_probe.safetensors")]
ok = len(labels) == 1
print(f"  [{'ok ' if ok else 'FAIL'}] probe listed {len(labels)}x: {labels}")
if not ok:
    failures.append(f"duplicate dropdown entries: {labels}")
ok = labels and model_scan.resolve("dit", labels[0]) == os.path.join(dm, "diffhdr_scan_probe.safetensors")
print(f"  [{'ok ' if ok else 'FAIL'}] label resolves back to the file")
if not ok:
    failures.append("label did not resolve")

print("tokenizer search follows extra model paths and symlinks, and survives a cycle")
lib = os.path.join(tmp, "shared_lib", "Wan2.1-VACE-14B", "google", "umt5-xxl")
os.makedirs(lib)
open(os.path.join(lib, "spiece.model"), "wb").close()
linkroot = os.path.join(tmp, "linkroot")
os.makedirs(linkroot)
try:
    os.symlink(os.path.join(tmp, "shared_lib"), os.path.join(linkroot, "wan"))
    os.symlink(linkroot, os.path.join(linkroot, "loop"))
    folder_paths.add_model_folder_path("diffhdr", linkroot)
    toks = model_scan.find_tokenizers()
    ok = any(os.path.realpath(t) == os.path.realpath(lib) for t in toks)
    print(f"  [{'ok ' if ok else 'FAIL'}] found via a symlink in a registered folder: {toks}")
    if not ok:
        failures.append("tokenizer behind a symlink in a registered folder not found")
except OSError as exc:  # no symlink permission (Windows without developer mode)
    print(f"  [skip] symlinks unavailable ({exc})")
shutil.rmtree(tmp, ignore_errors=True)

print("tokenizer auto-detection on this install")
toks = [t for t in model_scan.find_tokenizers() if not t.startswith(tmp)]
print(f"  [{'ok ' if toks else 'skip'}] found {len(toks)}: {toks[:2]}"
      + ("" if toks else " (none installed -- the loader will say how to fetch one)"))

print()
if failures:
    print("FAILED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all model-scan checks passed")
