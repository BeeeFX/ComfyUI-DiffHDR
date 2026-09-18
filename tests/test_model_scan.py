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
import torch  # noqa: E402

from diffhdr_runtime import vendor  # noqa: E402

vendor.diffsynth()
import importlib  # noqa: E402

up = importlib.import_module("diffhdr_vendored_diffsynth.models.utils")
probe = os.path.join(M, "vae/wan_2.1_vae.safetensors")
if os.path.exists(probe):
    ks = model_scan.read_key_shapes(probe)
    fake_sd = {k: torch.empty(tuple(s), dtype=torch.uint8, device="meta") for k, s in ks.items()}
    for with_shape in (True, False):
        mine = model_scan._hash(ks, with_shape)
        theirs = up.hash_state_dict_keys(fake_sd, with_shape=with_shape)
        ok = mine == theirs
        print(f"  [{'ok ' if ok else 'FAIL'}] with_shape={with_shape}: {mine}")
        if not ok:
            failures.append(f"hash mismatch with_shape={with_shape}: {mine} vs {theirs}")

print("tokenizer auto-detection")
toks = model_scan.find_tokenizers()
print(f"  [{'ok ' if toks else 'FAIL'}] found {len(toks)}: {toks[:2]}")
if not toks:
    failures.append("no umt5 tokenizer found")

print()
if failures:
    print("FAILED:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all model-scan checks passed")
