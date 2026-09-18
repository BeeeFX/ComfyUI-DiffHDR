"""Check the LoRA merge handles every encoding these models actually ship in.

diffsynth's own loader keys off ``.lora_B.`` and so loads *nothing* from a
lora_up/lora_down file -- silently. Since a LoRA that merges into nothing still yields
plausible-looking output, this suite checks both the arithmetic and the failure mode.

It also asserts, against the real files when they are installed, that a step-distill
LoRA and the DiffHDR LoRA target disjoint modules -- which is what makes stacking them
legitimate rather than a collision.
"""

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.path.dirname(os.path.dirname(_PACK))
sys.path.insert(0, _COMFY)
sys.path.insert(0, _PACK)

from diffhdr_runtime import runtime  # noqa: E402

TMP = tempfile.mkdtemp(prefix="diffhdr-lora-test-")
fails = []


def ck(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


def write(tensors, name="l.safetensors"):
    path = os.path.join(TMP, name)
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    return path


class Toy(torch.nn.Module):
    """Two linear layers named like Wan's, so module paths look realistic."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])


def fresh():
    torch.manual_seed(0)
    m = Toy()
    return m, m.blocks[0].weight.detach().clone(), m.blocks[0].bias.detach().clone()


up = torch.randn(8, 4) * 0.1
dn = torch.randn(4, 8) * 0.1
expect_delta = up @ dn

print("encoding 1: PEFT lora_A / lora_B with a .default adapter segment (DiffHDR's own)")
m, w0, _ = fresh()
n = runtime._merge_lora(m, write({"blocks.0.lora_B.default.weight": up,
                                  "blocks.0.lora_A.default.weight": dn}), 1.0, "test")
ck("merged 1 module", n == 1, f"n={n}")
ck("delta is up@dn", torch.allclose(m.blocks[0].weight - w0, expect_delta, atol=1e-6))

print("encoding 2: PEFT without .default")
m, w0, _ = fresh()
n = runtime._merge_lora(m, write({"blocks.0.lora_B.weight": up,
                                  "blocks.0.lora_A.weight": dn}), 1.0, "test")
ck("merged", n == 1 and torch.allclose(m.blocks[0].weight - w0, expect_delta, atol=1e-6))

print("encoding 3: kohya lora_up / lora_down with diffusion_model. prefix (lightx2v's)")
m, w0, _ = fresh()
n = runtime._merge_lora(m, write({"diffusion_model.blocks.0.lora_up.weight": up,
                                  "diffusion_model.blocks.0.lora_down.weight": dn}), 1.0, "test")
ck("prefix stripped and merged", n == 1, f"n={n}")
ck("delta is up@dn", torch.allclose(m.blocks[0].weight - w0, expect_delta, atol=1e-6))

print("encoding 3b: .alpha rescales by alpha/rank")
m, w0, _ = fresh()
n = runtime._merge_lora(m, write({"blocks.0.lora_up.weight": up,
                                  "blocks.0.lora_down.weight": dn,
                                  "blocks.0.alpha": torch.tensor(2.0)}), 1.0, "test")
ck("scaled by alpha/rank", torch.allclose(m.blocks[0].weight - w0,
                                          expect_delta * (2.0 / dn.shape[0]), atol=1e-6))

print("encoding 4: plain additive diff / diff_b (647 of lightx2v's 1053 tensors)")
m, w0, b0 = fresh()
dw, db = torch.randn(8, 8) * 0.05, torch.randn(8) * 0.05
n = runtime._merge_lora(m, write({"blocks.0.diff": dw, "blocks.0.diff_b": db}), 1.0, "test")
ck("weight diff applied", torch.allclose(m.blocks[0].weight - w0, dw, atol=1e-6))
ck("bias diff applied", torch.allclose(m.blocks[0].bias - b0, db, atol=1e-6))

print("low-rank and diff on the same module compose")
m, w0, b0 = fresh()
n = runtime._merge_lora(m, write({"blocks.0.lora_up.weight": up,
                                  "blocks.0.lora_down.weight": dn,
                                  "blocks.0.diff": dw, "blocks.0.diff_b": db}), 1.0, "test")
ck("both applied once", torch.allclose(m.blocks[0].weight - w0, expect_delta + dw, atol=1e-6))
ck("counted as one module", n == 1, f"n={n}")

print("strength scales the merge")
m, w0, _ = fresh()
runtime._merge_lora(m, write({"blocks.0.lora_up.weight": up,
                              "blocks.0.lora_down.weight": dn}), 0.5, "test")
ck("0.5 strength halves the delta",
   torch.allclose(m.blocks[0].weight - w0, expect_delta * 0.5, atol=1e-6))

print("untargeted modules are left alone")
m, _, _ = fresh()
b1 = m.blocks[1].weight.detach().clone()
runtime._merge_lora(m, write({"blocks.0.lora_up.weight": up,
                              "blocks.0.lora_down.weight": dn}), 1.0, "test")
ck("blocks.1 untouched", torch.equal(m.blocks[1].weight, b1))

print("a LoRA that matches nothing is a hard error, not a silent no-op")
m, _, _ = fresh()
try:
    runtime._merge_lora(m, write({"nonexistent.module.lora_up.weight": up,
                                  "nonexistent.module.lora_down.weight": dn}), 1.0, "speed")
    ck("raises on zero matches", False, "it returned quietly")
except RuntimeError as e:
    ck("raises on zero matches", "matched NONE" in str(e))
try:
    runtime._merge_lora(m, write({"not_a_lora": up}), 1.0, "test")
    ck("raises on unrecognisable file", False, "it returned quietly")
except RuntimeError as e:
    ck("raises on unrecognisable file", "no recognisable LoRA tensors" in str(e))

print("real files: DiffHDR and a step-distill LoRA must not target the same modules")
try:
    import folder_paths  # noqa: E402
    M = folder_paths.models_dir
except Exception:
    # Running from a bare clone rather than inside a ComfyUI install. Everything above
    # this point is pure torch and has already run; only the against-real-weights
    # cross-check needs to know where models live.
    M = None
    print("  [skip] not running inside ComfyUI, so no model library to check against")
pairs = [
    ("diffhdr/DiffHDR.safetensors",
     "loras/WAN21/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"),
]
for a, b in (pairs if M else []):
    pa, pb = os.path.join(M, a), os.path.join(M, b)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"  [skip] {os.path.basename(a)} + {os.path.basename(b)} (not installed)")
        continue
    ka = set(runtime._lora_plan(runtime._read_lora(pa)))
    kb = set(runtime._lora_plan(runtime._read_lora(pb)))
    ck("disjoint targets", not (ka & kb),
       f"DiffHDR {len(ka)} modules, speed {len(kb)}, overlap {len(ka & kb)}")
    ck("DiffHDR targets vace_blocks only", all(k.startswith("vace_blocks.") for k in ka))
    ck("speed LoRA avoids vace_blocks", not any(k.startswith("vace_blocks.") for k in kb))

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all LoRA merge checks passed")
