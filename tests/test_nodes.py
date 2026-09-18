"""Load the pack the way ComfyUI loads it and exercise everything that needs no GPU.

Covers node registration, schema sanity, and a full Encode pass on synthetic frames.
Sampling needs weights and a GPU, so it is not covered here -- see README "Verifying".
"""

import importlib.util
import os
import sys

import torch

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.path.dirname(os.path.dirname(_PACK))
sys.path.insert(0, _COMFY)

fails = []


def ck(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


print("pack imports the way ComfyUI imports it (hyphenated dir, not an identifier)")
spec = importlib.util.spec_from_file_location(
    "ComfyUI_DiffHDR_test", os.path.join(_PACK, "__init__.py"),
    submodule_search_locations=[_PACK],
)
pack = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_DiffHDR_test"] = pack
spec.loader.exec_module(pack)
ck("imported", True)
ck("global 'diffsynth' not claimed at import", "diffsynth" not in sys.modules)

print("node registration")
ck("4 nodes registered", len(pack.NODE_CLASS_MAPPINGS) == 4, str(list(pack.NODE_CLASS_MAPPINGS)))
ck("every class has a display name",
   set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS))
ck("WEB_DIRECTORY set", pack.WEB_DIRECTORY == "./web")

print("schemas")
for key, cls in pack.NODE_CLASS_MAPPINGS.items():
    schema = cls.INPUT_TYPES()
    ok = (
        isinstance(schema, dict) and "required" in schema
        and hasattr(cls, "RETURN_TYPES") and hasattr(cls, "FUNCTION")
        and hasattr(cls, "CATEGORY") and hasattr(cls, "DESCRIPTION")
        and hasattr(cls, cls.FUNCTION)
        and len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    )
    ck(f"{key}", ok)

print("help docs exist for every node")
for key in pack.NODE_CLASS_MAPPINGS:
    ck(f"web/docs/{key}.md", os.path.isfile(os.path.join(_PACK, "web", "docs", f"{key}.md")))

print("dtype list matches the runtime's")
from ComfyUI_DiffHDR_test.diffhdr_runtime import runtime  # noqa: E402

ck("DIT_DTYPES in step with runtime.DTYPES",
   list(pack.diffhdr_nodes.DIT_DTYPES) == list(runtime.DTYPES),
   f"{pack.diffhdr_nodes.DIT_DTYPES}")
ck("vram modes in step", set(pack.diffhdr_nodes.DiffHDRLoadModel.INPUT_TYPES()
   ["required"]["vram_mode"][0]) == set(runtime.VRAM_MODES))

print("frame-count helpers")
nodes = pack.diffhdr_nodes
ck("4k+1 rounding", [nodes._round_frames(n) for n in (1, 2, 5, 33, 34, 80)] == [1, 5, 5, 33, 37, 81])
ck("still at 720x1280 -> 33 frames (the trained token count)",
   nodes._frames_for_still(720, 1280) == 33, str(nodes._frames_for_still(720, 1280)))
ck("still at higher res -> fewer frames",
   nodes._frames_for_still(1024, 2048) < 33, str(nodes._frames_for_still(1024, 2048)))

print("Encode on synthetic frames")
enc = nodes.DiffHDREncode()
img = torch.rand(5, 128, 160, 3) * 0.5
img[:, :20, :30, :] = 1.0    # blown patch
img[:, 40:60, 40:70, :] = 0.0  # crushed patch
cond, over, under = enc.encode(img, "video", 160, 128, "crop_and_resize", 0.95, 0.01,
                               False, 0.02, True)
ck("cond carries all frames", cond["frames"] == 5)
ck("log video shape", tuple(cond["log_video"].shape) == (5, 3, 128, 160), str(cond["log_video"].shape))
ck("log video in [0,1]", 0.0 <= float(cond["log_video"].min()) and float(cond["log_video"].max()) <= 1.0)
ck("one PIL mask per frame", len(cond["masks"]) == 5 and cond["masks"][0].mode == "RGB")
ck("MASK layout is (T,H,W)", tuple(over.shape) == (5, 128, 160), str(over.shape))
ck("blown patch detected", float(over[:, :20, :30].mean()) > 0.9)
ck("crushed patch detected", float(under[:, 40:60, 40:70].mean()) > 0.9)

print("Encode: under-exposure mode rewrites the conditioning to mid-grey")
cond_u, _, _ = enc.encode(img, "video", 160, 128, "crop_and_resize", 0.95, 0.01,
                          True, 0.02, True)
mid = nodes.colour.lin_to_log(nodes.colour.srgb_to_linear(torch.tensor(0.5)))
ck("crushed region moved to mid-grey in log",
   torch.allclose(cond_u["log_video"][:, :, 45:55, 45:65], mid, atol=1e-5))

print("Encode: panorama mode produces an over-only mask")
cond_p, over_p, under_p = enc.encode(img[:1], "panorama", 160, 128, "crop_and_resize",
                                     0.95, 0.01, False, 0.02, True)
ck("panorama under mask is empty", float(under_p.max()) == 0.0)
ck("panorama over mask is not", float(over_p.max()) > 0.0)

print("Encode rejects bad geometry")
for args, why in [
    ((img, "video", 163, 128, "crop_and_resize", 0.95, 0.01, False, 0.02, True), "width not /16"),
    ((img, "video", 160, 130, "crop_and_resize", 0.95, 0.01, False, 0.02, True), "height not /16"),
]:
    try:
        enc.encode(*args)
        ck(why, False, "was accepted")
    except ValueError:
        ck(why, True)
try:
    enc.encode(torch.rand(1, 130, 163, 3), "video", 160, 128, "none", 0.95, 0.01, False, 0.02, True)
    ck("resize_mode=none with odd input", False, "was accepted")
except ValueError:
    ck("resize_mode=none with odd input", True)

print("Sample threads its settings into the pipeline correctly (fake pipe, no GPU)")


class FakePipe:
    """Records the kwargs the node hands to diffsynth, and returns log-space frames."""

    device = "cpu"

    def __init__(self):
        self.seen = None

    def __call__(self, **kw):
        self.seen = kw
        n, h, w = kw["num_frames"], kw["height"], kw["width"]
        return [torch.full((h, w, 3), 0.5) for _ in range(n)]


class FakeHandle:
    def __init__(self, pipe):
        self.pipe = pipe


fp = FakePipe()
smp = nodes.DiffHDRSample()
small = torch.rand(2, 32, 48, 3)
cond2, _, _ = enc.encode(small, "video", 48, 32, "crop_and_resize", 0.95, 0.01, False, 0.02, True)

(res,) = smp.sample(FakeHandle(fp), cond2, "", "", 4, 1.0, 7, 5.0, 33, 16, True, 0.0, True)
ck("tea_cache=0 becomes None, not a 0.0 threshold", fp.seen["tea_cache_l1_thresh"] is None)
ck("the two HDR flags are always set", fp.seen["srgb_to_lg"] is True and fp.seen["is_hdr"] is True)
ck("seed/steps/cfg/shift forwarded",
   (fp.seen["seed"], fp.seen["num_inference_steps"], fp.seen["cfg_scale"],
    fp.seen["sigma_shift"]) == (7, 4, 1.0, 5.0))
ck("output is decoded out of log space",
   torch.allclose(res, nodes.colour.log_to_lin(torch.tensor(0.5)), atol=1e-4),
   f"got {float(res.mean()):.3f}")

(res,) = smp.sample(FakeHandle(fp), cond2, "", "", 4, 1.0, 7, 5.0, 33, 16, True, 0.12, True)
ck("tea_cache>0 is forwarded as the threshold", fp.seen["tea_cache_l1_thresh"] == 0.12)
ck("tea_cache model id is the 14B one", fp.seen["tea_cache_model_id"] == "Wan2.1-T2V-14B")

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all node checks passed")
