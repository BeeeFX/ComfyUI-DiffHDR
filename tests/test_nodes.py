"""Load the pack the way ComfyUI loads it and exercise everything that needs no GPU.

Covers node registration, schema sanity, and a full Encode pass on synthetic frames.
Sampling needs weights and a GPU, so it is not covered here -- see README "Verifying".
"""

import importlib.util
import json
import os
import sys
import uuid

import numpy as np
import torch

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.path.dirname(os.path.dirname(_PACK))
sys.path.insert(0, _COMFY)

# comfy.model_management initialises CUDA at import unless told otherwise, which on a
# machine with no GPU is a crash rather than a skip. This suite needs no GPU.
if not torch.cuda.is_available():
    import comfy.cli_args  # noqa: E402

    comfy.cli_args.args.cpu = True

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
cond, over, under, final = enc.encode(img, "video", 160, 128, "crop_and_resize", 0.95, 0.01,
                                      False, 0.02, True)
ck("cond carries all frames", cond["frames"] == 5)
ck("log video shape", tuple(cond["log_video"].shape) == (5, 3, 128, 160), str(cond["log_video"].shape))
ck("log video in [0,1]", 0.0 <= float(cond["log_video"].min()) and float(cond["log_video"].max()) <= 1.0)
ck("one PIL mask per frame", len(cond["masks"]) == 5 and cond["masks"][0].mode == "RGB")
ck("MASK layout is (T,H,W)", tuple(over.shape) == (5, 128, 160), str(over.shape))
ck("blown patch detected", float(over[:, :20, :30].mean()) > 0.9)
ck("crushed patch detected", float(under[:, 40:60, 40:70].mean()) > 0.9)
ck("final mask output is what the model is sent",
   all(torch.equal(torch.from_numpy(np.array(p)[..., 0] / 255.0).float(), final[i])
       for i, p in enumerate(cond["masks"])))
ck("final == over when under-exposure is off", torch.equal(final, over))
ck("the incoming IMAGE is not modified", float(img[:, 40:60, 40:70].abs().max()) == 0.0)

print("Encode is the whole-clip maths, done a frame at a time")
ref_log = nodes.colour.lin_to_log(nodes.colour.srgb_to_linear(img.permute(0, 3, 1, 2).contiguous()))
ck("log video bit-identical to the whole-clip conversion", torch.equal(cond["log_video"], ref_log))

print("Encode: under-exposure mode rewrites the conditioning to mid-grey")
cond_u, _, _, final_u = enc.encode(img, "video", 160, 128, "crop_and_resize", 0.95, 0.01,
                                   True, 0.02, True)
mid = nodes.colour.lin_to_log(nodes.colour.srgb_to_linear(torch.tensor(0.5)))
ck("crushed region moved to mid-grey in log",
   torch.allclose(cond_u["log_video"][:, :, 45:55, 45:65], mid, atol=1e-5))

print("Encode: panorama mode produces an over-only mask")
cond_p, over_p, under_p, _ = enc.encode(img[:1], "panorama", 160, 128, "crop_and_resize",
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

print("Encode: resize_mode=none generates at the frames' own size, not the widgets'")
cond_n, *_ = enc.encode(torch.rand(2, 96, 176, 3), "video", 1280, 720, "none", 0.95, 0.01,
                        False, 0.02, True)
ck("height/width taken from the frames", (cond_n["height"], cond_n["width"]) == (96, 176),
   str((cond_n["height"], cond_n["width"])))

print("Encode: a user mask adds to, subtracts from, or replaces the detection")
user = torch.zeros(1, 128, 160)
user[0, 100:120, 100:150] = 1.0  # somewhere the detector found nothing
user[0, :10, :10] = 1.0          # inside the blown patch
args = (img, "video", 160, 128, "crop_and_resize", 0.95, 0.01, False, 0.02, True)
_, det, _, f_add = enc.encode(*args, mask=user, mask_mode="add")
_, _, _, f_sub = enc.encode(*args, mask=user, mask_mode="subtract")
_, _, _, f_rep = enc.encode(*args, mask=user, mask_mode="replace")
ub = user.bool().expand(5, -1, -1)
ck("add = detection | mask", torch.equal(f_add.bool(), det.bool() | ub))
ck("subtract = detection & ~mask", torch.equal(f_sub.bool(), det.bool() & ~ub))
ck("replace = mask", torch.equal(f_rep.bool(), ub))
ck("detected over_mask output is unaffected by the user mask", torch.equal(det, over))
_, _, _, f_small = enc.encode(*args, mask=torch.nn.functional.interpolate(
    user[None], size=(64, 80), mode="nearest")[0], mask_mode="replace")
ck("a mask of another size is conformed like the frames", tuple(f_small.shape) == (5, 128, 160))
try:
    enc.encode(*args, mask=torch.zeros(3, 128, 160))
    ck("mask frame count mismatch rejected", False, "was accepted")
except ValueError:
    ck("mask frame count mismatch rejected", True)

print("Encode: mid-grey fill only where the model reconstructs")
protect = torch.zeros(1, 128, 160)
protect[0, 40:60, 40:70] = 1.0  # the crushed patch
cond_s, _, _, _ = enc.encode(img, "video", 160, 128, "crop_and_resize", 0.95, 0.01,
                             True, 0.02, True, mask=protect, mask_mode="subtract")
black = nodes.colour.lin_to_log(nodes.colour.srgb_to_linear(torch.tensor(0.0)))
ck("a subtracted crushed region keeps its plate value, not grey",
   torch.allclose(cond_s["log_video"][:, :, 45:55, 45:65], black, atol=1e-6))

print("Sample threads its settings into the pipeline correctly (fake pipe, no GPU)")


class FakePipe:
    """Records the kwargs the node hands to diffsynth, and returns log-space frames."""

    device = "cpu"

    def __init__(self):
        self.seen = None
        self.calls = []

    def __call__(self, **kw):
        self.seen = kw
        self.calls.append(kw)
        n, h, w = kw["num_frames"], kw["height"], kw["width"]
        list(kw["progress_bar_cmd"](range(kw["num_inference_steps"])))
        return [torch.full((h, w, 3), 0.5) for _ in range(n)]


class FakeHandle:
    def __init__(self, pipe):
        self.pipe = pipe


fp = FakePipe()
smp = nodes.DiffHDRSample()
small = torch.rand(2, 32, 48, 3)
cond2, *_ = enc.encode(small, "video", 48, 32, "crop_and_resize", 0.95, 0.01, False, 0.02, True)

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

ck("seed defaults to a fixed control, so re-queues hit the cache",
   nodes.DiffHDRSample.INPUT_TYPES()["required"]["seed"][1].get("control_after_generate") == "fixed")

print("Sample logs tokens, time and peak VRAM per pass on CUDA (stubbed: no GPU here)")
import logging  # noqa: E402

records = []
handler = logging.Handler()
handler.emit = lambda r: records.append(r.getMessage())
nodes.logger.addHandler(handler)
_saved = (nodes._on_cuda, torch.cuda.reset_peak_memory_stats,
          torch.cuda.max_memory_allocated, torch.cuda.get_device_properties)
try:
    nodes._on_cuda = lambda device: True
    torch.cuda.reset_peak_memory_stats = lambda device=None: None
    torch.cuda.max_memory_allocated = lambda device=None: 17 * 2**30
    torch.cuda.get_device_properties = lambda device=None: type("P", (), {"total_memory": 24 * 2**30})()
    smp._run(FakePipe(), cond2["log_video"], cond2["masks"], None, "", "", 2, 1.0, 7, 5.0,
             32, 48, 5, True, label="probe")
finally:
    (nodes._on_cuda, torch.cuda.reset_peak_memory_stats,
     torch.cuda.max_memory_allocated, torch.cuda.get_device_properties) = _saved
    nodes.logger.removeHandler(handler)
ck("tokens logged", any("(12 tokens)" in r for r in records), str(records[:1]))
ck("peak VRAM logged", any("peak VRAM 17.0 of 24.0 GiB" in r for r in records), str(records[-1:]))

print("Sample: a still is replicated to the trained token count, capped by window_size")
still, *_ = enc.encode(torch.rand(1, 720, 1280, 3), "video", 1280, 720, "crop_and_resize",
                       0.95, 0.01, False, 0.02, True)
fp.calls.clear()
smp.sample(FakeHandle(fp), still, "", "", 2, 1.0, 7, 5.0, 33, 16, True, 0.0, True)
ck("window_size 33 -> the trained 33 frames", fp.calls[-1]["num_frames"] == 33,
   str(fp.calls[-1]["num_frames"]))
smp.sample(FakeHandle(fp), still, "", "", 2, 1.0, 7, 5.0, 9, 8, True, 0.0, True)
ck("window_size 9 -> 9 frames (fits 24 GB)", fp.calls[-1]["num_frames"] == 9,
   str(fp.calls[-1]["num_frames"]))

print("Sample: a batch of panoramas is N single-frame passes, not a clip")
panos, *_ = enc.encode(torch.rand(3, 32, 64, 3), "panorama", 64, 32, "crop_and_resize",
                       0.95, 0.01, False, 0.02, True)
fp.calls.clear()
(res,) = smp.sample(FakeHandle(fp), panos, "", "", 2, 1.0, 7, 5.0, 33, 16, True, 0.0, True)
ck("one pass per panorama", len(fp.calls) == 3, str(len(fp.calls)))
ck("each pass is a single frame", all(c["num_frames"] == 1 for c in fp.calls))
ck("all panoramas returned", tuple(res.shape) == (3, 32, 64, 3), str(tuple(res.shape)))

print("Sample: long clips run as windows, blended back to full length")
clip, *_ = enc.encode(torch.rand(40, 32, 48, 3), "video", 48, 32, "crop_and_resize",
                      0.95, 0.01, False, 0.02, True)
fp.calls.clear()
(res,) = smp.sample(FakeHandle(fp), clip, "", "", 2, 1.0, 7, 5.0, 17, 8, True, 0.0, True)
ck("several windows", len(fp.calls) > 1, str(len(fp.calls)))
ck("output length matches the input", res.shape[0] == 40, str(tuple(res.shape)))
ck("later windows get the previous window's output as reference",
   fp.calls[0]["vace_reference_image"] is None
   and all(c["vace_reference_image"] is not None for c in fp.calls[1:]))

print("Free Memory passes images through, so it can be ordered after Sample")
fm = nodes.DiffHDRFreeMemory
ck("optional images input", "images" in fm.INPUT_TYPES().get("optional", {}))
ck("images is appended after status (saved links stay valid)",
   fm.RETURN_TYPES == ("STRING", "IMAGE") and fm.RETURN_NAMES[0] == "status")
passthrough = torch.rand(1, 8, 8, 3)
ck("disabled: no-op, still passes through", fm().free(False, passthrough)[1] is passthrough)
ck("IS_CHANGED takes the optional input", fm.IS_CHANGED(trigger=True, images=passthrough) != 0)

print("example workflows: widget arrays line up with what the frontend builds")
# The frontend serialises one value per widget, in input order: required, then
# optional, and an extra 'control_after_generate' value straight after any INT that
# has one (every input called 'seed' gets one by default). Missing that is what
# shifted every Sample value after 'seed' by one in the first release.


def frontend_widgets(cls):
    names = []
    spec_ = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, entry in spec_.get(section, {}).items():
            t = entry[0]
            opts = entry[1] if len(entry) > 1 else {}
            if isinstance(t, str) and t.isupper() and t not in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                continue  # a socket, not a widget
            names.append(name)
            if t == "INT" and opts.get("control_after_generate", name in ("seed", "noise_seed")):
                names.append("control_after_generate")
    return names


wf_dir = os.path.join(_PACK, "example_workflows")
for fname in sorted(os.listdir(wf_dir)):
    wf = json.load(open(os.path.join(wf_dir, fname)))
    try:
        uuid.UUID(wf.get("id", ""))
        ok_id = True
    except ValueError:
        ok_id = False
    ck(f"{fname}: id is a UUID (the frontend's schema requires one)", ok_id, repr(wf.get("id")))
    for nd in wf["nodes"]:
        cls = pack.NODE_CLASS_MAPPINGS.get(nd["type"])
        if cls is None:
            continue
        names = frontend_widgets(cls)
        vals = nd["widgets_values"]
        ok = len(vals) == len(names)
        ck(f"{fname}/{nd['type']}: {len(vals)} values for {len(names)} widgets", ok)
        if ok and nd["type"] == "DiffHDR_Sample":
            v = dict(zip(names, vals))
            ck(f"{fname}: Sample values land on the right widgets",
               v["sigma_shift"] == 5.0 and v["control_after_generate"] in ("fixed", "randomize")
               and v["window_size"] % 4 == 1 and 0.0 <= v["tea_cache"] <= 0.5, str(v))

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all node checks passed")
