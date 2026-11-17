"""Pipeline cache and handle semantics. No GPU, no weights: the build is stubbed.

What this guards is the interaction with ComfyUI's own output cache. ComfyUI keeps the
Load Model node's output (the handle) and will not re-run that node while its widgets
are unchanged, so:

* evicting the pipeline must actually free it -- the handle must not hold it;
* a handle whose pipeline was evicted (DiffHDR Free Memory, or a second loader taking
  the single slot) must rebuild on demand. In the first release it raised instead, and
  because ComfyUI would not re-run the cached loader, every later run failed until a
  loader widget was changed.
"""

import gc
import os
import sys
import weakref

_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PACK)

from diffhdr_runtime import runtime  # noqa: E402

fails = []


def ck(name, cond, extra=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


class FakePipe:
    pass


builds = []


def fake_build(**kw):
    builds.append(os.path.basename(kw["lora_path"]))
    return FakePipe(), 80, (607 if kw.get("speed_path") else 0)


runtime._build = fake_build


def handle(key, lora, speed=None):
    return runtime.DiffHDRPipeline((key,), dict(lora_path=f"/models/{lora}", speed_path=speed))


print("build once, then reuse")
h1 = handle("video", "DiffHDR.safetensors")
ck("not built until asked", h1.released and not builds)
p = h1.pipe
ck("first access builds", builds == ["DiffHDR.safetensors"])
ck("second access reuses", h1.pipe is p and len(builds) == 1)
ck("merge counts reach the handle", h1.lora_modules_merged == 80)
ck("an identical handle shares the entry", handle("video", "DiffHDR.safetensors").pipe is p
   and len(builds) == 1)

print("free() really frees: the handle holds no reference")
ref = weakref.ref(p)
del p
runtime.free()
gc.collect()
ck("pipeline object collected", ref() is None)
ck("handle reports released", h1.released)

print("a released handle rebuilds instead of failing")
p = h1.pipe
ck("rebuilt on demand", len(builds) == 2 and isinstance(p, FakePipe) and not h1.released)

print("a second loader evicts the first; each rebuilds when used again")
h2 = handle("pano", "DiffHDR_Pano.safetensors", speed="/models/lightx2v.safetensors")
ref = weakref.ref(p)
del p
p2 = h2.pipe
gc.collect()
ck("second pipeline built", builds[-1] == "DiffHDR_Pano.safetensors")
ck("first evicted and collected (never two resident)", h1.released and ref() is None)
ck("speed LoRA count reaches its handle", h2.speed_modules_merged == 607)
h1.pipe
ck("first rebuilds, second is evicted in turn", builds[-1] == "DiffHDR.safetensors" and h2.released)
ck("cache never holds more than one", len(runtime._CACHE) == 1)

print()
if fails:
    print("FAILED: " + ", ".join(fails))
    sys.exit(1)
print("all runtime checks passed")
