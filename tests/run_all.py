#!/usr/bin/env python
"""Run every CPU-only test suite in this pack.

    cd ComfyUI && env/bin/python custom_nodes/ComfyUI-DiffHDR/tests/run_all.py

Exits non-zero if any suite fails. No GPU and no model weights are required.

Two suites need to be run from inside a ComfyUI install (they import folder_paths):
test_model_scan and test_nodes. From a bare clone of this repo they are skipped, and
the rest -- the colour parity checks, the LoRA merge arithmetic, the window
scheduling, the VRAM dtypes and the pipeline cache -- run anywhere torch is available.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
IN_COMFY = os.path.isfile(os.path.join(COMFY, "folder_paths.py"))
NEEDS_COMFY = {"test_model_scan.py", "test_nodes.py"}
SUITES = [
    ("colour parity", "test_colour_parity.py",
     "the ported log curve and exposure masks vs the vendored upstream originals"),
    ("model scan", "test_model_scan.py",
     "header-only checkpoint identification, the known-bad rejections, dropdowns"),
    ("lora merge", "test_lora_merge.py",
     "all three LoRA encodings, and that DiffHDR + a speed LoRA stay disjoint"),
    ("windows", "test_windows.py",
     "sliding-window scheduling and the temporal blend ramp"),
    ("vram dtypes", "test_vram_dtypes.py",
     "DiT and VACE resident in their own dtype (the vendored VACE fp32-upcast fix)"),
    ("runtime", "test_runtime.py",
     "one resident pipeline, freed for real, rebuilt on demand after eviction"),
    ("nodes", "test_nodes.py",
     "pack import, schemas, Encode/Sample behaviour, the shipped example workflows"),
]

failed, skipped = [], []
for name, script, why in SUITES:
    print(f"\n{'=' * 72}\n{name}  --  {why}\n{'=' * 72}")
    if script in NEEDS_COMFY and not IN_COMFY:
        print(f"  [skip] needs a ComfyUI install (no folder_paths.py at {COMFY})")
        skipped.append(name)
        continue
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       capture_output=True, text=True)
    out = "\n".join(l for l in r.stdout.splitlines()
                    if not l.startswith(("WARNING", "INFO:", "  import pynvml"))
                    and "FutureWarning" not in l)
    print(out)
    if r.returncode != 0:
        failed.append(name)
        print(r.stderr[-2000:], file=sys.stderr)

print(f"\n{'=' * 72}")
if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
ran = len(SUITES) - len(skipped)
print(f"all {ran} suites passed" + (f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""))
