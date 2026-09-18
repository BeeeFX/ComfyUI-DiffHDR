#!/usr/bin/env python
"""Run every CPU-only test suite in this pack.

    cd ComfyUI && env/bin/python custom_nodes/ComfyUI-DiffHDR/tests/run_all.py

Exits non-zero if any suite fails. No GPU and no model weights are required.

Two suites need to be run from inside a ComfyUI install (they import folder_paths):
test_model_scan and test_nodes. From a bare clone of this repo they are skipped, and
the rest -- the colour parity checks, the LoRA merge arithmetic and the window
scheduling -- run anywhere torch is available.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    ("colour parity", "test_colour_parity.py",
     "the ported log curve and exposure masks vs the vendored upstream originals"),
    ("model scan", "test_model_scan.py",
     "header-only checkpoint identification, and the known-bad rejections"),
    ("lora merge", "test_lora_merge.py",
     "all three LoRA encodings, and that DiffHDR + a speed LoRA stay disjoint"),
    ("windows", "test_windows.py",
     "sliding-window scheduling and the temporal blend ramp"),
    ("nodes", "test_nodes.py",
     "pack import, node schemas, and a full Encode pass"),
]

failed = []
for name, script, why in SUITES:
    print(f"\n{'=' * 72}\n{name}  --  {why}\n{'=' * 72}")
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
print(f"all {len(SUITES)} suites passed")
