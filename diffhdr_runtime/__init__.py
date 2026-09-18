"""Wrapper-side code for ComfyUI-DiffHDR.

Named ``diffhdr_runtime/`` rather than the more obvious ``src/`` or ``utils/`` on
purpose. Upstream DiffHDR reaches for a top-level ``utils`` package and ComfyUI
ships its own ``ComfyUI/utils/``; a wrapper package sharing either name is a
sys.path collision waiting to happen. Nothing in here is vendored -- the upstream
tree lives under ``../DiffHDR/`` and is reached only through ``vendor.py``.

Submodules are deliberately NOT re-exported here: ``runtime`` and ``vendor`` pull
in torch and the 88 KB pipeline module, and importing those at pack-load time
would add seconds to every ComfyUI start for users who never touch these nodes.
``diffhdr_nodes`` imports what it needs, when it needs it.
"""
