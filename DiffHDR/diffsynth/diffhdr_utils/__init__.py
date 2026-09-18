"""Upstream DiffHDR ``utils/`` modules, relocated.

Upstream imports these as a TOP-LEVEL ``utils`` package (``from utils.color_utils
import ...``). ComfyUI ships its own ``ComfyUI/utils/`` on sys.path, so honouring
that spelling would either fail or — far worse — shadow ComfyUI core. They are
moved here and the three import sites rewritten to be package-relative instead.
See ../../VENDORED.md.
"""
