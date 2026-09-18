"""Import the vendored upstream tree without claiming the global ``diffsynth`` name.

Upstream is an ordinary Python package at ``../DiffHDR/diffsynth``. The obvious way
in is ``sys.path.insert(0, ".../DiffHDR")`` followed by ``import diffsynth`` -- which
is what the other wrapper packs in this suite do for their upstreams. It has a known
failure mode: sys.path is only consulted for names that are not already in
sys.modules, so if any other custom node vendors or pip-installs something called
``diffsynth`` and imports it first, our ``import diffsynth`` silently resolves to
THEIR copy and the pipeline breaks in ways that look like a model bug.

Nothing else in this install ships a ``diffsynth`` today, but custom-node sets are
not stable and the failure is silent, so we do not rely on that. Every self-import
inside the vendored tree is relative (verified at vendoring time -- there is not one
``from diffsynth.x import y`` in it), which means the package works perfectly well
under a different top-level name. We load it as ``diffhdr_vendored_diffsynth``:

  * we never claim, shadow, or depend on the global ``diffsynth`` name;
  * another pack's ``diffsynth`` cannot shadow ours;
  * ComfyUI's own ``utils`` package is untouched (see DiffHDR/VENDORED.md).

The import is deferred to first use, not done at module import, because it costs
roughly a second and pulls in torch.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading

logger = logging.getLogger("ComfyUI-DiffHDR")

_PRIVATE_NAME = "diffhdr_vendored_diffsynth"
_UPSTREAM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DiffHDR"
)
_PACKAGE_DIR = os.path.join(_UPSTREAM_DIR, "diffsynth")

_LOCK = threading.Lock()
_module = None
_import_error: BaseException | None = None


def upstream_dir() -> str:
    """Absolute path of the vendored upstream repo root (the dir holding ``diffsynth/``)."""
    return _UPSTREAM_DIR


def diffsynth():
    """Return the vendored diffsynth package, importing it on first call.

    A failed import is cached and re-raised rather than retried, so a broken
    install reports the same clear error on every graph run instead of paying the
    import cost again each time.
    """
    global _module, _import_error
    with _LOCK:
        if _module is not None:
            return _module
        if _import_error is not None:
            raise RuntimeError(
                "ComfyUI-DiffHDR: the vendored diffsynth package failed to import "
                f"earlier in this session ({_import_error!r}). Fix the cause and "
                "restart ComfyUI."
            ) from _import_error
        try:
            _module = _load()
        except BaseException as exc:  # noqa: BLE001 - cached and re-raised verbatim
            _import_error = exc
            raise
        return _module


def _load():
    if _PRIVATE_NAME in sys.modules:
        return sys.modules[_PRIVATE_NAME]

    init_py = os.path.join(_PACKAGE_DIR, "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError(
            "ComfyUI-DiffHDR: the vendored upstream tree is missing "
            f"({init_py} not found). This pack vendors DiffHDR rather than cloning "
            "it at runtime, so the directory should be present in the repo -- a "
            "missing tree usually means a partial checkout or an rsync deploy that "
            "excluded it. Re-deploy the pack."
        )

    spec = importlib.util.spec_from_file_location(
        _PRIVATE_NAME, init_py, submodule_search_locations=[_PACKAGE_DIR]
    )
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module: the package's own relative imports resolve
    # through sys.modules while it is still executing.
    sys.modules[_PRIVATE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_PRIVATE_NAME, None)
        raise

    logger.info("ComfyUI-DiffHDR: vendored diffsynth loaded as %r", _PRIVATE_NAME)
    return module


def pipeline_classes():
    """Return ``(WanVideoPipeline, ModelConfig)`` from the vendored tree."""
    diffsynth()  # registers the parent package under its private name
    mod = importlib.import_module(f"{_PRIVATE_NAME}.pipelines.wan_video_new")
    return mod.WanVideoPipeline, mod.ModelConfig


def model_loader_configs():
    """Return upstream's model-detection table (hash -> model names).

    Used by ``model_scan`` to validate a checkpoint from its safetensors header
    before committing to a multi-gigabyte load.
    """
    diffsynth()  # registers the parent package under its private name
    mod = importlib.import_module(f"{_PRIVATE_NAME}.configs.model_config")
    return mod.model_loader_configs
