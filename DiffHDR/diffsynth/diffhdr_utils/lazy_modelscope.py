"""Deferred ``modelscope`` import.

``diffsynth/models/downloader.py`` is pulled in at import time by ``model_manager``,
and it imports ``snapshot_download`` at module scope. Every code path that would
call it is dead for this nodepack -- the loader always resolves explicit local paths
through ``folder_paths`` -- so requiring the (large) ``modelscope`` package just to
import the pipeline would be a dependency for nothing. This shim defers the import
to the call, so ``modelscope`` is only needed if someone actually triggers a
ModelScope download. See ../../VENDORED.md.
"""


def snapshot_download(*args, **kwargs):
    from modelscope import snapshot_download as _real

    return _real(*args, **kwargs)
