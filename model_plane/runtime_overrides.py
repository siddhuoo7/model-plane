"""Runtime override store — shared mutable state for live configuration changes.

Both ``model_plane.adapters.admin_api_router`` and
``model_plane.compression.processor`` need to read the same in-process
compression overrides.  Putting the dict here breaks the circular import that
would arise from processor → admin_api_router.

Usage:
    from model_plane.runtime_overrides import compression_overrides

    # write (admin_api_router PUT /compression/config):
    compression_overrides["enabled"] = False

    # read (processor.py compress()):
    enabled = compression_overrides.get("enabled", settings.compression_enabled)
"""

from __future__ import annotations

# Mutable shared dict — deliberately not frozen.
# Keys are the same as the CompressionConfig API schema.
compression_overrides: dict[str, object] = {}
