"""Re-export shim — real implementation lives in model_plane.provider_creds.

Kept here so any existing imports of model_plane.routing.credentials continue
to work without changes.
"""
from model_plane.provider_creds import provider_has_creds  # noqa: F401
