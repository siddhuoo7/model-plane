"""Adapters package."""
from model_plane.adapters.executor import aexecute, aexecute_with_fallback, stream_response

__all__ = ["aexecute", "aexecute_with_fallback", "stream_response"]
