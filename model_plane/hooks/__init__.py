"""Hooks package."""
from model_plane.hooks.litellm_hooks import PostCallHook, PreCallHook, register_hooks

__all__ = ["PreCallHook", "PostCallHook", "register_hooks"]
