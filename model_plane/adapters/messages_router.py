"""Anthropic-compatible /v1/messages endpoint (translated through LiteLLM)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from model_plane.adapters.auth import verify_api_key
from model_plane.adapters.executor import aexecute_with_fallback
from model_plane.logging_setup import get_logger
from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["messages"])


@router.post("/messages")
async def messages(
    request: Request,
    _auth: Annotated[str, Depends(verify_api_key)],
) -> Any:
    body: dict = await request.json()
    request_id = f"mp-{uuid.uuid4().hex[:12]}"

    # Convert Anthropic messages format to OpenAI format for LiteLLM
    oai_messages = _anthropic_to_openai_messages(body)

    oai_body: dict = {
        "messages": oai_messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 1.0),
        "stream": body.get("stream", False),
    }
    if body.get("tools"):
        oai_body["tools"] = body["tools"]

    ctx = build_routing_context(
        oai_body,
        request_id=request_id,
        tenant_id=request.headers.get("X-Tenant-Id"),
        session_id=request.headers.get("X-Session-Id"),
        agent_type="anthropic_client",
    )

    try:
        decision = run_routing_pipeline(ctx)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Routing failed") from exc

    fallback_chain = decision.deployment.fallback_to or []
    try:
        response = await aexecute_with_fallback(ctx, fallback_chain)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model execution failed: {exc}") from exc

    return JSONResponse(
        content=_openai_to_anthropic_response(response, body.get("model", "claude-3-haiku")),
        headers={"X-Request-Id": request_id, "X-Deployment": decision.deployment.name},
    )


def _anthropic_to_openai_messages(body: dict) -> list[dict]:
    msgs = []
    if body.get("system"):
        msgs.append({"role": "system", "content": body["system"]})
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            # flatten text blocks
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        msgs.append({"role": msg["role"], "content": content})
    return msgs


def _openai_to_anthropic_response(response: Any, original_model: str) -> dict:
    """Minimal translation of OpenAI response format to Anthropic format."""
    try:
        choice = response.choices[0]
        content_text = choice.message.content or ""
        usage = response.usage
        return {
            "id": f"msg_{response.id}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content_text}],
            "model": original_model,
            "stop_reason": choice.finish_reason or "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        }
    except Exception:
        return {"type": "message", "role": "assistant", "content": [], "model": original_model}
