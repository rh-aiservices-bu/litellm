"""Shared utilities for forwarding requests to backend services."""

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, Response

from litellm.proxy._types import UserAPIKeyAuth


async def run_proxy_pre_call_hook(
    user_api_key_dict: UserAPIKeyAuth,
    data: dict,
) -> None:
    """Run the proxy pre-call hook to enforce RPM limits and other pre-call checks."""
    from litellm.proxy.proxy_server import proxy_logging_obj

    await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=data,
        call_type="pass_through_endpoint",
    )


async def resolve_deployment(model_name: str) -> Dict[str, Any]:
    """Look up the model's deployment via the router and return its litellm_params.

    The returned dict contains at least 'api_base' and 'model', and optionally
    'api_key' plus any other provider-specific params from the deployment config.
    """
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")

    try:
        deployment: Optional[Dict[str, Any]] = await llm_router.async_get_available_deployment(
            model=model_name,
            request_kwargs={},
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"No available deployment found for model '{model_name}': {e}",
        )

    if deployment is None:
        raise HTTPException(
            status_code=400,
            detail=f"No available deployment found for model '{model_name}'",
        )

    litellm_params = deployment.get("litellm_params", {})

    if not litellm_params.get("api_base"):
        raise HTTPException(
            status_code=500,
            detail=f"Deployment for model '{model_name}' has no api_base configured",
        )

    return litellm_params


def get_backend_model(litellm_params: Dict[str, Any]) -> str:
    """Extract the real backend model name from litellm_params, stripping the provider prefix."""
    model = litellm_params.get("model", "")
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def strip_v1_suffix(api_base: str) -> str:
    """Strip trailing /v1 from api_base URL.

    Handles http://host:8000/v1, http://host:8000/v1/, and http://host:8000.
    """
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


async def forward_request(
    target_url: str,
    body: dict,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> Response:
    """Forward a POST request to the target URL and return the upstream response."""
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(target_url, json=body, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error forwarding request to {target_url}: {e}",
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
