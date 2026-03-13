"""Docling provider endpoint - forwards requests to a Docling backend service."""

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.endpoint_utils.forward_request import (
    forward_request,
    resolve_deployment,
    run_proxy_pre_call_hook,
)

router = APIRouter()


@router.post(
    "/v1/docling/{subpath:path}",
    tags=["docling"],
)
@router.post(
    "/docling/{subpath:path}",
    tags=["docling"],
)
async def docling_proxy(
    request: Request,
    subpath: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Docling provider pass-through endpoint.

    Forwards requests to the backend Docling service while enforcing
    authentication, virtual key permissions, and RPM limits.

    The sub-path is forwarded as-is, e.g.:
    - /docling/v1/convert/source  ->  {api_base}/v1/convert/source
    - /docling/v1/convert/file    ->  {api_base}/v1/convert/file
    - /docling/v1/chunk           ->  {api_base}/v1/chunk

    Example:
    ```bash
    curl -X POST "http://localhost:4000/docling/v1/convert/source" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "Docling",
            "source": "https://example.com/document.pdf"
        }'
    ```
    """
    body = await request.body()
    data = orjson.loads(body)

    model = data.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' is required")

    await run_proxy_pre_call_hook(user_api_key_dict=user_api_key_dict, data=data)

    # Docling conversions can take several minutes; allow caller to override
    timeout = data.pop("timeout", 300.0)

    litellm_params = await resolve_deployment(model)
    base_url = litellm_params["api_base"].rstrip("/")
    target_url = f"{base_url}/{subpath}"

    # Strip 'model' key from forwarded body since Docling doesn't expect it
    forwarded_data = {k: v for k, v in data.items() if k != "model"}

    return await forward_request(
        target_url=target_url,
        body=forwarded_data,
        timeout=float(timeout),
        api_key=litellm_params.get("api_key"),
    )
