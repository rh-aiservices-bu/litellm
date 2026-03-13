"""Tokenize endpoint - forwards requests to VLLM's /tokenize endpoint."""

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.endpoint_utils.forward_request import (
    forward_request,
    get_backend_model,
    resolve_deployment,
    run_proxy_pre_call_hook,
    strip_v1_suffix,
)

router = APIRouter()


@router.post(
    "/v1/tokenize",
    tags=["tokenize"],
)
@router.post(
    "/tokenize",
    tags=["tokenize"],
)
async def tokenize(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Tokenize endpoint - forwards requests to the backend tokenize service (e.g. VLLM).

    Resolves the model's deployment api_base, strips /v1 suffix, and forwards
    the request to {base}/tokenize.

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/tokenize" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "openai/my-vllm-model",
            "prompt": "Hello world"
        }'
    ```
    """
    body = await request.body()
    data = orjson.loads(body)

    model = data.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' is required")

    await run_proxy_pre_call_hook(user_api_key_dict=user_api_key_dict, data=data)

    litellm_params = await resolve_deployment(model)
    base_url = strip_v1_suffix(litellm_params["api_base"])
    target_url = f"{base_url}/tokenize"

    data["model"] = get_backend_model(litellm_params)

    return await forward_request(target_url=target_url, body=data, api_key=litellm_params.get("api_key"))
