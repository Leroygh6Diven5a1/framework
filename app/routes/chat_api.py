from fastapi import APIRouter, Depends, Request
from models import OpenAIRequest
from auth import get_api_key

# 引入运行状态管理器与多通道分发策略
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.headless_proxy import HeadlessProxyUpstream

router = APIRouter()

# 实例化多通道策略
express_upstream = ExpressSDKUpstream()
headless_upstream = HeadlessProxyUpstream()


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    """
    /v1/chat/completions 动态分流路由器
    根据大盘设置的全局开关，决定将 OpenAI 请求路由至：
    - True  -> HeadlessProxyUpstream (无头浏览器反代通道，规避 429 限流)
    - False -> ExpressSDKUpstream (官方 API Key 标准通道)
    """
    if app_state.is_web_proxy_enabled():
        return await headless_upstream.chat_completions(request, fastapi_request)
    else:
        return await express_upstream.chat_completions(request, fastapi_request)
