from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from models import OpenAIRequest
from auth import get_api_key

# 引入运行状态管理器与多通道分发策略
from runtime_state import app_state
from upstreams.express_sdk import ExpressSDKUpstream
from upstreams.cookie_proxy import CookieProxyUpstream
from api_helpers import extract_upstream_error, create_openai_error_response

router = APIRouter()

# 实例化多通道策略
express_upstream = ExpressSDKUpstream()
cookie_upstream = CookieProxyUpstream()


@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    """
    /v1/chat/completions 动态分流路由器
    根据大盘设置的全局开关，决定将 OpenAI 请求路由至：
    - True  -> CookieProxyUpstream (Cookie 直连反代通道，规避 429 限流)
    - False -> ExpressSDKUpstream (官方 API Key 标准通道)

    统一异常兜底：把上游抛出的 404/403/400 等如实转成 OpenAI 错误格式，
    避免笼统的 500 Internal Server Error（非流式路径此前直接 raise）。
    统计由 main.py 的中间件按响应状态码统一计入，这里不重复计数。
    """
    try:
        if app_state.is_web_proxy_enabled():
            return await cookie_upstream.chat_completions(request, fastapi_request)
        else:
            return await express_upstream.chat_completions(request, fastapi_request)
    except Exception as e:
        code, msg = extract_upstream_error(e)
        print(f"❌ [路由兜底] 模型 {request.model} 调用失败 | HTTP {code} | {msg[:200]}")
        return JSONResponse(status_code=code, content=create_openai_error_response(code, msg, "upstream_error"))
