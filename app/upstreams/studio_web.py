import json
import httpx
import traceback
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from upstreams.studio_payload import build_studio_graphql_payload
from runtime_state import app_state
import config as app_config

from stream_engine.processor import StreamProcessor

class WebProxyUpstream(BaseUpstream):
    """
    谷歌 Agent Platform Studio 网页反代渠道处理器
    加入底层 HTTP/1.1 降维打击与全套浏览器人皮伪装机制
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        auth_bundle = app_state.get_auth_bundle()
        if not auth_bundle or "headers" not in auth_bundle:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Web Proxy 凭证尚未配置，请检查浏览器自愈脚本", "type": "auth_error"}}
            )

        base_model_name = request_obj.model
        is_search = False
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
            is_search = True

        from api_helpers import create_generation_config
        gen_config_dict = create_generation_config(request_obj)

        payload = build_studio_graphql_payload(base_model_name, request_obj, gen_config_dict, auth_bundle)
        
        if is_search:
            payload["variables"].setdefault("tools", []).append({"googleSearch": {}})
            print(f"🔎 [搜索增强] 已为 Web 模式模型挂载 googleSearch 插件。")

        url = auth_bundle.get("url")
        raw_headers = auth_bundle.get("headers", {}).copy()
        
        # 1. 全部键名转换为小写
        headers = {k.lower(): str(v) for k, v in raw_headers.items()}
        
        # 2. 补全防跨站安全头
        headers["referer"] = "https://console.cloud.google.com/"
        headers["origin"] = "https://console.cloud.google.com"
        
        # 3. 剥离可能导致解压乱码或重算冲突的头
        headers.pop("accept-encoding", None)
        headers.pop("content-length", None)
        # 不要手动设置 content-type，交由 httpx.post(json=payload) 自动推导并设置长度

        # ==========================================
        # 核心修复：关闭 HTTP/2 严格指纹校验，降级为 HTTP/1.1
        # 极大提升绕过谷歌 GFE 网关 IP+TLS 复合校验的成功率
        # ==========================================
        client_kwargs = {
            "timeout": 120.0,
            "follow_redirects": True,
            "http2": False  
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        if request_obj.stream:
            async def stream_generator():
                try:
                    processor = StreamProcessor()
                    processor.enable_debug(True)
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                yield f"data: {json.dumps({'error': f'Studio Error {response.status_code}: {error_text.decode()}'})}\n\n"
                                return
                            
                            async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                                yield sse_event
                except Exception as e:
                    print("❌ [Web Proxy 异常中断] 详细网络或解析堆栈如下：")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': f'Stream translation failed: {str(e)}'})}\n\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            full_text = ""
            reasoning_text = ""
            final_finish_reason = "stop"
            tool_calls = []
            
            processor = StreamProcessor()
            processor.enable_debug(True)
            async with httpx.AsyncClient(**client_kwargs) as client:
                try:
                    async with client.stream("POST", url, headers=headers, json=payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            return JSONResponse(status_code=response.status_code, content={"error": error_text.decode()})
                        
                        async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                            if sse_event.startswith("data: "):
                                data_str = sse_event[6:].strip()
                                if data_str == "[DONE]":
                                    continue
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if "content" in delta and delta["content"] is not None:
                                            full_text += delta["content"]
                                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                            reasoning_text += delta["reasoning_content"]
                                        if choices[0].get("finish_reason"):
                                            final_finish_reason = choices[0]["finish_reason"]
                                except Exception:
                                    pass
                except Exception as e:
                    print("❌ [Web Proxy 非流式异常] 详细网络或解析堆栈如下：")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={"error": f"Failed to gather studio response: {str(e)}"})

            message_payload = {"role": "assistant"}
            if tool_calls:
                message_payload["tool_calls"] = tool_calls
                message_payload["content"] = None
            else:
                message_payload["content"] = full_text
                if reasoning_text:
                    message_payload["reasoning_content"] = reasoning_text
                    
            return JSONResponse(content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_obj.model,
                "choices": [{
                    "index": 0,
                    "message": message_payload,
                    "finish_reason": final_finish_reason
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })