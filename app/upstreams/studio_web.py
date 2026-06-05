import json
import time
import traceback
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from upstreams.studio_payload import build_studio_graphql_payload
from runtime_state import app_state
import config as app_config

from stream_engine.processor import StreamProcessor

# 引入能够完美伪装 Chrome 指纹的网络库，防止 Cookie 被谷歌网关静默剥离
try:
    from curl_cffi import requests
except ImportError:
    requests = None

class WebProxyUpstream(BaseUpstream):
    """
    谷歌 Agent Platform Studio 网页反代渠道处理器
    封装了动态 Payload 构造、curl_cffi 防剥离伪装以及非流式聚合逻辑
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        if requests is None:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "严重错误：未安装 curl_cffi。请在 requirements.txt 中添加 curl_cffi>=0.7.1 并重新构建镜像！", "type": "server_error"}}
            )

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

        # 动态组装 GraphQL 深拷贝载荷
        payload = build_studio_graphql_payload(base_model_name, request_obj, gen_config_dict, auth_bundle)
        
        if is_search:
            payload["variables"].setdefault("tools", []).append({"googleSearch": {}})
            print(f"🔎 [搜索增强] 已为 Web 模式下的模型 {base_model_name} 挂载 googleSearch 插件。")

        url = auth_bundle.get("url")
        raw_headers = auth_bundle.get("headers", {}).copy()
        
        # 强制小写 Header
        headers = {k.lower(): str(v) for k, v in raw_headers.items()}
        
        # 补全被浏览器屏蔽的安全防护头
        headers["referer"] = "https://console.cloud.google.com/"
        headers["origin"] = "https://console.cloud.google.com"
        
        headers.pop("accept-encoding", None)
        headers.pop("content-length", None)

        # 使用 curl_cffi 客户端配置，伪装为 Chrome 124，这能 100% 保住我们的 Cookie
        client_kwargs = {
            "timeout": 120.0,
            "impersonate": "chrome124"
        }
        # 适配你的代理配置
        if app_config.PROXY_URL:
            client_kwargs["proxies"] = {"http": app_config.PROXY_URL, "https": app_config.PROXY_URL}
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        if request_obj.stream:
            async def stream_generator():
                try:
                    processor = StreamProcessor()
                    processor.enable_debug(True)
                    
                    async with requests.AsyncSession(**client_kwargs) as client:
                        response = await client.post(url, headers=headers, json=payload, stream=True)
                        if response.status_code != 200:
                            error_text = await response.aread()
                            yield f"data: {json.dumps({'error': f'Studio Error {response.status_code}: {error_text.decode()}'})}\n\n"
                            return
                        
                        # 迭代 curl_cffi 的响应流
                        async def line_iterator():
                            async for line in response.aiter_lines():
                                yield line.decode('utf-8') if isinstance(line, bytes) else line
                        
                        # 传入消抖解析器
                        async for sse_event in processor.process_stream(line_iterator(), model=request_obj.model):
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
            try:
                async with requests.AsyncSession(**client_kwargs) as client:
                    response = await client.post(url, headers=headers, json=payload, stream=True)
                    if response.status_code != 200:
                        error_text = await response.aread()
                        return JSONResponse(status_code=response.status_code, content={"error": error_text.decode()})
                    
                    async def line_iterator():
                        async for line in response.aiter_lines():
                            yield line.decode('utf-8') if isinstance(line, bytes) else line
                            
                    async for sse_event in processor.process_stream(line_iterator(), model=request_obj.model):
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