import json
import time
import httpx
import traceback  # 用于在控制台打印详细错误堆栈
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from upstreams.studio_payload import build_studio_graphql_payload
from runtime_state import app_state
import config as app_config  # 引入全局配置，以便加载代理

# 引入 11-30 完美的流式追踪与消抖处理器
from stream_engine.processor import StreamProcessor


class WebProxyUpstream(BaseUpstream):
    """
    谷歌 Agent Platform Studio 网页反代渠道处理器
    封装了动态 Payload 构造、网络代理继承、安全通道以及非流式聚合拼装逻辑
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        auth_bundle = app_state.get_auth_bundle()
        if not auth_bundle or "headers" not in auth_bundle:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Web Proxy 凭证尚未配置，请在控制台填入最新 Auth Bundle", "type": "auth_error"}}
            )

        # 1. 规范化模型名称，如果是 -search 则在 payload 中启用 search_tool
        base_model_name = request_obj.model
        is_search = False
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
            is_search = True

        from api_helpers import create_generation_config
        gen_config_dict = create_generation_config(request_obj)

        # 2. 动态拼装 GraphQL 载荷
        payload = build_studio_graphql_payload(base_model_name, request_obj, gen_config_dict, auth_bundle)
        
        # 激活谷歌搜索 Grounding 插件
        if is_search:
            payload["variables"].setdefault("tools", []).append({"googleSearch": {}})
            print(f"🔎 [搜索增强] 已为 Web 模式下的模型 {base_model_name} 挂载 googleSearch 插件。")

        url = auth_bundle.get("url")
        headers = auth_bundle.get("headers", {}).copy()
        
        # ==========================================
        # 核心修复：强制补全安全保护头，防止 WAF 拦截
        # ==========================================
        headers["referer"] = "https://console.cloud.google.com/"
        headers["origin"] = "https://console.cloud.google.com"
        headers["user-agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
        # 清除会导致 httpx 双流异步解析异常的 Header
        headers.pop("accept-encoding", None)
        headers.pop("content-length", None)
        headers["content-type"] = "application/json"

        # 3. 构造 httpx 客户端参数，继承你的 .env 代理配置
        client_kwargs = {
            "timeout": 120.0,
            "follow_redirects": True
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        # 4. 流式处理通道 (stream = True)
        if request_obj.stream:
            async def stream_generator():
                # 防御式全局异常保护：确保连接报错时立刻通知客户端并输出详细日志
                try:
                    processor = StreamProcessor()
                    processor.enable_debug(True) # 开启调试日志输出，揭示每一个隐藏细节
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

        # 5. 非流式处理通道 (stream = False)，在后端自动聚合 GraphQL 流
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
                        
                        # 循环读取流，并实时解包拼接
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
                                        if "tool_calls" in delta and delta["tool_calls"] is not None:
                                            # 处理并缝合流式传输工具调用片段 (Stitch)
                                            for tc_delta in delta["tool_calls"]:
                                                idx = tc_delta.get("index", 0)
                                                if len(tool_calls) <= idx:
                                                    tool_calls.append({
                                                        "id": tc_delta.get("id", ""),
                                                        "type": "function",
                                                        "function": {"name": "", "arguments": ""}
                                                    })
                                                tc = tool_calls[idx]
                                                if tc_delta.get("id"):
                                                    tc["id"] = tc_delta["id"]
                                                if "function" in tc_delta:
                                                    fn_delta = tc_delta["function"]
                                                    if "name" in fn_delta:
                                                        tc["function"]["name"] = fn_delta["name"]
                                                    if "arguments" in fn_delta:
                                                        tc["function"]["arguments"] += fn_delta["arguments"]
                                        if choices[0].get("finish_reason"):
                                            final_finish_reason = choices[0]["finish_reason"]
                                except Exception:
                                    pass
                except Exception as e:
                    print("❌ [Web Proxy 非流式异常] 详细网络或解析堆栈如下：")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={"error": f"Failed to gather studio response: {str(e)}"})

            # 重组标准 OpenAI ChatCompletion 数据包
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
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })