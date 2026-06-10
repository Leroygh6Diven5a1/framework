"""
无头浏览器代理上游通道

使用 Playwright 无头浏览器截获的凭证调用 Vertex AI Studio API。
支持 REST (streamGenerateContent) 和 GraphQL (batchGraphql) 双通道。
"""

import copy
import json
import time
import httpx
import traceback
from typing import Any
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config

# 引入 google-genai 类型库和 OpenAI 格式转换器
from google.genai import types
from api_helpers import convert_chunk_to_openai
from message_processing import create_gemini_prompt

# 引入流式追踪与消抖处理器 (用于兼容 GraphQL 旧接口)
from stream_engine.processor import StreamProcessor


# ========== 载荷构建工具函数 ==========

def _serialize_pydantic(obj: Any) -> Any:
    """将 Pydantic 模型递归序列化为 dict"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        return {k: _serialize_pydantic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_pydantic(x) for x in obj]
    return obj


_KEY_MAP = {
    "max_output_tokens": "maxOutputTokens",
    "stop_sequences": "stopSequences",
    "top_p": "topP",
    "top_k": "topK",
    "candidate_count": "candidateCount",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "response_mime_type": "responseMimeType",
    "thinking_config": "thinkingConfig",
    "include_thoughts": "includeThoughts",
    "thinking_budget": "thinkingBudget",
    "thinking_level": "thinkingLevel",
    "image_config": "imageConfig",
    "image_size": "imageSize",
    "aspect_ratio": "aspectRatio",
    "safety_settings": "safetySettings",
    "system_instruction": "systemInstruction",
    "inline_data": "inlineData",
    "mime_type": "mimeType",
    "function_call": "functionCall",
    "function_response": "functionResponse"
}


def _convert_keys_to_camel(obj: Any) -> Any:
    """snake_case 键名转 camelCase"""
    if isinstance(obj, dict):
        return {_KEY_MAP.get(k, k): _convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_camel(x) for x in obj]
    return obj


def _build_payload(model_name: str, request: OpenAIRequest, auth_bundle: dict) -> dict:
    """
    构建 API 请求载荷
    
    兼容两种格式：
    - 模式 A：旧版 batchGraphql (body 包含 variables 节点)
    - 模式 B：标准 REST streamGenerateContent (body 直接包含 contents)
    """
    payload = copy.deepcopy(auth_bundle.get("body", {}))
    
    # 编译 OpenAI 消息历史为 Gemini 格式，并完成 Pydantic 转换
    raw_contents = create_gemini_prompt(request.messages)
    camel_contents = _convert_keys_to_camel(_serialize_pydantic(raw_contents))
    
    # 提取系统提示
    system_texts = [m.content for m in request.messages if m.role == "system" and isinstance(m.content, str)]

    # 模式 A：旧版 batchGraphql 格式 (含有 variables 节点)
    if "variables" in payload:
        variables = payload["variables"]
        variables["contents"] = camel_contents
        
        # 动态替换模型资源标识（自动保留 locations/region 等区域指纹）
        harvested_model = variables.get("model", "")
        if harvested_model and "/" in harvested_model:
            parts = harvested_model.split("/")
            parts[-1] = model_name
            variables["model"] = "/".join(parts)
        else:
            variables["model"] = model_name
            
        if "generationConfig" not in variables:
            variables["generationConfig"] = {}
        gc = variables["generationConfig"]
        
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop

        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)

        if system_texts:
            variables["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}

    # 模式 B：标准 REST streamGenerateContent 格式
    else:
        payload["contents"] = camel_contents
        
        if "generationConfig" not in payload:
            payload["generationConfig"] = {}
        gc = payload["generationConfig"]
        
        if request.temperature is not None: gc["temperature"] = request.temperature
        if request.max_tokens is not None: gc["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None: gc["topP"] = request.top_p
        if request.stop is not None: gc["stopSequences"] = request.stop

        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            gc["thinkingConfig"] = {"includeThoughts": True, "thinkingLevel": "MEDIUM"}
        else:
            gc.pop("thinkingConfig", None)

        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
        
    return payload


def _prepare_headers(raw_headers: dict) -> dict:
    """从截获的 headers 中准备发送用的请求头"""
    headers = {k.lower(): str(v) for k, v in raw_headers.items()}
    headers.pop("accept-encoding", None)
    headers.pop("content-length", None)
    headers.pop("host", None)
    headers.pop("connection", None)
    headers["content-type"] = "application/json"
    
    # 补全被浏览器屏蔽的安全防护头
    headers["referer"] = "https://console.cloud.google.com/"
    headers["origin"] = "https://console.cloud.google.com"
    
    return headers


# ========== 全局无头浏览器引用（由 main.py 设置） ==========

_headless_browser = None

def set_headless_browser(browser):
    """设置全局无头浏览器实例引用（由 main.py 在启动时调用）"""
    global _headless_browser
    _headless_browser = browser


async def _trigger_credential_refresh():
    """触发无头浏览器凭证刷新"""
    global _headless_browser
    if _headless_browser and _headless_browser.is_running:
        print("🔄 [HeadlessProxy] 凭证过期，触发无头浏览器刷新...")
        success = await _headless_browser.send_test_message()
        if success:
            # 等待凭证实际更新
            refreshed = await app_state.wait_for_credential_refresh(timeout=30)
            if refreshed:
                print("✅ [HeadlessProxy] 凭证刷新成功")
            else:
                print("⚠️ [HeadlessProxy] 凭证刷新超时")
        else:
            print("❌ [HeadlessProxy] 发送刷新消息失败")
    else:
        print("⚠️ [HeadlessProxy] 无头浏览器未运行，无法刷新凭证")


class HeadlessProxyUpstream(BaseUpstream):
    """
    无头浏览器代理通道
    使用 Playwright 截获的凭证调用 Vertex AI Studio API
    支持 REST streamGenerateContent 和 legacy GraphQL 双通道
    """
    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        auth_bundle = app_state.get_auth_bundle()
        
        # 凭证检查与自动刷新
        if not auth_bundle or "headers" not in auth_bundle:
            # 尝试触发刷新
            await _trigger_credential_refresh()
            auth_bundle = app_state.get_auth_bundle()
            if not auth_bundle or "headers" not in auth_bundle:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Studio 代理凭证尚未就绪。请确保无头浏览器已启动并完成登录。", "type": "auth_error"}}
                )

        base_model_name = request_obj.model
        is_search = False
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]
            is_search = True

        payload = _build_payload(base_model_name, request_obj, auth_bundle)
        url = auth_bundle.get("url")
        headers = _prepare_headers(auth_bundle.get("headers", {}))

        # 客户端网络特征继承
        client_kwargs = {
            "timeout": 120.0,
            "follow_redirects": True
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_kwargs["verify"] = app_config.SSL_CERT_FILE

        # 核心判断：是否为标准的 REST 区域化聊天生成流
        is_standard_rest = "streamGenerateContent" in url

        # 流式处理通道 (stream = True)
        if request_obj.stream:
            async def stream_generator():
                response_id_for_stream = f"chatcmpl-realstream-{int(time.time())}"
                
                try:
                    async with httpx.AsyncClient(**client_kwargs) as client:
                        async with client.stream("POST", url, headers=headers, json=payload) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                error_code = response.status_code
                                error_msg = error_text.decode('utf-8', errors='replace')
                                
                                # 401/403 认证错误 → 尝试刷新凭证
                                if error_code in (401, 403):
                                    print(f"⚠️ [HeadlessProxy] 认证错误 {error_code}，触发凭证刷新...")
                                    await _trigger_credential_refresh()
                                
                                yield f"data: {json.dumps({'error': f'Studio Error {error_code}: {error_msg}'})}\\n\\n"
                                return
                            
                            # 通道 A：标准 REST streamGenerateContent 流
                            if is_standard_rest:
                                buffer = ""
                                async for chunk in response.aiter_content():
                                    if not chunk: continue
                                    text_chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                                    buffer += text_chunk
                                    
                                    while True:
                                        start_idx = buffer.find('{')
                                        if start_idx == -1:
                                            buffer = ""
                                            break
                                        
                                        brace_count = 0
                                        in_string = False
                                        escape = False
                                        end_idx = -1
                                        
                                        for i in range(start_idx, len(buffer)):
                                            char = buffer[i]
                                            if escape: escape = False; continue
                                            if char == '\\': escape = True; continue
                                            if char == '"': in_string = not in_string; continue
                                                
                                            if not in_string:
                                                if char == '{': brace_count += 1
                                                elif char == '}':
                                                    brace_count -= 1
                                                    if brace_count == 0:
                                                        end_idx = i
                                                        break
                                        if end_idx != -1:
                                            json_str = buffer[start_idx:end_idx+1]
                                            buffer = buffer[end_idx+1:]
                                            try:
                                                obj = json.loads(json_str)
                                                gemini_chunk_obj = types.GenerateContentResponse(**obj)
                                                yield convert_chunk_to_openai(
                                                    gemini_chunk_obj, 
                                                    request_obj.model, 
                                                    response_id_for_stream, 
                                                    0
                                                )
                                            except Exception:
                                                pass
                                        else:
                                            buffer = buffer[start_idx:]
                                            break
                                yield "data: [DONE]\\n\\n"

                            # 通道 B：旧版 batchGraphql 格式流
                            else:
                                processor = StreamProcessor()
                                async for sse_event in processor.process_stream(response.aiter_text(), model=request_obj.model):
                                    yield sse_event
                                    
                except Exception as e:
                    print("❌ [HeadlessProxy 异常中断] 详细网络或解析堆栈如下：")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': f'Stream translation failed: {str(e)}'})}\\n\\n"
            
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # 非流式处理通道 (stream = False)
        else:
            try:
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code != 200:
                        # 认证错误触发刷新
                        if response.status_code in (401, 403):
                            await _trigger_credential_refresh()
                        return JSONResponse(status_code=response.status_code, content={"error": response.text})
                    
                    if is_standard_rest:
                        obj = response.json()
                        gemini_response_obj = types.GenerateContentResponse(**obj)
                        from message_processing import convert_to_openai_format
                        openai_response_content = convert_to_openai_format(gemini_response_obj, request_obj.model)
                        return JSONResponse(content=openai_response_content)
                    else:
                        # 兼容 GraphQL 聚合模式
                        processor = StreamProcessor()
                        parsed_res = await processor.process_stream(response.text, model=request_obj.model)
                        return JSONResponse(content=parsed_res)
            except Exception as e:
                print("❌ [HeadlessProxy 非流式异常] 详细网络或解析堆栈如下：")
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": f"Failed to gather studio response: {str(e)}"})
