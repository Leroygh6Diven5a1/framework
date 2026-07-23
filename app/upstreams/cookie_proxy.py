"""
batchGraphql 直连代理上游通道

基于 Agent Platform Studio Express Mode 的 batchGraphql 协议实现。
无需任何浏览器，直接通过 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。

支持真正的实时流式响应（防 60s 超时）、429 智能退避重试，
并在客户端断开连接时立即停止上游调用。
"""

import re
import json
import time
import uuid
import asyncio
import httpx
import traceback
from typing import Any, Optional, List, Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from models import OpenAIRequest
from upstreams.base import BaseUpstream
from runtime_state import app_state
import config as app_config
import model_capabilities as mc
from message_processing import apply_prefill_compat
from logger import stats

from cookie_auth import (
    build_headers,
    BATCH_GRAPHQL_URL,
    STREAM_GENERATE_QUERY_SIGNATURE,
    STREAM_GENERATE_OPERATION_NAME,
)

# ========== 重试配置 ==========
MAX_RETRIES = 10
RETRY_BACKOFF = [5] * 10  # 每次重试等待秒数

# 可重试的错误关键词（429 限流类）
RETRYABLE_KEYWORDS = [
    "resource exhausted",
    "try again later",
    "429",
    "quota",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "internal error",
]

# Cookie 过期/权限失效的错误关键词（不可重试，需要刷新 Cookie）
COOKIE_EXPIRED_KEYWORDS = [
    "permission",
    "denied",
    "aiplatform.endpoints.predict",
    "not authorized",
    "unauthenticated",
    "login required",
    "session expired",
    "invalid credentials",
]

COOKIE_REFRESH_HINT = (
    "\n\n💡 Cookie 通常较为持久（只要不退出登录/改密码/被 Google 主动失效，可维持数周甚至更久）；"
    "仅当确实出现权限错误时才需更新。"
    "重新获取：电脑浏览器打开 console.cloud.google.com，F12 → Network，"
    "复制任意请求的 Cookie 头（或用 Cookie-Editor 导出），到大盘粘贴保存。"
)


def _is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试（429 限流类）"""
    lower = error_msg.lower()
    return any(kw in lower for kw in RETRYABLE_KEYWORDS)


def _is_cookie_expired_error(error_msg: str) -> bool:
    """判断是否为 Cookie 过期/权限失效错误"""
    lower = error_msg.lower()
    return any(kw in lower for kw in COOKIE_EXPIRED_KEYWORDS)


# ========== requestContext 模板 ==========

def _get_experiment_flags() -> str:
    """获取 experimentFlagsBinary（Express Mode 权限的关键标识，可选，从配置读取）"""
    return app_config.EXPERIMENT_FLAGS or ""


def _build_request_context(project_id: str) -> dict:
    """
    构建 batchGraphql 的 requestContext
    包含 experimentFlagsBinary，这是 Express Mode 权限的关键标识。
    """
    return {
        "clientVersion": "boq_cloud-boq-clientweb-vertexaistudio_20260609.06_p0",
        "pagePath": "/agent-platform/studio/multimodal",
        "pageViewId": int(time.time() * 1000) % (10**15),
        "trackingId": str(int(time.time() * 1000000) % (10**17)),
        "backendOverrides": {},
        "clientSessionId": str(uuid.uuid4()).upper(),
        "projectId": project_id,
        "selectedPurview": {"projectId": project_id},
        "jurisdiction": "global",
        "experimentFlagsBinary": _get_experiment_flags(),
        "localizationData": {"locale": "zh_CN", "timezone": "Asia/Hong_Kong"}
    }


# ========== 思考配置（委托中心能力模块，转 batchGraphql 的 camelCase） ==========

def _build_thinking_config(model_name: str, request: OpenAIRequest) -> Optional[dict]:
    """
    按模型能力档案 + 控制台设置 + 单次请求构建 thinkingConfig：
    - Gemini 3 及以上：thinkingLevel（MINIMAL/LOW/MEDIUM/HIGH）
    - Gemini 2.5：thinkingBudget（-1 动态；flash 可 0 关闭）
    - 其它/生图：None
    """
    settings = app_state.get_settings()
    t = mc.resolve_thinking(model_name, request, settings)
    if t.get("mode") == "level":
        return {"thinkingLevel": t["level"].upper(), "includeThoughts": t.get("include_thoughts", True)}
    if t.get("mode") == "budget":
        return {"thinkingBudget": t["budget"], "includeThoughts": t.get("include_thoughts", True)}
    return None


# ========== OpenAI → batchGraphql 消息格式转换 ==========

def _convert_messages_to_contents(messages: list) -> tuple:
    contents = []
    system_parts = []

    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        gemini_role = "user" if role == "user" else "model"

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()

                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item_type == "image_url":
                        url = item.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url.startswith("data:"):
                            try:
                                header, encoded = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append({
                                    "inlineData": {"mimeType": mime_type, "data": encoded}
                                })
                            except Exception:
                                parts.append({"text": "[图片解析失败]"})
                elif isinstance(item, str):
                    parts.append({"text": item})

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    system_text = "\n".join(system_parts) if system_parts else None
    return contents, system_text


def _build_batch_graphql_body(
    project_id: str,
    model_name: str,
    request: OpenAIRequest,
) -> dict:
    contents, system_text = _convert_messages_to_contents(request.messages)
    model_path = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"

    profile = mc.get_profile(model_name)
    settings = app_state.get_settings()
    allowed = profile["allowed_sampling"]

    gen_config = {}

    if profile["is_image"]:
        # 生图：设 responseModalities + imageConfig，不发采样/思考参数
        gen_config["responseModalities"] = ["TEXT", "IMAGE"]
        img_cfg = {}
        size = mc.resolve_image_size(model_name, request, settings)
        if size:
            img_cfg["imageSize"] = size
        ar = mc.resolve_aspect_ratio(model_name, request, settings)
        if ar:
            img_cfg["aspectRatio"] = ar
        if img_cfg:
            gen_config["imageConfig"] = img_cfg
    else:
        # 文本/多模态：仅注入该模型支持的采样参数（Gemini 3.x 不发 temperature/topP）
        if "temperature" in allowed:
            tv = request.temperature if request.temperature is not None else settings.get("default_temperature")
            gen_config["temperature"] = 1 if tv is None else tv
        if "top_p" in allowed:
            pv = request.top_p if request.top_p is not None else settings.get("default_top_p")
            gen_config["topP"] = 0.95 if pv is None else pv
        if "max_output_tokens" in allowed:
            mv = request.max_tokens if request.max_tokens is not None else settings.get("default_max_tokens")
            gen_config["maxOutputTokens"] = 65535 if mv is None else mv

        thinking_config = _build_thinking_config(model_name, request)
        if thinking_config:
            gen_config["thinkingConfig"] = thinking_config

    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    ]

    variables = {
        "contents": contents,
        "model": model_path,
        "generationConfig": gen_config,
        "safetySettings": safety_settings,
    }

    if system_text:
        variables["systemInstruction"] = {"parts": [{"text": system_text}]}

    if request.stop and "stop_sequences" in allowed:
        gen_config["stopSequences"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if hasattr(request, 'model') and request.model.endswith("-search") and profile["supports_search"]:
        variables["tools"] = [{"googleSearch": {}}]

    return {
        "requestContext": _build_request_context(project_id),
        "querySignature": STREAM_GENERATE_QUERY_SIGNATURE,
        "operationName": STREAM_GENERATE_OPERATION_NAME,
        "variables": variables,
    }


# ========== batchGraphql 流式响应解析 ==========

async def _iter_json_objects(response) -> AsyncGenerator[dict, None]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue
        buffer += chunk

        while True:
            start = buffer.find('{')
            if start == -1:
                buffer = ""
                break

            brace_count = 0
            in_string = False
            escape = False
            end = -1

            for i in range(start, len(buffer)):
                c = buffer[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break

            if end == -1:
                buffer = buffer[start:]
                break

            json_str = buffer[start:end + 1]
            buffer = buffer[end + 1:]

            try:
                yield json.loads(json_str)
            except json.JSONDecodeError:
                pass


def _extract_from_results(obj: dict):
    if "error" in obj:
        yield ("error", obj["error"])
        return

    results = obj.get("results", [])
    for result in results:
        if "errors" in result:
            for err in result["errors"]:
                yield ("error", err)
            continue

        data = result.get("data")
        if not data:
            continue

        candidates = data.get("candidates", [])
        for candidate in candidates:
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts") or []

            for part in parts:
                text = part.get("text", "")
                if text:
                    if part.get("thought", False):
                        yield ("thought", text)
                    else:
                        yield ("text", text)

                inline_data = part.get("inlineData")
                if inline_data:
                    mime_type = inline_data.get("mimeType", "")
                    b64 = inline_data.get("data", "")
                    if mime_type and b64:
                        image_md = f"![Generated Image](data:{mime_type};base64,{b64})"
                        yield ("image", image_md)

            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason in ("STOP", "MAX_TOKENS", "SAFETY"):
                yield ("finish", finish_reason)

        # 尽力解析 token 用量（batchGraphql 通常在 data.usageMetadata 中返回）
        usage = data.get("usageMetadata")
        if isinstance(usage, dict) and usage:
            yield ("usage", usage)


# ========== token 用量映射 ==========

def _log_and_map_usage(usage_meta: dict) -> dict:
    """打印 💰 统计行（供大盘解析计入 token 与成功数），并返回 OpenAI usage 字典"""
    p = int(usage_meta.get("promptTokenCount", 0) or 0)
    c = int(usage_meta.get("candidatesTokenCount", 0) or 0)
    t = int(usage_meta.get("totalTokenCount", p + c) or (p + c))
    print(f"💰 [算力消耗统计] 提示词: {p} | 思考与生成: {c} | 总计: {t} Tokens")
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


# ========== OpenAI SSE 格式化 ==========

def _make_openai_chunk(
    response_id: str,
    model: str,
    content: str = None,
    reasoning_content: str = None,
    finish_reason: str = None,
    role: str = None,
) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content

    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _make_usage_chunk(response_id: str, model: str, usage: dict) -> str:
    """OpenAI 风格的用量尾块（choices 为空，仅携带 usage）"""
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ========== 认证解析 ==========

def _get_cookie_string() -> str:
    return app_config.GOOGLE_COOKIE or app_state.get_google_cookie() or ""

def _get_project_id() -> str:
    return app_config.GOOGLE_PROJECT_ID or app_state.get_project_id() or ""


def _wants_usage(request_obj: OpenAIRequest) -> bool:
    opts = getattr(request_obj, "stream_options", None)
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


# ========== 单次流式请求执行器（重构为真正的异步实时生成器） ==========

async def _execute_stream_request_generator(
    client: httpx.AsyncClient,
    headers: dict,
    body: dict,
    model_display: str,
    response_id: str,
) -> AsyncGenerator[tuple[str, Any], None]:
    """
    异步生成器：真正实时地拉取数据并原样向外抛出。
    yield (status_type, payload)
    """
    has_content = False
    try:
        async with client.stream("POST", BATCH_GRAPHQL_URL, headers=headers, json=body) as response:

            # 1. 拦截 HTTP 状态码错误
            if response.status_code != 200:
                error_text = await response.aread()
                error_msg = error_text.decode('utf-8', errors='replace')[:1000]

                if response.status_code in (401, 403) or _is_cookie_expired_error(error_msg):
                    yield "cookie_error", error_msg + COOKIE_REFRESH_HINT
                    return

                is_retryable = response.status_code in (429, 503, 500) or _is_retryable_error(error_msg)
                yield "retryable_error" if is_retryable else "fatal_error", error_msg
                return

            # 2. 实时遍历并抛出流式 JSON 事件块
            async for obj in _iter_json_objects(response):
                for event_type, data in _extract_from_results(obj):
                    if event_type == "text":
                        yield "chunk", _make_openai_chunk(response_id, model_display, content=data)
                        has_content = True

                    elif event_type == "thought":
                        yield "chunk", _make_openai_chunk(response_id, model_display, reasoning_content=data)
                        has_content = True

                    elif event_type == "image":
                        yield "chunk", _make_openai_chunk(response_id, model_display, content=data)
                        has_content = True

                    elif event_type == "finish":
                        fr = "stop" if data == "STOP" else "length" if data == "MAX_TOKENS" else "stop"
                        yield "finish", _make_openai_chunk(response_id, model_display, finish_reason=fr)

                    elif event_type == "usage":
                        yield "usage", data

                    elif event_type == "error":
                        err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)

                        # 在还没发送任何有效数据前遇到错误，尝试走顶层重试
                        if _is_cookie_expired_error(err_msg) and not has_content:
                            yield "cookie_error", err_msg + COOKIE_REFRESH_HINT
                            return
                        if _is_retryable_error(err_msg) and not has_content:
                            yield "retryable_error", err_msg
                            return

                        # 如果流已经开始输出才发生错误，直接作为文本信息告知前端
                        yield "chunk", _make_openai_chunk(response_id, model_display, content=f"\n[Studio API 错误] {err_msg}")

    except Exception as e:
        err_msg = str(e)
        is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()
        if not has_content:
            yield "retryable_error" if is_retryable else "fatal_error", err_msg
        else:
            # 数据传输中途断开
            yield "chunk", _make_openai_chunk(response_id, model_display, content=f"\n[Studio 网络异常] 连接中断: {err_msg}")


async def _collect_full_response(project_id, base_model_name, request_obj, headers, client_kwargs,
                                 retry_max, backoff_sec, fastapi_request) -> dict:
    """非流式地完整取回一次响应（供生图假流式复用）。返回 dict。"""
    for attempt in range(retry_max + 1):
        if await fastapi_request.is_disconnected():
            return {"kind": "error", "message": "客户端已断开连接。"}
        try:
            body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
            req_headers = build_headers(_get_cookie_string()) or headers
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(BATCH_GRAPHQL_URL, headers=req_headers, json=body)

            if response.status_code in (429, 503, 500) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] HTTP {response.status_code}（生图，尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            if response.status_code != 200:
                return {"kind": "error", "message": f"HTTP {response.status_code}: {response.text[:300]}"}

            full_text, reasoning_text, finish_reason, api_error, usage_meta = "", "", "stop", None, None

            class _F:
                def __init__(self, t): self._t = t
                async def aiter_text(self): yield self._t

            async for obj in _iter_json_objects(_F(response.text)):
                for et, data in _extract_from_results(obj):
                    if et == "text":
                        full_text += data
                    elif et == "thought":
                        reasoning_text += data
                    elif et == "image":
                        full_text += data
                    elif et == "finish":
                        if data == "MAX_TOKENS":
                            finish_reason = "length"
                    elif et == "usage":
                        usage_meta = data
                    elif et == "error":
                        em = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        if _is_retryable_error(em) and attempt < retry_max:
                            api_error = em
                            break
                        full_text += f"\n[错误] {em}"

            if api_error and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图 429/限流（尝试 {attempt+1}），{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue

            return {"kind": "ok", "full_text": full_text, "reasoning_text": reasoning_text,
                    "finish_reason": finish_reason, "usage_meta": usage_meta}
        except Exception as e:
            em = str(e)
            if (_is_retryable_error(em) or "timeout" in em.lower()) and attempt < retry_max:
                stats.add_retry()
                print(f"⚠️ [Studio] 生图异常（尝试 {attempt+1}）：{em[:80]}，{backoff_sec}s 后重试...")
                await asyncio.sleep(backoff_sec)
                continue
            return {"kind": "error", "message": f"batchGraphql proxy error: {em}"}
    return {"kind": "error", "message": "重试多次仍失败。"}


# ========== 主代理类 ==========

class CookieProxyUpstream(BaseUpstream):
    """
    batchGraphql 直连代理
    使用 Cookie + SAPISIDHASH 鉴权调用 batchGraphql 端点。
    """

    async def chat_completions(self, request_obj: OpenAIRequest, fastapi_request: Request):
        # ===== 1. 验证认证 =====
        cookie_str = _get_cookie_string()
        if not cookie_str:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "未配置 Google Cookie。\n"
                "请在大盘控制台中粘贴 Cookie 和 Project ID，\n"
                "或设置环境变量 GOOGLE_COOKIE 和 GOOGLE_PROJECT_ID。"
            ), "type": "auth_error"}})

        project_id = _get_project_id()
        if not project_id:
            return JSONResponse(status_code=400, content={"error": {"message": (
                "未配置 Google Cloud Project ID。\n"
                "请在大盘中填写，或设置环境变量 GOOGLE_PROJECT_ID。\n"
                "可从 Studio URL 中获取：...?project=YOUR_PROJECT_ID"
            ), "type": "config_error"}})

        # ===== 2. 构建请求头 =====
        headers = build_headers(cookie_str)
        if not headers:
            return JSONResponse(status_code=401, content={"error": {"message": (
                "Cookie 中未找到 SAPISID，无法计算认证头。\n"
                "请确保 Cookie 来自已登录的 console.cloud.google.com 页面。"
            ), "type": "auth_error"}})

        # ===== 3. 解析模型名 =====
        model_display = request_obj.model
        base_model_name = model_display
        if base_model_name.endswith("-search"):
            base_model_name = base_model_name[:-len("-search")]

        # ===== 3.5 预填充智能兼容（按控制台模式；与模型名无关，新模型自动生效）=====
        _profile = mc.get_profile(base_model_name)
        prefill_text = ""
        _prefill_mode = app_state.get_setting("prefill_mode", "smart")
        if _prefill_mode != "off":
            _new_msgs, prefill_text = apply_prefill_compat(request_obj.messages, _prefill_mode)
            if _new_msgs is not request_obj.messages:
                request_obj = request_obj.model_copy(update={"messages": _new_msgs})
            if prefill_text:
                print(f"🩹 [预填充兼容] 已将末尾 assistant 预填充转为续写指令（{len(prefill_text)} 字），并将拼回输出开头。")

        # ===== 4. HTTP 客户端配置 =====
        client_kwargs = {
            "timeout": httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=10.0),
            "follow_redirects": True,
        }
        if app_config.PROXY_URL:
            client_kwargs["proxy"] = app_config.PROXY_URL

        # 重试配置（控制台可调）
        try:
            retry_max = int(app_state.get_setting("retry_max", MAX_RETRIES))
        except (TypeError, ValueError):
            retry_max = MAX_RETRIES
        try:
            backoff_sec = float(app_state.get_setting("retry_backoff_seconds", RETRY_BACKOFF[0]))
        except (TypeError, ValueError):
            backoff_sec = float(RETRY_BACKOFF[0])

        is_stream = request_obj.stream
        response_id = f"chatcmpl-studio-{int(time.time())}"
        start_time = time.time()
        want_usage = _wants_usage(request_obj)

        # 打印请求日志
        msg_count = len(request_obj.messages)
        print(f"→ [Studio] {base_model_name} | {msg_count} 条消息 | {'流式' if is_stream else '非流式'}")

        is_image = _profile["is_image"]

        # ========== 生图 + 流式：强制假流式 ==========
        # 生图输出是超大 base64，若按流式分块传输会卡死前端解析器；
        # 因此先完整取回，再把整张图作为“单个 chunk”一次性发出（与官方 SDK 通道一致）。
        if is_stream and is_image:
            async def image_fake_stream():
                if await fastapi_request.is_disconnected():
                    return
                print(f"🖼️ [生图保护] 图片模型 {base_model_name} 已自动切换为假流式输出（Cookie 通道），避免分块 base64 卡死前端。")
                res = await _collect_full_response(
                    project_id, base_model_name, request_obj, headers, client_kwargs,
                    retry_max, backoff_sec, fastapi_request,
                )
                yield _make_openai_chunk(response_id, model_display, role="assistant")
                if res.get("kind") != "ok":
                    stats.add_error()
                    print(f"❌ [Studio] 生图失败 | {res.get('message', '')[:150]}")
                    yield _make_openai_chunk(response_id, model_display, content=f"[Studio 错误] {res.get('message', '生图失败')}")
                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return
                full_text = res.get("full_text") or ""
                if prefill_text:
                    full_text = prefill_text + full_text
                if res.get("reasoning_text"):
                    yield _make_openai_chunk(response_id, model_display, reasoning_content=res["reasoning_text"])
                # 关键：整张图作为单个 chunk 发出，绝不分块
                yield _make_openai_chunk(response_id, model_display, content=full_text or " ")
                usage = _log_and_map_usage(res.get("usage_meta") or {})
                if want_usage:
                    yield _make_usage_chunk(response_id, model_display, usage)
                yield _make_openai_chunk(response_id, model_display, finish_reason=res.get("finish_reason", "stop"))
                yield "data: [DONE]\n\n"
                print(f"✅ [Studio] {base_model_name} | 生图假流式完成")
            return StreamingResponse(image_fake_stream(), media_type="text/event-stream")

        # ========== 流式处理（彻底解决 60s 超时的真·流式机制） ==========
        if is_stream:
            async def stream_generator():
                nonlocal start_time

                for attempt in range(retry_max + 1):
                    # 客户端已断开则立即停止，避免无谓的上游调用与重试
                    if await fastapi_request.is_disconnected():
                        print("ℹ️ [Studio] 客户端已断开连接，停止流式重试。")
                        return

                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        has_yielded_to_client = False
                        has_finish_chunk = False
                        should_retry = False
                        error_to_raise = None
                        usage_meta = None

                        # 消费实时生成器
                        async for status, data in _execute_stream_request_generator(
                            client, req_headers, body, model_display, response_id
                        ):
                            # 如果属于正常的内容块
                            if status in ("chunk", "finish"):
                                if not has_yielded_to_client:
                                    print(f"⚡ [Studio] {base_model_name} | 连接建立，正在实时流式输出...")
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                    has_yielded_to_client = True
                                    # 预填充智能兼容：先把预填充文本作为回复开头发出
                                    if prefill_text:
                                        yield _make_openai_chunk(response_id, model_display, content=prefill_text)

                                yield data

                                if status == "finish":
                                    has_finish_chunk = True

                            elif status == "usage":
                                usage_meta = data

                            # 如果属于 Cookie 权限错误
                            elif status == "cookie_error":
                                if not has_yielded_to_client:
                                    yield _make_openai_chunk(response_id, model_display, role="assistant")
                                stats.add_error()
                                print(f"🔑 [Studio] 权限错误: {data[:150]}")
                                yield _make_openai_chunk(response_id, model_display, content=f"[Studio 权限错误] {data}")
                                yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                yield "data: [DONE]\n\n"
                                return

                            # 如果属于其他网络故障或限流错误
                            elif status in ("retryable_error", "fatal_error"):
                                # 未发送任何有效数据前发生可重试错误 -> 触发安全退避重试
                                if not has_yielded_to_client and status == "retryable_error" and attempt < retry_max:
                                    should_retry = True
                                    error_to_raise = data
                                    break  # 跳出当前 async for，进入外部循环的 sleep 阶段
                                else:
                                    # 如果已经开始了输出，或者错误不可重试，直接抛给前端结束
                                    if not has_yielded_to_client:
                                        yield _make_openai_chunk(response_id, model_display, role="assistant")

                                    err_prefix = "不可重试错误" if status == "fatal_error" else "重试耗尽"
                                    stats.add_error()
                                    print(f"❌ [Studio] {err_prefix} | {data[:150]}")
                                    yield _make_openai_chunk(response_id, model_display, content=f"\n[Studio 错误] {data}")
                                    yield _make_openai_chunk(response_id, model_display, finish_reason="stop")
                                    yield "data: [DONE]\n\n"
                                    return

                        # 如果标记了需要重试，在当前 attempt 结束时等待并开启下一次循环
                        if should_retry:
                            # 重试前再次确认客户端仍在
                            if await fastapi_request.is_disconnected():
                                print("ℹ️ [Studio] 客户端已断开连接，取消后续重试。")
                                return
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] 遇到可重试拥堵/限流: {error_to_raise[:80]}... {wait_sec}s 后进行第 {attempt+2} 次退避重试")
                            await asyncio.sleep(wait_sec)
                            start_time = time.time()
                            continue

                        # 完全正常跳出循环
                        if has_yielded_to_client:
                            if not has_finish_chunk:
                                yield _make_openai_chunk(response_id, model_display, finish_reason="stop")

                            # token 统计（计入大盘）+ 可选用量尾块
                            usage = _log_and_map_usage(usage_meta) if usage_meta else _log_and_map_usage({})
                            if want_usage:
                                yield _make_usage_chunk(response_id, model_display, usage)

                            yield "data: [DONE]\n\n"

                            elapsed = time.time() - start_time
                            print(f"✅ [Studio] {base_model_name} | 流式传输顺利完毕 | 耗时 {elapsed:.1f}s")
                        return

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        # ========== 非流式处理 ==========
        else:
            for attempt in range(retry_max + 1):
                # 客户端已断开则立即停止，避免无谓的上游调用与重试
                if await fastapi_request.is_disconnected():
                    print("ℹ️ [Studio] 客户端已断开连接，停止非流式重试。")
                    return JSONResponse(status_code=499, content={
                        "error": {"message": "客户端已断开连接，请求已取消。", "type": "client_closed_request"}
                    })
                try:
                    body = _build_batch_graphql_body(project_id, base_model_name, request_obj)
                    req_headers = build_headers(_get_cookie_string()) or headers

                    async with httpx.AsyncClient(**client_kwargs) as client:
                        response = await client.post(
                            BATCH_GRAPHQL_URL, headers=req_headers, json=body
                        )

                    if response.status_code in (429, 503, 500):
                        if attempt < retry_max:
                            wait_sec = backoff_sec
                            stats.add_retry()
                            print(f"⚠️ [Studio] HTTP {response.status_code} (尝试 {attempt+1}), {wait_sec}s 后重试...")
                            await asyncio.sleep(wait_sec)
                            continue

                    if response.status_code != 200:
                        elapsed = time.time() - start_time
                        print(f"❌ [Studio] {base_model_name} | HTTP {response.status_code} | {elapsed:.1f}s")
                        return JSONResponse(status_code=response.status_code, content={
                            "error": {"message": response.text[:500], "type": "upstream_error"}
                        })

                    full_text = ""
                    reasoning_text = ""
                    finish_reason = "stop"
                    api_error = None
                    usage_meta = None

                    class _FakeResponse:
                        def __init__(self, text):
                            self._text = text
                        async def aiter_text(self):
                            yield self._text

                    fake_resp = _FakeResponse(response.text)
                    async for obj in _iter_json_objects(fake_resp):
                        for event_type, data in _extract_from_results(obj):
                            if event_type == "text":
                                full_text += data
                            elif event_type == "thought":
                                reasoning_text += data
                            elif event_type == "image":
                                full_text += data
                            elif event_type == "finish":
                                if data == "MAX_TOKENS":
                                    finish_reason = "length"
                            elif event_type == "usage":
                                usage_meta = data
                            elif event_type == "error":
                                err_msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                                if _is_retryable_error(err_msg) and attempt < retry_max:
                                    api_error = err_msg
                                    break
                                full_text += f"\n[错误] {err_msg}"

                    if api_error and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 429/限流 (尝试 {attempt+1}): {api_error[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    # 预填充智能兼容：把预填充文本拼回输出开头
                    if prefill_text:
                        full_text = prefill_text + full_text

                    if not full_text:
                        full_text = " "

                    elapsed = time.time() - start_time
                    text_len = len(full_text)
                    print(f"✅ [Studio] {base_model_name} | {text_len} 字符 | {elapsed:.1f}s")

                    usage = _log_and_map_usage(usage_meta) if usage_meta else _log_and_map_usage({})

                    message_obj = {"role": "assistant", "content": full_text}
                    if reasoning_text:
                        message_obj["reasoning_content"] = reasoning_text

                    return JSONResponse(content={
                        "id": response_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_display,
                        "choices": [{
                            "index": 0,
                            "message": message_obj,
                            "finish_reason": finish_reason,
                        }],
                        "usage": usage,
                    })

                except Exception as e:
                    err_msg = str(e)
                    is_retryable = _is_retryable_error(err_msg) or "timeout" in err_msg.lower()

                    if is_retryable and attempt < retry_max:
                        wait_sec = backoff_sec
                        stats.add_retry()
                        print(f"⚠️ [Studio] 异常 (尝试 {attempt+1}): {err_msg[:100]}, {wait_sec}s 后重试...")
                        await asyncio.sleep(wait_sec)
                        continue

                    elapsed = time.time() - start_time
                    print(f"❌ [Studio] {base_model_name} | 异常 | {elapsed:.1f}s: {err_msg[:150]}")
                    traceback.print_exc()
                    return JSONResponse(status_code=500, content={
                        "error": {"message": f"batchGraphql proxy error: {err_msg}", "type": "proxy_error"}
                    })

            elapsed = time.time() - start_time
            print(f"❌ [Studio] {base_model_name} | 重试 {retry_max} 次后仍失败 | {elapsed:.1f}s")
            return JSONResponse(status_code=429, content={
                "error": {"message": "请求被限流，已重试多次仍失败。请稍后再试。", "type": "rate_limit_error"}
            })
