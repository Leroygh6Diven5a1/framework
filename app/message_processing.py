import base64
import re
import json
import time
import random 
import httpx
import concurrent.futures
from typing import List, Dict, Any, Tuple
import config as app_config
from runtime_state import app_state

from google.genai import types
from models import OpenAIMessage, ContentPartText, ContentPartImage

import io
try:
    from PIL import Image
except ImportError:
    Image = None

def optimize_image_bytes(image_data: bytes, original_mime: str, max_size_bytes: int = None) -> Tuple[bytes, str]:
    """输入图片压缩引擎：可在控制台配置开关/边长/质量/体积阈值。
    超过阈值的图会限制最长边并重采样，避免多轮修图卡死。"""
    if Image is None:
        return image_data, original_mime

    settings = app_state.get_settings()
    if not settings.get("img_compress_enabled", True):
        return image_data, original_mime

    if max_size_bytes is None:
        try:
            max_size_bytes = int(float(settings.get("img_compress_max_mb", 1.5)) * 1024 * 1024)
        except (TypeError, ValueError):
            max_size_bytes = int(1.5 * 1024 * 1024)
    try:
        max_dim = int(settings.get("img_compress_max_dim", 1536) or 1536)
    except (TypeError, ValueError):
        max_dim = 1536
    try:
        quality = int(settings.get("img_compress_quality", 85) or 85)
    except (TypeError, ValueError):
        quality = 85

    # 在安全体积内的图片，原样发送，不损耗画质
    if len(image_data) <= max_size_bytes:
        return image_data, original_mime

    try:
        with Image.open(io.BytesIO(image_data)) as img:
            # 抹平透明通道以防转成 JPEG 时报错
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA' or img.mode == 'LA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            opt_data = output.getvalue()

            # 二次压缩兜底（锁死在阈值以下）
            if len(opt_data) > max_size_bytes:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=max(40, quality - 15), optimize=True)
                opt_data = output.getvalue()

            return opt_data, "image/jpeg"
    except Exception as e:
        print(f"⚠️ [图片处理] 输入图片压缩失败，已回退为原图传输：{e}")
        return image_data, original_mime

SUPPORTED_ROLES = ["user", "model", "function"] 

def extract_reasoning_by_tags(full_text: str, tag_name: str) -> Tuple[str, str]:
    if not tag_name or not isinstance(full_text, str):
        return "", full_text if isinstance(full_text, str) else ""
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    pattern = re.compile(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", re.DOTALL)
    reasoning_parts = pattern.findall(full_text)
    normal_text = pattern.sub("", full_text)
    reasoning_content = "".join(reasoning_parts)
    return reasoning_content.strip(), normal_text.strip()

def _extract_markdown_images_to_parts(text: str) -> Tuple[List[types.Part], str]:
    parts = []
    remaining_text = text
    pattern = r"!\[[^\]]*\]\(data:(image/[a-zA-Z0-9+.-]+);base64,([a-zA-Z0-9+/=]+)\)"
    matches = list(re.finditer(pattern, text))
    
    if matches:
        for match in reversed(matches):
            mime_type = match.group(1)
            b64_data = match.group(2)
            if not mime_type.startswith("image/"):
                continue
            try:
                raw_bytes = base64.b64decode(b64_data)
                opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                start, end = match.span()
                remaining_text = remaining_text[:start] + remaining_text[end:]
            except Exception as e:
                print(f"⚠️ [图片处理] 提取 Markdown 图片失败，已跳过该图片：{e}")
        parts.reverse()
    
    remaining_text = re.sub(r"[ \t]+", " ", remaining_text).strip()
    return parts, remaining_text

def _coerce_tool_response(content: Any) -> Dict[str, Any]:
    """把 OpenAI 工具结果安全地转成 function_response 需要的对象。

    修复：旧实现用 `isinstance(str) and (...) or (...)` 的错误优先级，
    当 content 为 list 时会对其调用 .strip() 抛 AttributeError。
    """
    if content is None:
        return {"result": ""}
    if not isinstance(content, str):
        # content 可能是 OpenAI 的分段 list / dict
        try:
            return {"result": json.dumps(content, ensure_ascii=False)}
        except Exception:
            return {"result": str(content)}
    s = content.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
            # function_response 的 response 需要是对象；数组则包一层
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            return {"result": content}
    return {"result": content}


def _message_text(content: Any) -> str:
    """从 OpenAIMessage.content（str 或分段 list）里提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif hasattr(p, "text") and isinstance(getattr(p, "text", None), str):
                parts.append(p.text)
        return "".join(parts)
    return ""


def _is_empty_message(msg: OpenAIMessage) -> bool:
    if getattr(msg, "tool_calls", None):
        return False
    return not _message_text(msg.content).strip()


DEFAULT_PREFILL_INSTRUCTION = (
    "[继续输出] 下面是你这条回复已经写好的开头，请从断点处无缝继续，"
    "不要重复开头内容，也不要添加任何前言、解释或标注："
)


def apply_prefill_compat(
    messages: List[OpenAIMessage],
    mode: str = "smart",
    allow_model_last: bool = False,
    instruction_template: str = "",
) -> Tuple[List[OpenAIMessage], str, bool]:
    """
    预填充(prefill)兼容：Gemini 3.x 拒绝以 assistant/model 结尾的请求（400）。

    - mode="off"：不处理（可能 400）。
    - mode="minimal"：末尾非 user 时追加一个占位 user，仅保证不报错（不还原预填充）。
    - mode="smart"：
      * allow_model_last=True（模型允许以 model 轮次结尾，如 2.5 及更早）→ **原生透传**：
        消息保持原样发给上游，模型直接续写末尾轮次，最忠实；
      * 否则（3.x 等）→ 把末尾 assistant 预填充取出，转成末尾 user 的“续写指令”
        （模板可用 instruction_template 自定义，留空用内置默认）。
      两种情况都返回 prefill 文本，由上游把它拼回输出开头（配合去重）。

    返回 (处理后的消息列表, 需拼回输出开头的预填充文本, 是否检测到预填充)。
    第三项供“预填充时压制原生思考”等联动逻辑使用。
    与模型名无关：按请求形状 + 能力档案触发；新加模型 ID 自动生效。
    """
    if not messages or mode == "off":
        return messages, "", False

    # 找到最后一条“非空”消息
    idx = len(messages) - 1
    while idx >= 0 and _is_empty_message(messages[idx]):
        idx -= 1
    if idx < 0:
        return messages, "", False

    last = messages[idx]
    if last.role != "assistant" or getattr(last, "tool_calls", None):
        return messages, "", False  # 已是 user 结尾 / 末尾是工具调用 → 无需处理

    prefill = _message_text(last.content).strip()
    if not prefill:
        return messages, "", False

    if mode == "minimal":
        new_msgs = list(messages)
        new_msgs.append(OpenAIMessage(role="user", content="(请继续)"))
        return new_msgs, "", True

    # smart + 模型支持 model 结尾 → 原生预填充透传（不改消息，模型直接续写）
    if allow_model_last:
        return messages, prefill, True

    # smart：丢弃末尾预填充 assistant（及其后的空消息），转成续写指令
    new_msgs = list(messages[:idx])
    intro = (instruction_template or "").strip() or DEFAULT_PREFILL_INSTRUCTION
    instruction = intro + "\n\n" + prefill
    if new_msgs and new_msgs[-1].role == "user" and isinstance(new_msgs[-1].content, str):
        merged = new_msgs[-1].content + "\n\n" + instruction
        new_msgs[-1] = OpenAIMessage(role="user", content=merged)
    else:
        new_msgs.append(OpenAIMessage(role="user", content=instruction))
    return new_msgs, prefill, True


def strip_prefill_overlap(prefill: str, output: str, min_overlap: int = 8) -> str:
    """预填充去重：若模型无视指令、把预填充的结尾复述在输出开头，裁掉重叠部分。

    - 输出以整段预填充开头（完整复述）→ 裁掉整段；
    - 否则找 预填充结尾 与 输出开头 的最长重叠（≥ min_overlap 字符，避免误伤）。
    """
    if not prefill or not output:
        return output
    if output.startswith(prefill):
        return output[len(prefill):]
    kmax = min(len(prefill), len(output))
    for k in range(kmax, min_overlap - 1, -1):
        if prefill[-k:] == output[:k]:
            return output[k:]
    return output


class PrefillDeduper:
    """流式版预填充去重器。

    预填充文本由代理先行发给客户端；若模型复述了预填充开头，需要在
    流式输出的起始处裁掉重叠。做法：先攒下输出开头的一小段（窗口 =
    min(len(prefill)+32, 600) 字符），做一次去重判定后放行，之后的
    文本原样透传（不再增加任何延迟）。

    用法：out = deduper.feed(chunk_text)（可能返回空串表示还在攒）；
    流结束时调用 deduper.flush() 取回剩余文本。
    """

    def __init__(self, prefill: str, window_cap: int = 600):
        self.prefill = prefill or ""
        self.window = min(len(self.prefill) + 32, window_cap) if self.prefill else 0
        self.buffer = ""
        self.done = self.window == 0

    def feed(self, text: str) -> str:
        if self.done:
            return text
        self.buffer += text
        if len(self.buffer) < self.window:
            return ""
        return self._resolve()

    def flush(self) -> str:
        if self.done:
            return ""
        return self._resolve()

    def _resolve(self) -> str:
        out, self.buffer, self.done = self.buffer, "", True
        return strip_prefill_overlap(self.prefill, out)


def create_gemini_prompt(messages: List[OpenAIMessage]) -> List[types.Content]:
    print("🔄 [消息转换] 正在将 OpenAI 格式消息转换为 Gemini contents。")
    raw_gemini_messages = []
    for idx, message in enumerate(messages):
        role = message.role
        if role == "system":
            continue

        parts = []
        current_gemini_role = "" 

        if role == "tool":
            tool_call_id_str = message.tool_call_id or ""

            if not message.name:
                # 没有函数名无法构造规范的 function_response，降级为文本观测
                mock_text = f"[System Observation - Tool Result]:\n{message.content}"
                parts.append(types.Part.from_text(text=mock_text))
                current_gemini_role = "user"
            else:
                # 无论是否带 __thought__ 后缀，都构造规范的 function_response，
                # 让标准 OpenAI 客户端的工具往返也能正确工作（修复退化为纯文本的问题）。
                tool_output_data = _coerce_tool_response(message.content)

                real_tool_id = tool_call_id_str
                thought_sig_bytes = None
                if "__thought__" in tool_call_id_str:
                    parts_id = tool_call_id_str.split("__thought__")
                    real_tool_id = parts_id[0]
                    try:
                        thought_sig_bytes = base64.b64decode(parts_id[1])
                    except Exception:
                        thought_sig_bytes = None

                func_resp_kwargs = {"name": message.name, "response": tool_output_data}
                if real_tool_id:
                    func_resp_kwargs["id"] = real_tool_id

                try:
                    part_kwargs = {"function_response": types.FunctionResponse(**func_resp_kwargs)}
                    if thought_sig_bytes:
                        part_kwargs["thought_signature"] = thought_sig_bytes
                    resp_part = types.Part(**part_kwargs)
                except Exception as e:
                    print(f"⚠️ [工具调用] 构造 FunctionResponse 失败，将回退为基础形式：{e}")
                    resp_part = types.Part.from_function_response(name=message.name, response=tool_output_data)

                parts.append(resp_part)
                current_gemini_role = "function"

        elif role == "assistant" and message.tool_calls:
            current_gemini_role = "model"
            for tool_call in message.tool_calls:
                function_call_data = tool_call.get("function", {})
                function_name = function_call_data.get("name", "unknown")
                arguments_str = function_call_data.get("arguments", "{}")
                tool_call_id_str = tool_call.get("id", "") or ""

                try:
                    parsed_arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else (arguments_str or {})
                except json.JSONDecodeError:
                    parsed_arguments = {}

                # 无论是否带 __thought__ 后缀，都构造规范的 function_call
                real_tool_id = tool_call_id_str
                thought_sig_bytes = None
                if "__thought__" in tool_call_id_str:
                    parts_id = tool_call_id_str.split("__thought__")
                    real_tool_id = parts_id[0]
                    try:
                        thought_sig_bytes = base64.b64decode(parts_id[1])
                    except Exception:
                        thought_sig_bytes = None

                fc_kwargs = {"name": function_name, "args": parsed_arguments}
                if real_tool_id:
                    fc_kwargs["id"] = real_tool_id

                try:
                    part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                    if thought_sig_bytes:
                        part_kwargs["thought_signature"] = thought_sig_bytes
                    fc_part = types.Part(**part_kwargs)
                except Exception as e:
                    print(f"⚠️ [工具调用] 构造 FunctionCall 失败，将回退为基础形式：{e}")
                    fc_part = types.Part.from_function_call(name=function_name, args=parsed_arguments)

                parts.append(fc_part)

            if message.content:
                if isinstance(message.content, str):
                    image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                    if clean_text: parts.append(types.Part.from_text(text=clean_text))
                    parts.extend(image_parts)
        else: 
            if message.content is None: continue
            
            current_gemini_role = role
            if current_gemini_role == "assistant": current_gemini_role = "model"
            if current_gemini_role not in SUPPORTED_ROLES:
                current_gemini_role = "user"

            if isinstance(message.content, str):
                image_parts, clean_text = _extract_markdown_images_to_parts(message.content)
                if clean_text: parts.append(types.Part.from_text(text=clean_text))
                parts.extend(image_parts)

            elif isinstance(message.content, list):
                for part_item in message.content:
                    if isinstance(part_item, dict):
                        if part_item.get("type") == "text":
                            text_content = part_item.get("text", "\n")
                            image_parts, clean_text = _extract_markdown_images_to_parts(text_content)
                            if clean_text: parts.append(types.Part.from_text(text=clean_text))
                            parts.extend(image_parts)

                        elif part_item.get("type") == "image_url":
                            image_url = part_item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                mime_match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                                if mime_match:
                                    mime_type, b64_data = mime_match.groups()
                                    raw_bytes = base64.b64decode(b64_data)
                                    opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                    parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                            elif image_url.startswith("http"):
                                try:
                                    def fetch_img():
                                        client_args = {"timeout": 10.0, "follow_redirects": True}
                                        if app_config.PROXY_URL:
                                            client_args["proxy"] = app_config.PROXY_URL
                                        if getattr(app_config, "SSL_CERT_FILE", None):
                                            client_args["verify"] = app_config.SSL_CERT_FILE
                                        with httpx.Client(**client_args) as client:
                                            resp = client.get(image_url)
                                            resp.raise_for_status()
                                            return resp.content, resp.headers.get("content-type", "image/jpeg")
                                    with concurrent.futures.ThreadPoolExecutor() as pool:
                                        future = pool.submit(fetch_img)
                                        img_bytes, mime_type = future.result(timeout=12) 
                                        opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                        parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                                except Exception as e:
                                    print(f"⚠️ [图片处理] 获取远程图片失败，已跳过：{image_url}，原因：{e}")

                    elif hasattr(part_item, "type") and getattr(part_item, "type") == "image_url":
                        img_url_data = part_item.image_url
                        url_str = getattr(img_url_data, "url", "") if hasattr(img_url_data, "url") else (img_url_data.get("url", "") if isinstance(img_url_data, dict) else "")
                        
                        if url_str.startswith("data:"):
                            mime_match = re.match(r"data:([^;]+);base64,(.+)", url_str)
                            if mime_match:
                                mime_type, b64_data = mime_match.groups()
                                raw_bytes = base64.b64decode(b64_data)
                                opt_bytes, opt_mime = optimize_image_bytes(raw_bytes, mime_type)
                                parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                        elif url_str.startswith("http"):
                            try:
                                def fetch_img():
                                    client_args = {"timeout": 10.0, "follow_redirects": True}
                                    if app_config.PROXY_URL:
                                        client_args["proxy"] = app_config.PROXY_URL
                                    if getattr(app_config, "SSL_CERT_FILE", None):
                                        client_args["verify"] = app_config.SSL_CERT_FILE
                                    with httpx.Client(**client_args) as client:
                                        resp = client.get(url_str)
                                        resp.raise_for_status()
                                        return resp.content, resp.headers.get("content-type", "image/jpeg")
                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    future = pool.submit(fetch_img)
                                    img_bytes, mime_type = future.result(timeout=12) 
                                    opt_bytes, opt_mime = optimize_image_bytes(img_bytes, mime_type)
                                    parts.append(types.Part.from_bytes(data=opt_bytes, mime_type=opt_mime))
                            except Exception as e:
                                print(f"⚠️ [图片处理] 获取远程图片失败，已跳过：{url_str}，原因：{e}")
                                
                    elif hasattr(part_item, "text"):
                        parts.append(types.Part.from_text(text=part_item.text))

        if not parts: continue
        raw_gemini_messages.append(types.Content(role=current_gemini_role, parts=parts))

    merged_messages = []
    for msg in raw_gemini_messages:
        if merged_messages and merged_messages[-1].role == msg.role:
            merged_messages[-1].parts.append(types.Part.from_text(text="\n\n"))
            merged_messages[-1].parts.extend(msg.parts)
        else:
            merged_messages.append(msg)

    if not merged_messages:
        merged_messages.append(types.Content(role="user", parts=[types.Part.from_text(text="继续")]))

    return merged_messages

def _create_safety_ratings_html(safety_ratings: list) -> str:
    if not safety_ratings:
        return ""
    highest_rating = max(safety_ratings, key=lambda r: r.probability_score)
    highest_score = highest_rating.probability_score

    if highest_score <= 0.33: color = "#0f8"  
    elif highest_score <= 0.66: color = "yellow"
    else: color = "#bf555d"

    summary_category = highest_rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
    summary_probability = highest_rating.probability.name
    summary_score_str = f"{highest_rating.probability_score:.7f}" if highest_rating.probability_score is not None else "None"
    summary_severity_str = f"{highest_rating.severity_score:.8f}" if highest_rating.severity_score is not None else "None"
    summary_line = f"{summary_category}: {summary_probability} (Score: {summary_score_str}, Severity: {summary_severity_str})"

    ratings_list = []
    for rating in safety_ratings:
        category = rating.category.name.replace("HARM_CATEGORY_", "").replace("_", " ").title()
        probability = rating.probability.name
        score_str = f"{rating.probability_score:.7f}" if rating.probability_score is not None else "None"
        severity_str = f"{rating.severity_score:.8f}" if rating.severity_score is not None else "None"
        ratings_list.append(f"{category}: {probability} (Score: {score_str}, Severity: {severity_str})")
    all_ratings_str = "\n".join(ratings_list)

    css_style = "<style>.cb{border:1px solid #444;margin:10px;border-radius:4px;background:#111}.cb summary{padding:8px;cursor:pointer;background:#222}.cb pre{margin:0;padding:10px;border-top:1px solid #444;white-space:pre-wrap}</style>"
    html_output = (
        f"{css_style}"
        f"<details class='cb'>"
        f"<summary style='color:{color}'>{summary_line} ▼</summary>"
        f"<pre>\n--- Safety Ratings ---\n{all_ratings_str}\n</pre>"
        f"</details>"
    )
    return html_output

def _convert_image_to_markdown(image_data: bytes, mime_type: str) -> str:
    try:
        b64_data = base64.b64encode(image_data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"
        return f"![Image]({data_url})"
    except Exception as e:
        print(f"⚠️ [图片处理] 将 Gemini 图片转换为 Markdown 失败：{e}")
        return "[Image could not be displayed]"

def parse_gemini_response_for_reasoning_and_content(gemini_response_candidate: Any) -> Tuple[str, str]:
    reasoning_text_parts = []
    normal_text_parts = []
    candidate_part_text = ""
    if hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None:
        candidate_part_text = str(gemini_response_candidate.text)

    gemini_candidate_content = None
    if hasattr(gemini_response_candidate, "content"):
        gemini_candidate_content = gemini_response_candidate.content

    if gemini_candidate_content and hasattr(gemini_candidate_content, "parts") and gemini_candidate_content.parts:
        for part_item in gemini_candidate_content.parts:
            if hasattr(part_item, "function_call") and part_item.function_call is not None: 
                continue
            
            part_text = ""
            if hasattr(part_item, "text") and part_item.text is not None:
                part_text = str(part_item.text)
            elif hasattr(part_item, "inline_data") and part_item.inline_data is not None:
                inline_data = part_item.inline_data
                if hasattr(inline_data, "data") and hasattr(inline_data, "mime_type"):
                    image_bytes = inline_data.data
                    mime_type = inline_data.mime_type
                    part_text = _convert_image_to_markdown(image_bytes, mime_type)
            elif hasattr(part_item, "file_data") and part_item.file_data is not None:
                file_data = part_item.file_data
                if hasattr(file_data, "file_uri"):
                    file_uri = file_data.file_uri
                    part_text = f"![Image]({file_uri})"
            
            part_is_thought = hasattr(part_item, "thought") and part_item.thought is True

            if part_is_thought: reasoning_text_parts.append(part_text)
            elif part_text: normal_text_parts.append(part_text)
            
    elif candidate_part_text: normal_text_parts.append(candidate_part_text)
    elif gemini_candidate_content and hasattr(gemini_candidate_content, "text") and gemini_candidate_content.text is not None:
        normal_text_parts.append(str(gemini_candidate_content.text))
    elif hasattr(gemini_response_candidate, "text") and gemini_response_candidate.text is not None and not gemini_candidate_content: 
        normal_text_parts.append(str(gemini_response_candidate.text))

    return "".join(reasoning_text_parts), "".join(normal_text_parts)

def process_gemini_response_to_openai_dict(gemini_response_obj: Any, request_model_str: str) -> Dict[str, Any]:
    choices = []
    response_timestamp = int(time.time())
    base_id = f"chatcmpl-{response_timestamp}-{random.randint(1000,9999)}"

    if hasattr(gemini_response_obj, "candidates") and gemini_response_obj.candidates:
        for i, candidate in enumerate(gemini_response_obj.candidates):
            message_payload = {"role": "assistant"}
            
            raw_finish_reason = getattr(candidate, "finish_reason", None)
            openai_finish_reason = "stop" 
            if raw_finish_reason:
                if hasattr(raw_finish_reason, "name"): raw_gemini_finish_reason_str = raw_finish_reason.name.upper()
                else: raw_gemini_finish_reason_str = str(raw_finish_reason).upper()

                if raw_gemini_finish_reason_str == "STOP": openai_finish_reason = "stop"
                elif raw_gemini_finish_reason_str == "MAX_TOKENS": openai_finish_reason = "length"
                elif raw_gemini_finish_reason_str == "SAFETY": openai_finish_reason = "content_filter"
                elif raw_gemini_finish_reason_str in ["TOOL_CODE", "FUNCTION_CALL"]: openai_finish_reason = "tool_calls"
            
            function_call_detected = False
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call is not None: 
                        fc = part.function_call
                        
                        real_id = getattr(fc, "id", None)
                        if not real_id: real_id = getattr(fc, "thought_signature", None)
                        
                        thought_sig = getattr(part, "thought_signature", None)
                        thought_sig_b64 = ""
                        if thought_sig:
                            if isinstance(thought_sig, bytes):
                                thought_sig_b64 = base64.b64encode(thought_sig).decode("utf-8")
                            elif isinstance(thought_sig, str):
                                thought_sig_b64 = thought_sig

                        safe_name = fc.name.replace(" ", "_")
                        rand_num = int(time.time() * 10000 + random.randint(0, 9999))

                        if real_id:
                            if thought_sig_b64:
                                tool_call_id = f"{real_id}__thought__{thought_sig_b64}"
                            else:
                                tool_call_id = real_id
                        else:
                            if thought_sig_b64:
                                tool_call_id = f"call_{base_id}_{i}_{safe_name}__thought__{thought_sig_b64}"
                            else:
                                tool_call_id = f"call_{base_id}_{i}_{safe_name}_{rand_num}"
                        
                        if "tool_calls" not in message_payload:
                            message_payload["tool_calls"] = []
                        
                        message_payload["tool_calls"].append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args or {})
                            }
                        })
                        message_payload["content"] = None 
                        openai_finish_reason = "tool_calls" 
                        function_call_detected = True
            
            if not function_call_detected:
                reasoning_str, normal_content_str = parse_gemini_response_for_reasoning_and_content(candidate)
                if app_state.get_setting("safety_score", app_config.SAFETY_SCORE) and hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                    safety_html = _create_safety_ratings_html(candidate.safety_ratings)
                    if reasoning_str: reasoning_str += safety_html
                    else: normal_content_str += safety_html
                
                message_payload["content"] = normal_content_str
                if reasoning_str: message_payload["reasoning_content"] = reasoning_str
            
            choice_item = {"index": i, "message": message_payload, "finish_reason": openai_finish_reason}
            if hasattr(candidate, "logprobs") and candidate.logprobs is not None: choice_item["logprobs"] = candidate.logprobs
            choices.append(choice_item)
            
    elif hasattr(gemini_response_obj, "text") and gemini_response_obj.text is not None:
         content_str = gemini_response_obj.text or ""
         choices.append({"index": 0, "message": {"role": "assistant", "content": content_str}, "finish_reason": "stop"})
    else: 
         choices.append({"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"})

    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(gemini_response_obj, "usage_metadata"):
        um = gemini_response_obj.usage_metadata
        if hasattr(um, "prompt_token_count"): usage_data["prompt_tokens"] = um.prompt_token_count
        if hasattr(um, "candidates_token_count"):
            usage_data["completion_tokens"] = um.candidates_token_count
            if hasattr(um, "total_token_count"): usage_data["total_tokens"] = um.total_token_count
            else: usage_data["total_tokens"] = usage_data["prompt_tokens"] + usage_data["completion_tokens"]
        elif hasattr(um, "total_token_count"): 
             usage_data["total_tokens"] = um.total_token_count
             if usage_data["prompt_tokens"] > 0 and usage_data["total_tokens"] > usage_data["prompt_tokens"]:
                 usage_data["completion_tokens"] = usage_data["total_tokens"] - usage_data["prompt_tokens"]
        else: usage_data["total_tokens"] = usage_data["prompt_tokens"] 

    return {
        "id": base_id, "object": "chat.completion", "created": response_timestamp,
        "model": request_model_str, "choices": choices,
        "usage": usage_data
    }

def convert_to_openai_format(gemini_response: Any, model: str) -> Dict[str, Any]:
    return process_gemini_response_to_openai_dict(gemini_response, model)