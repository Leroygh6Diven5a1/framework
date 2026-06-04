from typing import Any
from pydantic import BaseModel
from models import OpenAIRequest
from message_processing import create_gemini_prompt

def serialize_pydantic(obj: Any) -> Any:
    """递归将 google-genai 的 Pydantic 模型序列化为基础 JSON 类型"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        return {k: serialize_pydantic(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_pydantic(x) for x in obj]
    return obj

# 谷歌 Web 私有 GraphQL API 所期望的 camelCase 属性键映射表
KEY_MAP = {
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

def convert_keys_to_camel(obj: Any) -> Any:
    """将参数字典中的所有 snake_case 键转换为 camelCase 驼峰命名"""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = KEY_MAP.get(k, k)
            new_dict[new_key] = convert_keys_to_camel(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys_to_camel(x) for x in obj]
    return obj

def build_studio_graphql_payload(model_name: str, request: OpenAIRequest, gen_config_dict: dict, auth_bundle: dict) -> dict:
    """
    动态结合浏览器抓取的 requestContext 和 querySignature，
    组装出与谷歌前端最新构建完全吻合的 GraphQL 私有网关请求载荷
    """
    # 动态捕获上下文
    harvested_body = auth_bundle.get("body", {})
    request_context = harvested_body.get("requestContext")
    query_signature = harvested_body.get("querySignature")
    operation_name = harvested_body.get("operationName", "StreamGenerateContentAnonymous")
    
    if not request_context or not query_signature:
        print("⚠️ [Web Proxy] 警告：当前 Auth Bundle 数据不完整，可能面临请求拒绝！")

    # 4. 动态构建符合当前 GCP Project 环境的模型全路径
    project_id = request_context.get("projectId") if request_context else None
    if project_id and not model_name.startswith("projects/"):
        model_name = f"projects/{project_id}/locations/global/publishers/google/models/{model_name}"

    # 1. 编译 OpenAI 消息历史，并一键完成 Pydantic 剥离和驼峰转换
    raw_contents = create_gemini_prompt(request.messages)
    camel_contents = convert_keys_to_camel(serialize_pydantic(raw_contents))
    
    # 2. 转换参数配置
    camel_config = convert_keys_to_camel(serialize_pydantic(gen_config_dict))
    
    # 从配置中摘出系统指令与安全配置
    system_instruction = camel_config.pop("systemInstruction", None)
    safety_settings = camel_config.get("safetySettings", [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    ])

    # 3. 构造 Web 端请求
    payload = {
        "operationName": operation_name,
        "querySignature": query_signature,
        "variables": {
            "model": model_name,
            "contents": camel_contents,
            "generationConfig": camel_config,
            "safetySettings": safety_settings,
        }
    }
    
    # 深度克隆抓取到的环境会话指纹
    if request_context:
        payload["requestContext"] = request_context
        
    if system_instruction:
        if isinstance(system_instruction, dict):
            payload["variables"]["systemInstruction"] = system_instruction
        else:
            payload["variables"]["systemInstruction"] = {"parts": [{"text": str(system_instruction)}]}
            
    return payload