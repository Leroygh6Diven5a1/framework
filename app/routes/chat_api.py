import asyncio
import json
import re
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from google.genai import types
from google import genai

from models import OpenAIRequest
from auth import get_api_key
from message_processing import create_gemini_prompt
from api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call,
)
from openai_handler import OpenAIDirectHandler
from project_id_discovery import discover_project_id

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        credential_manager_instance = fastapi_request.app.state.credential_manager
        express_key_manager_instance = fastapi_request.app.state.express_key_manager
        
        OPENAI_DIRECT_SUFFIX = "-openai"
        OPENAI_SEARCH_SUFFIX = "-openaisearch"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " 
        
        base_model_name = request.model 
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        is_openai_direct_model = False
        is_openai_search_model = False
        
        if base_model_name.endswith(OPENAI_SEARCH_SUFFIX):
            is_openai_search_model = True
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_SEARCH_SUFFIX)]
        elif base_model_name.endswith(OPENAI_DIRECT_SUFFIX):
            is_openai_direct_model = True
            base_model_name = base_model_name[:-len(OPENAI_DIRECT_SUFFIX)]
            
        if EXPERIMENTAL_MARKER in base_model_name:
            is_openai_direct_model = True

        is_grounded_search = base_model_name.endswith("-search")
        if is_grounded_search: base_model_name = base_model_name[:-len("-search")]

        is_image_model = "image" in request.model.lower()
        if is_image_model:
            is_openai_direct_model = False
            
        gen_config_dict = create_generation_config(request)

        # 【致崩 Bug 修复 / 参数优化】：
        # Gemini 3.x / 3.5 必须使用 thinking_level 控制思考深度，坚决不能传 legacy thinking_budget。
        # Gemini 2.5 只能使用 legacy thinking_budget，坚决不能传 thinking_level。
        is_thinking_capable = "gemini-2.5" in base_model_name or "gemini-3" in base_model_name or "gemini-3.5" in base_model_name
        is_gemini_2_5 = "gemini-2.5" in base_model_name
        is_gemini_3_or_above = "gemini-3" in base_model_name or "gemini-3.5" in base_model_name
        
        # 提取客户端可能传入的 reasoning_effort 参数 (支持 extra_body / standard)
        reasoning_effort = getattr(request, "reasoning_effort", None)
        if not reasoning_effort and hasattr(request, "model_extra") and request.model_extra:
            reasoning_effort = request.model_extra.get("reasoning_effort")

        if is_thinking_capable and not is_image_model:
            thinking_config = {"include_thoughts": True}
            
            if is_gemini_3_or_above:
                # 适配 Gemini 3 / 3.5 代思考层级配置
                if reasoning_effort == "low":
                    thinking_config["thinking_level"] = "low"
                elif reasoning_effort == "medium":
                    thinking_config["thinking_level"] = "medium"
                elif reasoning_effort == "high":
                    thinking_config["thinking_level"] = "high"
                else:
                    thinking_config["thinking_level"] = "high" # 默认高推理深度
                    
            elif is_gemini_2_5:
                # 适配 Gemini 2.5 代思考预算配置
                if reasoning_effort == "low":
                    thinking_config["thinking_budget"] = 1024
                else:
                    thinking_config["thinking_budget"] = -1 # -1 激活谷歌自动动态预算
            
            gen_config_dict["thinking_config"] = thinking_config

        client_to_use = None

        if is_express_model_request:
            if express_key_manager_instance.get_total_keys() == 0:
                error_msg = f"Model '{request.model}' requires an Express API key, but none are configured."
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

            total_keys = express_key_manager_instance.get_total_keys()
            for attempt in range(total_keys):
                key_tuple = express_key_manager_instance.get_express_api_key()
                if key_tuple:
                    original_idx, key_val = key_tuple
                    try:
                        if "gemini-2.5-pro" in base_model_name or "gemini-2.5-flash" in base_model_name:
                            project_id = await discover_project_id(key_val)
                            base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global"
                            client_to_use = genai.Client(
                                vertexai=True,
                                api_key=key_val,
                                http_options=types.HttpOptions(base_url=base_url)
                            )
                            client_to_use._api_client._http_options.api_version = None
                        else:
                            client_to_use = genai.Client(vertexai=True, api_key=key_val)
                        break 
                    except Exception as e:
                        client_to_use = None 
                else:
                    client_to_use = None

            if client_to_use is None: 
                return JSONResponse(status_code=500, content=create_openai_error_response(500, "All configured Express API keys failed.", "server_error"))
        
        else: 
            rotated_credentials, rotated_project_id = credential_manager_instance.get_credentials()
            
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                except Exception as e:
                    return JSONResponse(status_code=500, content=create_openai_error_response(500, str(e), "server_error"))
            else: 
                return JSONResponse(status_code=401, content=create_openai_error_response(401, "No SA credentials available.", "authentication_error"))

        if not is_openai_direct_model and client_to_use is None:
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        if is_openai_direct_model:
            if is_express_model_request:
                openai_handler = OpenAIDirectHandler(express_key_manager=express_key_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_express=True, is_openai_search=is_openai_search_model)
            else:
                openai_handler = OpenAIDirectHandler(credential_manager=credential_manager_instance)
                return await openai_handler.process_request(request, base_model_name, is_openai_search=is_openai_search_model)
        else: 
            current_prompt_func = create_gemini_prompt

            # 【Bug 优化】：统一将 Grounded Search 追加为 Dict，避免 Pydantic 实体与字典混用在 SDK 验证时触发警告
            if is_grounded_search and not is_image_model:
                search_tool = {"google_search": {}}
                if "tools" in gen_config_dict and isinstance(gen_config_dict["tools"], list):
                    gen_config_dict["tools"].append(search_tool)
                else:
                    gen_config_dict["tools"] = [search_tool]

            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, gen_config_dict, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        print(error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))