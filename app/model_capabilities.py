"""
模型能力矩阵（按家族识别）

用于决定：
- 每个模型支持的思考方式（thinking_level / thinking_budget / 无）与合法档位
- 可用的采样参数集合（避免把不支持的参数发给模型导致 400）
- 生图模型支持的宽高比与分辨率白名单
- 是否要求最后一条消息为 user（Gemini 3.x 强制，否则 400 → 影响预填充）

设计原则：**按家族模式识别，而非写死每个型号**，因此往 vertexModels.json 里新增模型 ID
基本即插即用；未知/未来型号保守回退到“最新代（gemini-3）”档案。

依据（2026-07 官方文档核实）：
- thinking: ai.google.dev/gemini-api/docs/thinking
- 3.6/3.5 开发指南（采样弃用 + 预填充 400 + 去 candidate_count）
- 生图: ai.google.dev/gemini-api/docs/image-generation ；Agent Platform 3-pro-image / 3-1-flash-image
"""

import re
from typing import Any, Dict, Optional

# 所有“采样类”参数键（用于按模型剥离）
SAMPLING_KEYS = {
    "temperature", "top_p", "top_k",
    "presence_penalty", "frequency_penalty",
    "candidate_count", "seed", "stop_sequences", "max_output_tokens",
}

# 生图分辨率与比例白名单
_PRO_IMAGE_ARS = {"1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
_FLASH_IMAGE_ARS = _PRO_IMAGE_ARS | {"1:4", "4:1", "1:8", "8:1", "9:21"}


def _strip_known_suffixes(name: str) -> str:
    """去掉别名后缀（-search / 分辨率 / 思考档位后缀），得到判断家族用的基础名。"""
    n = name.lower()
    for suf in ("-search", "-1k", "-2k", "-4k", "-512"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"-think-(minimal|low|medium|high|off|none)$", "", n)
    return n


def _temp_deprecated(name: str) -> bool:
    """temperature/top_p/top_k 是否已废弃。

    据官方 latest-model 文档：自 Gemini 3.6 Flash 与 Gemini 3.5 Flash-Lite 起，
    以及**所有更新/未来的 Gemini 模型**，这三个采样参数被废弃（现忽略、未来 400），
    需从请求中移除。更早的 3.x（3.0–3.5 非 lite）仍接受（但建议保持默认）。
    """
    n = name.lower()
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
    if not m:
        return True  # 未知/未来型号 → 前向安全，按已废弃处理
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    if major >= 4:
        return True
    if major == 3 and minor >= 6:
        return True
    if major == 3 and minor == 5 and "flash-lite" in n:
        return True
    return False


def get_profile(model_name: str) -> Dict[str, Any]:
    """返回给定模型的能力档案。"""
    raw = (model_name or "").lower()
    name = _strip_known_suffixes(raw)

    # ---- 生图模型 ----
    if "image" in name:
        is_lite = "flash-lite" in name
        is_flash = "flash" in name
        if is_lite:
            ars, sizes = _PRO_IMAGE_ARS, {"1K"}
        elif is_flash:
            ars, sizes = _FLASH_IMAGE_ARS, {"512", "1K", "2K", "4K"}
        else:  # pro-image 及未知生图
            ars, sizes = _PRO_IMAGE_ARS, {"1K", "2K", "4K"}
        return {
            "family": "image",
            "is_image": True,
            "thinking_kind": None,
            "allowed_sampling": set(),          # 生图剥离所有采样参数
            "image_aspect_ratios": ars,
            "image_sizes": sizes,
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    # ---- 文本 / 多模态 ----
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 3          # 未知 → 视作最新代
    minor = int(m.group(2)) if (m and m.group(2)) else 0
    is_pro = "pro" in name

    if major >= 3:
        # Gemini 3.x：思考用 thinking_level（取代 thinking_budget），且不支持 candidate_count。
        # temperature/top_p/top_k：自 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）已废弃，
        # 官方要求从请求移除（现忽略、未来 400）；更早的 3.x 仍可用但建议保持默认。
        levels = {"low", "medium", "high"} if is_pro else {"minimal", "low", "medium", "high"}
        if "flash-lite" in name:
            default_level = "minimal"
        elif is_pro:
            default_level = "high"
        else:
            default_level = "medium"
        allowed = set(SAMPLING_KEYS)
        allowed.discard("candidate_count")  # 所有 Gemini 3.x 不支持 candidate_count
        temp_dep = _temp_deprecated(name)
        if temp_dep:
            allowed -= {"temperature", "top_p", "top_k"}
        return {
            "family": "g3",
            "is_image": False,
            "thinking_kind": "level",
            "thinking_levels": levels,
            "default_level": default_level,
            "allowed_sampling": allowed,
            "sampling_advice": "deprecated" if temp_dep else "recommend_default",
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    if major == 2 and minor >= 5:
        # Gemini 2.5：thinking_budget；保留全部采样参数
        if is_pro:
            budget_min, budget_max, can_zero = 128, 32768, False
        else:
            budget_min, budget_max, can_zero = 0, 24576, True
        return {
            "family": "g25",
            "is_image": False,
            "thinking_kind": "budget",
            "budget_min": budget_min,
            "budget_max": budget_max,
            "budget_can_zero": can_zero,
            "allowed_sampling": set(SAMPLING_KEYS),
            "supports_search": True,
            "requires_user_last_turn": False,
        }

    # ---- 更早（2.0 / 1.5 等）：无思考、全采样、宽松 ----
    return {
        "family": "legacy",
        "is_image": False,
        "thinking_kind": None,
        "allowed_sampling": set(SAMPLING_KEYS),
        "supports_search": True,
        "requires_user_last_turn": False,
    }


def _effort(request: Any) -> Optional[str]:
    e = getattr(request, "reasoning_effort", None)
    if not e and getattr(request, "model_extra", None):
        e = request.model_extra.get("reasoning_effort")
    return e.lower() if isinstance(e, str) else None


def _extra(request: Any, key: str) -> Any:
    v = getattr(request, key, None)
    if v is None and getattr(request, "model_extra", None):
        v = request.model_extra.get(key)
    return v


def resolve_thinking(model_name: str, request: Any, settings: Dict[str, Any],
                     prefill_active: bool = False) -> Dict[str, Any]:
    """
    计算思考配置（中立结构，各通道再转成自己的线格式）。
    优先级：单次请求 > 预填充压制 > 控制台设置 > 家族默认。
    返回 {"mode": None} 或 {"mode":"level","level":..} 或 {"mode":"budget","budget":..}

    prefill_active=True 且控制台开启 prefill_suppress_thinking（默认开）时，
    把思考压到该模型最低（3.x=minimal/low 且不回传思考；2.5-flash=0 全关、2.5-pro=128），
    让 roleplay 预设里的“预填充卡思维链”真正生效；单次请求显式指定思考时不压制。
    """
    prof = get_profile(model_name)
    if prof["is_image"] or prof["thinking_kind"] is None:
        return {"mode": None}

    settings = settings or {}
    suppress = bool(prefill_active and settings.get("prefill_suppress_thinking", True))

    if prof["thinking_kind"] == "level":
        levels = prof["thinking_levels"]
        if suppress and _effort(request) is None:
            level = "minimal" if "minimal" in levels else "low"
            # 3.x 无法完全关闭思考：压到最低档并不回传思考内容（官方 thinking 文档）
            return {"mode": "level", "level": level, "include_thoughts": False}
        level = _effort(request) or settings.get("thinking_g3_level") or prof.get("default_level", "high")
        level = str(level).lower()
        if level in ("off", "none"):
            level = "minimal" if "minimal" in levels else "low"
        if level not in levels:
            level = "high" if "high" in levels else sorted(levels)[-1]
        return {"mode": "level", "level": level, "include_thoughts": True}

    # budget（2.5）
    bmin, bmax, can_zero = prof["budget_min"], prof["budget_max"], prof["budget_can_zero"]
    rb = _extra(request, "thinking_budget")
    eff = _effort(request)
    if suppress and rb is None and eff is None:
        # 2.5-flash 可预算 0 完全关闭；2.5-pro 最低 128（无法全关）
        budget = 0 if can_zero else bmin
        return {"mode": "budget", "budget": budget, "include_thoughts": False}
    if rb is not None:
        try:
            budget = int(rb)
        except (TypeError, ValueError):
            budget = -1
    elif eff == "low":
        budget = max(bmin, 1024)
    elif eff in ("medium", "high"):
        budget = -1
    else:
        try:
            budget = int(settings.get("thinking_g25_budget", -1))
        except (TypeError, ValueError):
            budget = -1
    if budget == 0 and not can_zero:
        budget = bmin
    if budget != -1:
        budget = max(bmin, min(bmax, budget))
    return {"mode": "budget", "budget": budget, "include_thoughts": True}


def sanitize_sampling(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """按档案剥离不支持的采样参数（防止未来 3.x 传弃用参数直接 400）。"""
    allowed = profile.get("allowed_sampling", set())
    for key in list(config.keys()):
        if key in SAMPLING_KEYS and key not in allowed:
            config.pop(key, None)
    return config


def resolve_image_size(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """确定生图分辨率（大写 1K/2K/4K/512），按模型白名单校验，缺省回退。"""
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    sizes = prof["image_sizes"]
    settings = settings or {}
    raw = _extra(request, "image_size") or settings.get("image_size") or "1K"
    size = str(raw).upper().replace("0.5K", "512").replace("512PX", "512")
    if size in sizes:
        return size
    # 回退：优先给相近的高档，再退 1K
    for cand in ("4K", "2K", "1K", "512"):
        if cand in sizes:
            return "1K" if "1K" in sizes else cand
    return "1K"


def _prompt_aspect_ratio(request: Any) -> Optional[str]:
    """从最后一条 user 消息文本里解析宽高比（--ar 优先，其次独立比例）。"""
    try:
        for msg in reversed(request.messages):
            if getattr(msg, "role", None) != "user":
                continue
            c = msg.content
            content = ""
            if isinstance(c, str):
                content = c
            elif isinstance(c, list):
                content = " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                )
            m = re.search(r"(?i)--ar\s*(\d+[:：]\d+)", content) or re.search(r"\b(\d+[:：]\d+)\b", content)
            return m.group(1) if m else None
    except Exception:
        return None
    return None


def resolve_aspect_ratio(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """
    生图宽高比解析（两通道共用）。优先级：
    请求额外字段 aspect_ratio/ar > OpenAI size 映射 > 提示词解析 > 控制台默认，
    最后按该生图模型白名单校验；不合法则返回 None（交给模型自动决定）。
    """
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    settings = settings or {}
    raw = _extra(request, "aspect_ratio") or _extra(request, "ar")
    if not raw:
        size_param = _extra(request, "size")
        if isinstance(size_param, str):
            if size_param == "1024x1024":
                raw = "1:1"
            elif size_param == "1024x768":
                raw = "4:3"
            elif size_param == "768x1024":
                raw = "3:4"
            elif ":" in size_param:
                raw = size_param
    if not raw:
        raw = _prompt_aspect_ratio(request)
    if not raw:
        raw = settings.get("image_aspect_ratio") or None
    return validate_aspect_ratio(model_name, raw)


def validate_aspect_ratio(model_name: str, ar: Optional[str]) -> Optional[str]:
    """校验宽高比是否在该生图模型白名单内；不在则返回 None（交给模型自动决定）。"""
    if not ar:
        return None
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    norm = str(ar).replace("：", ":").strip()
    return norm if norm in prof["image_aspect_ratios"] else None


def capabilities_summary(model_name: str) -> Dict[str, Any]:
    """给控制台前端用的精简能力描述（决定显示/禁用哪些控件）。"""
    prof = get_profile(model_name)
    thinking: Dict[str, Any] = {"kind": prof["thinking_kind"]}
    if prof["thinking_kind"] == "level":
        thinking["levels"] = sorted(prof["thinking_levels"])
        thinking["can_off"] = False
    elif prof["thinking_kind"] == "budget":
        thinking["budget_min"] = prof["budget_min"]
        thinking["budget_max"] = prof["budget_max"]
        thinking["can_off"] = prof["budget_can_zero"]
    return {
        "family": prof["family"],
        "is_image": prof["is_image"],
        "thinking": thinking,
        "sampling": sorted(prof["allowed_sampling"]),
        "sampling_advice": prof.get("sampling_advice"),
        "image_aspect_ratios": sorted(prof.get("image_aspect_ratios", set())),
        "image_sizes": sorted(prof.get("image_sizes", set())),
        "supports_search": prof["supports_search"],
    }
