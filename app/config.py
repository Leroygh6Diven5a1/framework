from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    API_KEY: str = "123456"
    VERTEX_EXPRESS_API_KEY: Optional[str] = None
    FAKE_STREAMING: bool = False
    FAKE_STREAMING_INTERVAL: float = 1.0
    MODELS_CONFIG_URL: str = ""
    ROUNDROBIN: bool = False
    SAFETY_SCORE: bool = False
    PROXY_URL: Optional[str] = None
    SSL_CERT_FILE: Optional[str] = None

    # Cookie direct mode settings (Recommended for cloud deployments like Render)
    GOOGLE_COOKIE: Optional[str] = None         # Google Cookie string
    GOOGLE_PROJECT_ID: Optional[str] = None     # Google Cloud Project ID
    EXPERIMENT_FLAGS: Optional[str] = None      # experimentFlagsBinary (optional; paste from a console request if needed)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_settings = AppSettings()

API_KEY = _settings.API_KEY

raw_vertex_keys = _settings.VERTEX_EXPRESS_API_KEY
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(",") if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

FAKE_STREAMING_ENABLED = _settings.FAKE_STREAMING
FAKE_STREAMING_INTERVAL_SECONDS = _settings.FAKE_STREAMING_INTERVAL
MODELS_CONFIG_URL = _settings.MODELS_CONFIG_URL
ROUNDROBIN = _settings.ROUNDROBIN
SAFETY_SCORE = _settings.SAFETY_SCORE
PROXY_URL = _settings.PROXY_URL
SSL_CERT_FILE = _settings.SSL_CERT_FILE

GOOGLE_COOKIE = _settings.GOOGLE_COOKIE
GOOGLE_PROJECT_ID = _settings.GOOGLE_PROJECT_ID
EXPERIMENT_FLAGS = _settings.EXPERIMENT_FLAGS

REASONING_TAG = "agent_platform_think_tag"
# 向后兼容别名（历史代码引用 VERTEX_REASONING_TAG）
VERTEX_REASONING_TAG = REASONING_TAG


# ============================================================
# 控制台可调的运行时默认值（可在大盘热更新，持久化到 web_state.json）
# 优先级：单次请求 > 控制台设置(这些值) > 代码内置兜底
# 环境变量仅作为“初始值”。
# ============================================================
DEFAULT_SETTINGS = {
    # 思考
    "thinking_g3_level": "",              # 空=按模型各自默认(3.6-flash=medium/pro=high/flash-lite=minimal)；也可强制 minimal|low|medium|high
    "thinking_g25_budget": -1,            # Gemini 2.5 默认思考预算: -1=动态, 0=关(仅flash), 或整数
    # 生图
    "image_size": "4K",                   # 默认分辨率: 512|1K|2K|4K（按模型白名单校验）
    "image_aspect_ratio": "",             # 默认宽高比, ""=自动
    # 采样默认（客户端未显式传时使用；None=不注入）
    "default_temperature": None,
    "default_top_p": None,
    "default_max_tokens": None,
    # 输入图片压缩
    "img_compress_enabled": True,
    "img_compress_max_dim": 1536,
    "img_compress_max_mb": 1.5,
    "img_compress_quality": 85,
    # 重试
    "retry_max": 10,
    "retry_backoff_seconds": 5,
    # 开关（初始值取环境变量）
    "fake_streaming": FAKE_STREAMING_ENABLED,
    "fake_streaming_interval": FAKE_STREAMING_INTERVAL_SECONDS,
    "roundrobin": ROUNDROBIN,
    "safety_score": SAFETY_SCORE,
    # 预填充兼容模式: smart|minimal|off
    "prefill_mode": "smart",
    # 预填充触发时压制原生思考（“卡思维链”核心开关）：
    # 3.x 压到最低档（minimal/低于则 low）并关闭思考回传；2.5-flash 预算设 0 全关、2.5-pro 降到最低 128。
    # 单次请求显式传 reasoning_effort / thinking_budget 时不压制（请求优先）。
    "prefill_suppress_thinking": True,
    # smart 模式续写指令模板（留空=用内置默认；预填充文本会自动附在模板之后）
    "prefill_instruction": "",
    # Cookie 通道调试：打印出站 generationConfig（无正文时的原始响应样本总是自动记录，无需开启）
    "cookie_debug": False,
    # 按模型单独保存的参数覆盖：{ "模型ID": { 键: 值, ... } }
    # 仅覆盖“与模型相关”的参数（见 PER_MODEL_KEYS）；优先级 请求 > 模型专属 > 全局 > 内置。
    "model_overrides": {},
}

# 允许按模型单独保存（覆盖全局默认）的参数键。
# 其余为基础设施级（图压缩/重试/假流式/预填充/安全分/调试等），保持全局唯一。
PER_MODEL_KEYS = [
    "thinking_g3_level",
    "thinking_g25_budget",
    "image_size",
    "image_aspect_ratio",
    "default_temperature",
    "default_top_p",
    "default_max_tokens",
]
