import json
import os
import threading

import config as app_config

STATE_FILE = "web_state.json"

class AppState:
    """
    多进程/多线程安全的运行态管理器
    支持 I/O 异常降级，确保在任何 Docker 权限受限环境下都不会发生崩溃

    仅保留 Cookie 直连模式所需的运行态：
    - use_web_proxy：是否走 Cookie 直连反代通道
    - google_cookie / google_project_id：Cookie 直连凭证
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._memory_state = {"use_web_proxy": False}
        self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 增量安全合并
                    self._memory_state.update(data)
            except Exception as e:
                print(f"⚠️ [状态管理器] 无法读取持久化配置文件，已自动降级为内存模式: {e}")
        return self._memory_state

    def _save_state(self, state: dict):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法保存状态到磁盘: {e}")

    def enable_web_proxy(self, enabled: bool):
        with self._lock:
            state = self._load_state()  # 确保返回非空字典引用
            state["use_web_proxy"] = enabled
            self._save_state(state)
            print(f"🔄 [状态管理器] 网页反代状态已更新：{enabled}")

    def is_web_proxy_enabled(self) -> bool:
        with self._lock:
            state = self._load_state()
            return state.get("use_web_proxy", False)

    def set_google_cookie(self, cookie_str: str):
        with self._lock:
            state = self._load_state()
            state["google_cookie"] = cookie_str
            self._save_state(state)
            print("🔄 [状态管理器] 谷歌独立 Cookie 已保存到运行状态")

    def get_google_cookie(self) -> str:
        with self._lock:
            state = self._load_state()
            return state.get("google_cookie", "")

    def set_project_id(self, project_id: str):
        with self._lock:
            state = self._load_state()
            state["google_project_id"] = project_id
            self._save_state(state)
            print(f"🔄 [状态管理器] 项目 ID 已保存: {project_id}")

    def get_project_id(self) -> str:
        with self._lock:
            state = self._load_state()
            return state.get("google_project_id", "")

    # ========== 控制台可调设置 ==========

    def get_settings(self) -> dict:
        """返回完整设置（内置默认 + 持久化覆盖），保证所有键都存在。"""
        with self._lock:
            state = self._load_state()
            merged = dict(app_config.DEFAULT_SETTINGS)
            stored = state.get("settings")
            if isinstance(stored, dict):
                merged.update({k: v for k, v in stored.items() if k in merged})
            return merged

    def get_setting(self, key: str, default=None):
        return self.get_settings().get(key, default)

    def update_settings(self, patch: dict) -> dict:
        """合并更新设置，只接受已知键，返回更新后的完整设置。"""
        if not isinstance(patch, dict):
            return self.get_settings()
        with self._lock:
            state = self._load_state()
            current = state.get("settings")
            current = dict(current) if isinstance(current, dict) else {}
            for k, v in patch.items():
                if k in app_config.DEFAULT_SETTINGS and k != "model_overrides":
                    current[k] = v
            state["settings"] = current
            self._save_state(state)
            merged = dict(app_config.DEFAULT_SETTINGS)
            merged.update(current)
            print(f"🔧 [状态管理器] 已更新 {len(patch)} 项运行时设置。")
            return merged

    # ========== 按模型参数覆盖（per-model overrides） ==========

    def _get_overrides_map(self, state: dict) -> dict:
        stored = state.get("settings")
        if isinstance(stored, dict) and isinstance(stored.get("model_overrides"), dict):
            return dict(stored["model_overrides"])
        return {}

    def get_model_overrides(self) -> dict:
        """返回全部按模型覆盖映射：{ 模型ID: {键: 值} }。"""
        with self._lock:
            return self._get_overrides_map(self._load_state())

    def set_model_override(self, model_name: str, patch: dict) -> dict:
        """为某模型保存专属参数（仅接受 PER_MODEL_KEYS 中的键）。返回该模型的最新专属值。"""
        model_name = (model_name or "").strip()
        if not model_name or not isinstance(patch, dict):
            return {}
        clean = {k: v for k, v in patch.items() if k in app_config.PER_MODEL_KEYS}
        with self._lock:
            state = self._load_state()
            settings = state.get("settings")
            settings = dict(settings) if isinstance(settings, dict) else {}
            overrides = settings.get("model_overrides")
            overrides = dict(overrides) if isinstance(overrides, dict) else {}
            overrides[model_name] = clean
            settings["model_overrides"] = overrides
            state["settings"] = settings
            self._save_state(state)
            print(f"🔧 [状态管理器] 已保存模型 {model_name} 的专属参数（{len(clean)} 项）。")
            return clean

    def clear_model_override(self, model_name: str) -> bool:
        """清除某模型的专属参数，回退到全局默认。"""
        model_name = (model_name or "").strip()
        with self._lock:
            state = self._load_state()
            settings = state.get("settings")
            if not isinstance(settings, dict):
                return False
            overrides = settings.get("model_overrides")
            if not isinstance(overrides, dict) or model_name not in overrides:
                return False
            overrides.pop(model_name, None)
            settings["model_overrides"] = overrides
            state["settings"] = settings
            self._save_state(state)
            print(f"🔧 [状态管理器] 已清除模型 {model_name} 的专属参数。")
            return True

    def get_effective_settings(self, model_name: str) -> dict:
        """返回“该模型生效的设置”：全局默认叠加该模型专属覆盖（覆盖仅限 PER_MODEL_KEYS）。

        用于参数构建：调用方无需感知覆盖逻辑，拿到的字典里
        thinking_g3_level/image_size/default_temperature 等已是该模型的最终值。
        """
        base = self.get_settings()
        overrides = base.get("model_overrides") or {}
        ov = overrides.get((model_name or "").strip())
        if isinstance(ov, dict):
            for k in app_config.PER_MODEL_KEYS:
                if k in ov:
                    base[k] = ov[k]
        return base


# 单例模式导出
app_state = AppState()
