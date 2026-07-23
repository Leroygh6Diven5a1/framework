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
                if k in app_config.DEFAULT_SETTINGS:
                    current[k] = v
            state["settings"] = current
            self._save_state(state)
            merged = dict(app_config.DEFAULT_SETTINGS)
            merged.update(current)
            print(f"🔧 [状态管理器] 已更新 {len(patch)} 项运行时设置。")
            return merged


# 单例模式导出
app_state = AppState()
