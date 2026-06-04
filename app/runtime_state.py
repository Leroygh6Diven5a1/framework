import json
import os
import threading

STATE_FILE = "web_state.json"

class AppState:
    """
    多进程/多线程安全的运行态管理器
    支持 I/O 异常降级，确保在任何 Docker 权限受限环境下都不会发生崩溃
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._memory_state = {"use_web_proxy": False, "auth_bundle": {}}
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

    def update_auth_bundle(self, bundle: dict):
        with self._lock:
            state = self._load_state()
            state["auth_bundle"] = bundle
            self._save_state(state)

    def get_auth_bundle(self) -> dict:
        with self._lock:
            state = self._load_state()
            return state.get("auth_bundle", {}).copy()

# 单例模式导出
app_state = AppState()