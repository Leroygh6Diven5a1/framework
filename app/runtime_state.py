import json
import os
import threading
from typing import Dict, Any

STATE_FILE = "web_state.json"

class AppState:
    """
    多进程/多线程安全的运行态管理器
    采用文件持久化，防止 Uvicorn 多 Worker 部署时内存不共享
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self) -> dict:
        state = {"use_web_proxy": False, "auth_bundle": {}}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    state.update(json.load(f))
            except Exception as e:
                print(f"⚠️ [状态管理器] 无法读取持久化状态文件: {e}")
        return state

    def _save_state(self, state: dict):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法保存持久化状态文件: {e}")

    def enable_web_proxy(self, enabled: bool):
        """开启或关闭网页端反代模式"""
        with self._lock:
            state = self._load_state()
            state["use_web_proxy"] = enabled
            self._save_state(state)
            print(f"🔄 [状态管理器] 网页反代模式已切换为: {enabled}")

    def is_web_proxy_enabled(self) -> bool:
        """查询当前是否启用了网页反代模式"""
        with self._lock:
            state = self._load_state()
            return state.get("use_web_proxy", False)

    def update_auth_bundle(self, bundle: dict):
        """更新并落盘由油猴脚本推送过来的最新 Auth 凭证"""
        with self._lock:
            state = self._load_state()
            state["auth_bundle"] = bundle
            self._save_state(state)

    def get_auth_bundle(self) -> dict:
        """获取当前可用的 Auth 凭证拷贝"""
        with self._lock:
            state = self._load_state()
            return state.get("auth_bundle", {}).copy()

# 单例模式导出
app_state = AppState()