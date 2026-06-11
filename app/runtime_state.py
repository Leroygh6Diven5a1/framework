import json
import os
import time
import asyncio
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
        self._credential_timestamp = 0  # 凭证最近更新时间戳
        self._refresh_event = None  # asyncio.Event，用于等待凭证刷新完成
        self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 增量安全合并
                    self._memory_state.update(data)
                    self._credential_timestamp = data.get("credential_timestamp", 0)
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
            self._credential_timestamp = time.time()
            state["credential_timestamp"] = self._credential_timestamp
            self._save_state(state)
            print(f"🔄 [状态管理器] 凭证已更新 @ {time.strftime('%H:%M:%S')}")
        # 通知所有等待者凭证已刷新
        self._fire_refresh_event()

    def get_auth_bundle(self) -> dict:
        with self._lock:
            state = self._load_state()
            return state.get("auth_bundle", {}).copy()

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

    # ========== 凭证生命周期管理（新增） ==========

    def get_credential_age(self) -> float:
        """获取凭证年龄（秒）"""
        if self._credential_timestamp == 0:
            return float('inf')
        return time.time() - self._credential_timestamp

    def is_credential_expired(self, max_age: int = 180) -> bool:
        """
        检查凭证是否过期
        
        Args:
            max_age: 最大有效期（秒），默认3分钟
        """
        bundle = self.get_auth_bundle()
        if not bundle or "headers" not in bundle:
            return True
        return self.get_credential_age() > max_age

    def get_credential_timestamp(self) -> float:
        """获取凭证最近更新时间戳"""
        return self._credential_timestamp

    # ========== 异步刷新等待机制 ==========

    def _get_or_create_refresh_event(self) -> asyncio.Event:
        """获取或创建 refresh event（延迟创建，确保在事件循环中）"""
        if self._refresh_event is None:
            try:
                self._refresh_event = asyncio.Event()
            except RuntimeError:
                return None
        return self._refresh_event

    def _fire_refresh_event(self):
        """触发刷新完成事件"""
        if self._refresh_event is not None:
            self._refresh_event.set()

    async def wait_for_credential_refresh(self, timeout: float = 60) -> bool:
        """
        等待凭证刷新完成
        
        Args:
            timeout: 最大等待时间（秒）
            
        Returns:
            是否在超时前获取到新凭证
        """
        event = self._get_or_create_refresh_event()
        if event is None:
            return False
        
        # 先清除事件，等待新的触发
        event.clear()
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"⚠️ [状态管理器] 等待凭证刷新超时 ({timeout}秒)")
            return False

# 单例模式导出
app_state = AppState()