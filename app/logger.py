import builtins
import time
import asyncio
import re
import threading
from typing import List

original_print = builtins.print
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class ProxyStats:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.lock = threading.Lock()

    def add_request(self, success=True):
        with self.lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
            else:
                self.error_requests += 1

    def add_tokens(self, p_tokens, c_tokens):
        with self.lock:
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens

    def get_json_stats(self):
        """向新版前端面板提供实时数据"""
        return {
            "uptime": round(time.time() - self.start_time, 2),
            "total": self.total_requests,
            "success": self.success_requests,
            "error": self.error_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens
        }

stats = ProxyStats()

class SSELogger:
    def __init__(self):
        self.queues: List[asyncio.Queue] = []
        self.max_history = 100
        self.history = []

    def push(self, plain_text):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {plain_text}"
        self.history.append(formatted_msg)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        for q in self.queues:
            try:
                q.put_nowait(formatted_msg)
            except asyncio.QueueFull:
                pass

rt_logger = SSELogger()

def custom_print(*args, **kwargs):
    import io
    buf = io.StringIO()
    original_print(*args, file=buf, **kwargs)
    raw_msg = buf.getvalue().strip()
    
    if not raw_msg:
        return

    if "💰" in raw_msg and "Tokens" in raw_msg:
        try:
            m = re.search(r'提示词:\s*(\d+).*?思考与生成:\s*(\d+)', raw_msg)
            if m: stats.add_tokens(int(m.group(1)), int(m.group(2)))
        except:
            pass

    original_print(raw_msg)
    plain_msg = ANSI_ESCAPE.sub('', raw_msg)
    rt_logger.push(plain_msg)

builtins.print = custom_print
