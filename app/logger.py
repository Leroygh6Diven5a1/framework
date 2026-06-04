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
        self.retry_counts = 0  
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.lock = threading.Lock()

    def increment_total(self):
        with self.lock:
            self.total_requests += 1

    def add_error(self):
        with self.lock:
            self.error_requests += 1

    def add_request(self, success=True, is_error=False):
        with self.lock:
            if is_error:
                self.error_requests += 1

    def add_retry(self):
        with self.lock:
            self.retry_counts += 1

    def add_tokens(self, p_tokens, c_tokens):
        with self.lock:
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            self.success_requests += 1

    def get_json_stats(self):
        return {
            "uptime": round(time.time() - self.start_time, 2),
            "total": self.total_requests,
            "success": self.success_requests,
            "error": self.error_requests,
            "retries": self.retry_counts,
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
    """
    精修后的多线程安全打印代理
    防御式编程：杜绝双重 file 参数导致的冲突，捕获所有运行时异常
    """
    import io
    buf = io.StringIO()
    
    # 1. 深度隔离 kwargs，重定向输出到缓冲区
    kwargs_for_buffer = kwargs.copy()
    kwargs_for_buffer["file"] = buf
    
    raw_msg = ""
    try:
        original_print(*args, **kwargs_for_buffer)
        raw_msg = buf.getvalue().strip()
    except Exception:
        pass  # 缓冲写入异常时安全退出，不阻断主输出
        
    # 2. 调用原始打印，保持原有的控制台输出目标不变
    try:
        original_print(*args, **kwargs)
    except Exception:
        pass

    if not raw_msg:
        return

    # Token 统计解析
    if "💰" in raw_msg and "Tokens" in raw_msg:
        try:
            m = re.search(r'提示词:\s*(\d+).*?(?:模型)?思考与生成:\s*(\d+)', raw_msg)
            if m: 
                stats.add_tokens(int(m.group(1)), int(m.group(2)))
        except Exception:
            pass

    # 推送至前端大盘日志
    try:
        plain_msg = ANSI_ESCAPE.sub('', raw_msg)
        rt_logger.push(plain_msg)
    except Exception:
        pass

builtins.print = custom_print