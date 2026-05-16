import os
import time
import httpx
import asyncio
import secrets
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from auth import get_api_key
from credentials_manager import CredentialManager
from express_key_manager import ExpressKeyManager
from vertex_ai_init import init_vertex_ai

from routes import models_api
from routes import chat_api

# 引入我们刚才重写的炫酷日志与统计面板
from logger import rt_logger, stats, console 
import config

credential_manager = CredentialManager()
express_key_manager = ExpressKeyManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    sa_credentials_available = await init_vertex_ai(credential_manager)
    sa_count = credential_manager.get_total_credentials() if sa_credentials_available else 0
    express_keys_count = express_key_manager.get_total_keys()
    
    print(f"INFO: SA credentials loaded: {sa_count}")
    print(f"INFO: Express API keys loaded: {express_keys_count}")
    
    if sa_count > 0 or express_keys_count > 0:
        print("INFO: Vertex AI authentication initialization completed successfully.")
        console.print(stats.get_stats_panel()) # 启动时打印一次统计面板
    else:
        print("ERROR: Failed to initialize any authentication method.")
        
    yield 

app = FastAPI(title="OpenAI to Gemini Adapter", lifespan=lifespan)

# 修复隐含的 Bug：原始代码缺失 CORS 配置，导致第三方 Web 前端调用直接报错
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.credential_manager = credential_manager
app.state.express_key_manager = express_key_manager

# 全局请求拦截器，负责暗中统计数据
@app.middleware("http")
async def stats_tracker_middleware(request: Request, call_next):
    if "chat/completions" in request.url.path:
        try:
            response = await call_next(request)
            stats.add_request(success=(response.status_code == 200))
            return response
        except Exception as e:
            stats.add_request(success=False)
            raise e
    return await call_next(request)


security = HTTPBasic()

def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_password = secrets.compare_digest(credentials.password, config.API_KEY)
    if not is_correct_password:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. 连本小姐的密码都记错了吗？",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# 前端 HTML 保持不变
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Vertex2OpenAI | 神性监控面板</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .log-container { font-family: 'Fira Code', monospace; scroll-behavior: smooth; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        .log-info { color: #38bdf8; }
        .log-warn { color: #fbbf24; font-weight: 500; }
        .log-error { color: #ef4444; font-weight: bold; }
        .log-success { color: #34d399; }
        
        .hl-model { color: #10b981; font-weight: bold; text-shadow: 0 0 5px rgba(16,185,129,0.3); } 
        .hl-number { color: #f472b6; font-weight: bold; } 
        .hl-keyword { color: #d946ef; } 
        .hl-express { color: #818cf8; font-weight: bold; } 
    </style>
</head>
<body class="h-screen flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-5xl bg-slate-800 rounded-xl shadow-2xl overflow-hidden border border-slate-700 flex flex-col h-[85vh]">
        <div class="bg-slate-900 px-6 py-4 border-b border-slate-700 flex justify-between items-center shadow-md z-10">
            <div class="flex items-center gap-3">
                <div class="flex gap-2">
                    <div class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]"></div>
                    <div class="w-3 h-3 rounded-full bg-yellow-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]"></div>
                    <div class="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
                </div>
                <h1 class="text-lg font-semibold text-slate-200 ml-4 tracking-wider">Vertex2OpenAI / 运行状态中枢</h1>
            </div>
            <div class="flex items-center gap-2 bg-slate-800 px-3 py-1 rounded-full border border-slate-600">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <span class="text-sm text-green-400 font-medium">代理监听中</span>
            </div>
        </div>
        <div id="log-window" class="log-container p-6 flex-1 overflow-y-auto text-sm space-y-1.5 break-all">
            </div>
    </div>
    <script>
        const logWindow = document.getElementById('log-window');
        const eventSource = new EventSource('/stream-logs');
        let isAutoScroll = true;

        logWindow.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = logWindow;
            isAutoScroll = scrollHeight - scrollTop - clientHeight < 50;
        });

        function formatLog(msg) {
            let html = msg.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            html = html.replace(/(gemini-[a-zA-Z0-9\-\.]+)/g, '<span class="hl-model">$1</span>');
            html = html.replace(/(提示词:|思考与生成:|总计:|Tokens?)/g, '<span class="hl-keyword">$1</span>');
            html = html.replace(/(\[EXPRESS\]|\[OpenAI Express Path\])/g, '<span class="hl-express">$1</span>');
            html = html.replace(/\b(\d+)\b(?![^<]*>)/g, '<span class="hl-number">$1</span>');

            let lineClass = "text-slate-400"; 
            if (html.includes('INFO:') || html.includes('DEBUG:')) lineClass = "log-info";
            else if (html.includes('WARNING:') || html.includes('⚠️')) lineClass = "log-warn";
            else if (html.includes('ERROR:') || html.includes('❌') || html.includes('Exception')) lineClass = "log-error";
            else if (html.includes('200 OK') || html.includes('SUCCESS') || html.includes('💰')) lineClass = "log-success";

            return `<div class="${lineClass}">${html}</div>`;
        }

        eventSource.onmessage = function(event) {
            logWindow.insertAdjacentHTML('beforeend', formatLog(event.data));
            if (isAutoScroll) logWindow.scrollTop = logWindow.scrollHeight;
        };

        eventSource.onerror = function(err) {
            logWindow.insertAdjacentHTML('beforeend', formatLog("[系统] ❌ SSE 链接断开，试图重新连接..."));
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(username: str = Depends(verify_auth)):
    # 每次打开网页面板时，顺便在 Docker 控制台打印一次全量统计！
    console.print(stats.get_stats_panel())
    return DASHBOARD_HTML

@app.get("/stream-logs")
async def stream_logs_endpoint(request: Request, username: str = Depends(verify_auth)):
    async def log_generator():
        q = asyncio.Queue()
        rt_logger.queues.append(q)
        try:
            for msg in rt_logger.history:
                yield f"data: {msg}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive heartbeat\n\n"
        finally:
            if q in rt_logger.queues:
                rt_logger.queues.remove(q)
                
    return StreamingResponse(log_generator(), media_type="text/event-stream")

# 加载业务路由
app.include_router(models_api.router) 
app.include_router(chat_api.router)
