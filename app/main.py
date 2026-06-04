import time
import httpx
import asyncio
import secrets
from fastapi import FastAPI, Depends, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from auth import get_api_key
from express_key_manager import ExpressKeyManager
from routes import models_api, chat_api

from logger import rt_logger, stats
import config
from runtime_state import app_state

express_key_manager = ExpressKeyManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    from model_loader import refresh_models_config_cache
    print("🚀 [服务启动] Vertex2OpenAI 已启动多模式多进程守护层。")
    if express_key_manager.get_total_keys() > 0:
        print(f"✅ [密钥配置] 已加载 {express_key_manager.get_total_keys()} 个 Express API Key。")
    else:
        print("⚠️ [密钥配置] 未检测到 VERTEX_EXPRESS_API_KEY。若不启用网页反代，聊天请求将会报错。")
    await refresh_models_config_cache()
    yield

app = FastAPI(title="OpenAI to Gemini Adapter", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.express_key_manager = express_key_manager

@app.middleware("http")
async def stats_tracker_middleware(request: Request, call_next):
    if "chat/completions" in request.url.path:
        stats.increment_total()
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                stats.add_error()
            return response
        except Exception as e:
            stats.add_error()
            raise e
    return await call_next(request)

security = HTTPBasic()
def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not secrets.compare_digest(credentials.password, config.API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ==========================================
# 💎 现代控制大盘 - 集成 Web 模式控制
# ==========================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Vertex2OpenAI | 管理控制台</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #F8FAFC; color: #334155; font-family: 'Inter', sans-serif; }
        .glass-panel { background: #FFFFFF; border: 1px solid #F1F5F9; box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03), 0 0 3px rgba(0,0,0,0.02); }
        .log-container { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.85rem; word-break: break-all; background: #FAFAF9; border: 1px solid #E5E7EB; color: #475569;}
        .nav-item { cursor: pointer; transition: all 0.25s ease; border-left: 3px solid transparent; color: #64748B; font-weight: 500;}
        .nav-item.active { background: #EFF6FF; border-left-color: #3B82F6; color: #2563EB; }
        .nav-item:hover:not(.active) { background: #F8FAFC; color: #334155; }
        @media (max-width: 768px) {
            .nav-item { border-left: none; border-bottom: 3px solid transparent; justify-content: center; flex: 1; }
            .nav-item.active { border-bottom-color: #3B82F6; background: #EFF6FF; }
        }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
        .stat-value { letter-spacing: -0.03em; }
    </style>
</head>
<body class="h-screen flex flex-col md:flex-row overflow-hidden bg-slate-50/50">
    <aside class="w-full md:w-64 glass-panel border-b md:border-b-0 md:border-r border-slate-200 flex flex-col z-20 flex-shrink-0">
        <div class="h-14 md:h-16 flex items-center px-4 md:px-6 border-b border-slate-100">
            <div class="w-7 h-7 md:w-8 md:h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-sm mr-3">V</div>
            <span class="font-bold text-base md:text-lg tracking-tight text-slate-800">Vertex2OpenAI</span>
        </div>
        <nav class="flex flex-row md:flex-col py-0 md:py-4 overflow-x-auto h-full">
            <div onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-item active px-4 py-3 md:px-6 md:py-3.5 flex items-center gap-2.5 whitespace-nowrap text-sm md:text-base">
                <svg class="w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                数据大盘
            </div>
            <div onclick="switchTab('logs')" id="nav-logs" class="nav-item px-4 py-3 md:px-6 md:py-3.5 flex items-center gap-2.5 whitespace-nowrap text-sm md:text-base">
                <svg class="w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                运行日志
            </div>
        </nav>
        <div class="mt-auto px-6 py-5 hidden md:block border-t border-slate-100">
            <div class="bg-slate-50/80 rounded-xl p-4 border border-slate-200/60 shadow-sm">
                <div class="text-[11px] text-slate-400 mb-1.5 font-semibold uppercase tracking-wider">系统状态</div>
                <div class="flex items-center gap-2 mb-2">
                    <span class="relative flex h-2.5 w-2.5">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                    </span>
                    <span class="text-sm text-emerald-600 font-semibold">Running</span>
                </div>
                <div class="text-xs text-slate-500" id="sys-uptime">已运行: 0.0 h</div>
            </div>
        </div>
    </aside>

    <main class="flex-1 flex flex-col relative z-10 overflow-hidden">
        <header class="h-14 md:h-16 glass-panel border-b border-slate-200 flex items-center justify-between px-4 md:px-8 shrink-0">
            <h1 id="page-title" class="text-base md:text-lg font-bold text-slate-800 tracking-tight">数据大盘</h1>
        </header>

        <div class="flex-1 overflow-y-auto p-4 md:p-8 relative">
            <div id="view-dashboard" class="max-w-6xl mx-auto space-y-4 md:space-y-6">
                <!-- 顶部指标网格 -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-5">
                    <div class="glass-panel p-4 md:p-5 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-20 h-20 bg-blue-50 rounded-full blur-2xl"></div>
                        <h3 class="text-slate-500 text-xs font-semibold mb-2 uppercase tracking-widest">总请求</h3>
                        <p id="stat-total" class="stat-value text-2xl md:text-3xl font-bold text-slate-800">0</p>
                    </div>
                    <div class="glass-panel p-4 md:p-5 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-20 h-20 bg-emerald-50 rounded-full blur-2xl"></div>
                        <h3 class="text-slate-500 text-xs font-semibold mb-2 uppercase tracking-widest">成功响应</h3>
                        <p id="stat-success" class="stat-value text-2xl md:text-3xl font-bold text-emerald-600">0</p>
                    </div>
                    <div class="glass-panel p-4 md:p-5 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-20 h-20 bg-amber-50 rounded-full blur-2xl"></div>
                        <h3 class="text-slate-500 text-xs font-semibold mb-2 uppercase tracking-widest">API 拥堵重试</h3>
                        <p id="stat-retries" class="stat-value text-2xl md:text-3xl font-bold text-amber-500">0</p>
                    </div>
                    <div class="glass-panel p-4 md:p-5 rounded-2xl relative overflow-hidden group">
                        <div class="absolute -right-4 -top-4 w-20 h-20 bg-rose-50 rounded-full blur-2xl"></div>
                        <h3 class="text-slate-500 text-xs font-semibold mb-2 uppercase tracking-widest">错误 / 拦截</h3>
                        <p id="stat-error" class="stat-value text-2xl md:text-3xl font-bold text-rose-600">0</p>
                    </div>
                </div>

                <!-- 模式切换控制面板卡片 -->
                <div class="glass-panel p-5 md:p-6 rounded-2xl">
                    <h3 class="text-slate-800 text-sm font-bold mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path></svg>
                        上游调用通道切换
                    </h3>
                    <div class="flex flex-col md:flex-row gap-6 items-start md:items-center">
                        <div class="flex items-center gap-5">
                            <label class="flex items-center gap-2 cursor-pointer font-medium text-sm text-slate-700">
                                <input type="radio" name="api_mode" value="api_key" checked onchange="updateMode('api_key')" class="w-4 h-4 text-blue-600 border-slate-300">
                                <span>Express API Key (标准模式)</span>
                            </label>
                            <label class="flex items-center gap-2 cursor-pointer font-medium text-sm text-slate-700">
                                <input type="radio" name="api_mode" value="web_proxy" onchange="updateMode('web_proxy')" class="w-4 h-4 text-blue-600 border-slate-300">
                                <span>Agent Platform Studio (网页免额度)</span>
                            </label>
                        </div>
                    </div>
                    
                    <div id="web-proxy-config" class="hidden mt-5 pt-5 border-t border-slate-100 space-y-4">
                        <div>
                            <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">手动导入凭证 (Auth Bundle JSON)</label>
                            <textarea id="auth-bundle-input" class="w-full text-xs font-mono p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none" rows="4" placeholder="在此粘贴书签脚本或日志中生成的最新 JSON 格式凭证..."></textarea>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-slate-400">💡 提示：更推荐通过自愈脚本实现保活。若已部署脚本，WebSocket 会全自动热更新该配置，无需手动粘贴。</span>
                            <button onclick="saveAuthBundle()" class="bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs px-5 py-2 rounded-xl shadow-sm transition-all shrink-0">应用手动凭证</button>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-5">
                    <div class="glass-panel p-5 md:p-6 rounded-2xl lg:col-span-1 flex flex-col items-center justify-center">
                        <h3 class="text-slate-800 text-sm font-bold w-full text-left mb-6">服务健康度</h3>
                        <div class="w-40 h-40 md:w-48 md:h-48 relative">
                            <canvas id="successChart"></canvas>
                        </div>
                    </div>
                    <div class="glass-panel p-5 md:p-6 rounded-2xl lg:col-span-2 flex flex-col justify-center">
                        <h3 class="text-slate-800 text-sm font-bold mb-6">Token 算力消耗量</h3>
                        <div class="space-y-6">
                            <div>
                                <div class="flex justify-between text-xs md:text-sm mb-2.5">
                                    <span class="text-slate-600 font-medium flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span> Prompt (输入)</span>
                                    <span id="stat-prompt" class="font-mono text-blue-600 font-bold">0</span>
                                </div>
                                <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden"><div class="bg-blue-500 h-full rounded-full" style="width: 80%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs md:text-sm mb-2.5">
                                    <span class="text-slate-600 font-medium flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span> Completion (输出)</span>
                                    <span id="stat-comp" class="font-mono text-indigo-600 font-bold">0</span>
                                </div>
                                <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden"><div class="bg-indigo-500 h-full rounded-full" style="width: 60%"></div></div>
                            </div>
                            <div class="pt-5 border-t border-slate-100 mt-5 flex justify-between items-center">
                                <span class="text-xs md:text-sm text-slate-500 font-bold uppercase tracking-wider">总计消耗 (Total)</span>
                                <span id="stat-total-tokens" class="text-xl md:text-2xl font-bold text-slate-800 font-mono tracking-tight">0</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="view-logs" class="hidden h-full max-w-6xl mx-auto flex flex-col glass-panel rounded-2xl overflow-hidden">
                <div class="bg-white px-4 py-3 border-b border-slate-200 flex items-center gap-2.5">
                    <div class="flex gap-1.5">
                        <div class="w-3 h-3 rounded-full bg-rose-400"></div>
                        <div class="w-3 h-3 rounded-full bg-amber-400"></div>
                        <div class="w-3 h-3 rounded-full bg-emerald-400"></div>
                    </div>
                    <span class="ml-3 text-[11px] md:text-xs text-slate-400 font-mono font-medium">terminal ~ 实时监控</span>
                </div>
                <div id="log-window" class="log-container p-4 md:p-5 flex-1 overflow-y-auto space-y-2 text-[13px]"></div>
            </div>
        </div>
    </main>

    <script>
        let chartInstance = null;
        function formatNumber(num) { return num.toLocaleString('en-US'); }

        function switchTab(tabId) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            document.getElementById('nav-' + tabId).classList.add('active');
            
            document.getElementById('view-dashboard').classList.add('hidden');
            document.getElementById('view-logs').classList.add('hidden');
            document.getElementById('view-' + tabId).classList.remove('hidden');
            
            document.getElementById('page-title').innerText = tabId === 'dashboard' ? '数据大盘' : '运行日志';
        }

        function renderChart(success, error, retries) {
            const ctx = document.getElementById('successChart').getContext('2d');
            let dataArr = [success, error, retries];
            let colorArr = ['#10B981', '#E11D48', '#F59E0B'];
            if (success === 0 && error === 0 && retries === 0) {
                dataArr = [1]; colorArr = ['#E2E8F0'];
            }
            
            if (chartInstance) {
                chartInstance.data.datasets[0].data = dataArr;
                chartInstance.data.datasets[0].backgroundColor = colorArr;
                chartInstance.update();
                return;
            }
            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['成功', '错误', '拥堵重试'],
                    datasets: [{
                        data: dataArr,
                        backgroundColor: colorArr,
                        borderWidth: 2, borderColor: '#FFFFFF', hoverOffset: 4
                    }]
                },
                options: { maintainAspectRatio: false, cutout: '75%', plugins: { legend: { display: false } }, animation: { animateScale: true } }
            });
        }

        async function fetchStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('stat-total').innerText = formatNumber(data.total);
                document.getElementById('stat-success').innerText = formatNumber(data.success);
                document.getElementById('stat-error').innerText = formatNumber(data.error);
                document.getElementById('stat-retries').innerText = formatNumber(data.retries);
                
                let hours = (data.uptime / 3600).toFixed(1);
                document.getElementById('sys-uptime').innerText = '已运行: ' + hours + ' h';
                
                document.getElementById('stat-prompt').innerText = formatNumber(data.prompt_tokens);
                document.getElementById('stat-comp').innerText = formatNumber(data.completion_tokens);
                document.getElementById('stat-total-tokens').innerText = formatNumber(data.prompt_tokens + data.completion_tokens);
                
                renderChart(data.success, data.error, data.retries);
            } catch (e) {
                console.error("Fetch stats failed", e);
            }
        }

        async function updateMode(mode) {
            if(mode === 'web_proxy') document.getElementById('web-proxy-config').classList.remove('hidden');
            else document.getElementById('web-proxy-config').classList.add('hidden');
            
            await fetch('/api/settings/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            });
        }

        async function saveAuthBundle() {
            const rawText = document.getElementById('auth-bundle-input').value.trim();
            if(!rawText) return;
            try {
                const bundle = JSON.parse(rawText);
                const res = await fetch('/api/settings/auth', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bundle)
                });
                if(res.ok) alert("🎉 凭证加载成功！");
                else alert("❌ 凭证格式或内容有误。");
            } catch(e) {
                alert("❌ JSON 格式解析失败，请检查粘贴的内容。");
            }
        }

        async function loadRuntimeSettings() {
            try {
                const res = await fetch('/api/settings/runtime');
                const state = await res.json();
                if (state.use_web_proxy) {
                    document.querySelector('input[name="api_mode"][value="web_proxy"]').checked = true;
                    document.getElementById('web-proxy-config').classList.remove('hidden');
                } else {
                    document.querySelector('input[name="api_mode"][value="api_key"]').checked = true;
                }
                if (state.auth_bundle && Object.keys(state.auth_bundle).length > 0) {
                    document.getElementById('auth-bundle-input').value = JSON.stringify(state.auth_bundle, null, 2);
                }
            } catch (e) {
                console.error("获取运行状态失败", e);
            }
        }

        const logWindow = document.getElementById('log-window');
        let isAutoScroll = true;
        
        logWindow.addEventListener('scroll', () => {
            isAutoScroll = logWindow.scrollHeight - logWindow.scrollTop - logWindow.clientHeight < 50;
        });

        function formatLogText(text) {
            let color = "#475569";
            let bgColor = "transparent";
            let borderLeft = "3px solid transparent";
            
            if(text.includes("INFO") || text.includes("✅") || text.includes("🎉")) {
                color = "#0369A1";
                borderLeft = "3px solid #38BDF8";
            }
            else if(text.includes("WARN") || text.includes("⚠️")) {
                color = "#B45309"; 
                bgColor = "#FFFBEB"; 
                borderLeft = "3px solid #F59E0B";
            }
            else if(text.includes("ERROR") || text.includes("❌")) {
                color = "#BE123C"; 
                bgColor = "#FEF2F2"; 
                borderLeft = "3px solid #F43F5E";
            }
            else if(text.includes("💰")) {
                color = "#6D28D9"; 
                bgColor = "#FAF5FF";
                borderLeft = "3px solid #A855F7";
            }
            
            let safeText = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            safeText = safeText.replace(/(gemini-[a-zA-Z0-9.-]+)/g, '<span style="color: #059669; font-weight: 700;">$1</span>');
            
            return `<div style="color: ${color}; background-color: ${bgColor}; border-left: ${borderLeft}; padding: 6px 10px; border-radius: 4px;">${safeText}</div>`;
        }

        const evtSource = new EventSource('/stream-logs');
        evtSource.onmessage = (e) => {
            if(e.data.includes("keep-alive heartbeat")) return;
            logWindow.insertAdjacentHTML('beforeend', formatLogText(e.data));
            if (isAutoScroll) logWindow.scrollTop = logWindow.scrollHeight;
        };

        fetchStats();
        loadRuntimeSettings();
        setInterval(fetchStats, 3000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard_ui(username: str = Depends(verify_auth)):
    return DASHBOARD_HTML

@app.get("/api/stats")
async def get_stats_api(username: str = Depends(verify_auth)):
    return JSONResponse(content=stats.get_json_stats())

# ==========================================
# 💎 新增：大盘设置接口（动态切换与导入）
# ==========================================
class ModeSetting(BaseModel):
    mode: str

@app.get("/api/settings/runtime")
async def get_runtime_settings(username: str = Depends(verify_auth)):
    return JSONResponse(content={
        "use_web_proxy": app_state.is_web_proxy_enabled(),
        "auth_bundle": app_state.get_auth_bundle()
    })

@app.post("/api/settings/mode")
async def set_settings_mode(setting: ModeSetting, username: str = Depends(verify_auth)):
    app_state.enable_web_proxy(setting.mode == "web_proxy")
    return JSONResponse(content={"status": "success"})

@app.post("/api/settings/auth")
async def set_settings_auth(auth_data: dict, username: str = Depends(verify_auth)):
    app_state.update_auth_bundle(auth_data)
    print("✅ [手工导入] 成功通过大盘 UI 导入并写入最新 Web 凭证！")
    return JSONResponse(content={"status": "success"})

# ==========================================
# 🔌 新增：WebSocket 双向自愈保活端点
# ==========================================
@app.websocket("/ws/harvester")
async def websocket_harvester(websocket: WebSocket, key: str = None):
    # 安全屏障：必须传入与后端完全匹配的 API_KEY 才能握手
    if not key or key != config.API_KEY:
        print("❌ [WebSocket] 鉴权拒绝：浏览器自愈脚本传入的 API_KEY 不正确。")
        await websocket.close(code=1008)
        return
        
    await websocket.accept()
    print("🔌 [WebSocket] 浏览器自愈插件连接成功，正在监听会话心跳...")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "credentials_harvested":
                harvest_payload = data.get("data")
                if harvest_payload:
                    app_state.update_auth_bundle(harvest_payload)
                    print("🔄 [WebSocket] 自愈心跳：已接收浏览器推送的最新 Web 凭证并完成热更新！")
    except WebSocketDisconnect:
        print("🔌 [WebSocket] 浏览器自愈插件连接安全断开。")
    except Exception as e:
        print(f"⚠️ [WebSocket] 通信发生异常: {e}")

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

app.include_router(models_api.router) 
app.include_router(chat_api.router)