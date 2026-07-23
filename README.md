---
title: Vertex2OpenAI Express Adapter
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vertex2OpenAI Express Adapter

Vertex2OpenAI 是一个 **OpenAI API 兼容代理**。它对外提供 OpenAI 风格的 `/v1/chat/completions` 和 `/v1/models` 接口，对内支持调用 **Google Agent Platform（原 Vertex AI）Express Mode 的 Gemini API**，或通过网页控制台 **Cookie 直连反代（batchGraphql）**。

> 当前版本已进行全面重构，支持**标准 API 模式**和 **Cookie 直连反代模式**的双上游切换。
>
> 📌 **说明**：Google 已将 Vertex AI 更名为 **Agent Platform**，文档中相关表述已同步更新（部分历史标识符如环境变量 `VERTEX_EXPRESS_API_KEY`、SDK 参数 `vertexai=True` 为兼容性保留）。此外，**无头浏览器（Playwright）模式已移除**，Cookie 直连模式已完全替代它，部署更轻量。

## 功能特性

- **双上游调用通道切换**
  - **Express API Key (标准模式)**：通过 `VERTEX_EXPRESS_API_KEY` 轮询或随机调用官方 API。
  - **Agent Platform Studio (网页直连反代)**：使用 Google Cloud 控制台 Cookie 和 Project ID 直接调用控制台私有 `batchGraphql` 接口，模拟网页端交互，调用网页版 Studio 模型（例如最新的预览模型）。
- **OpenAI 兼容接口**
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- **现代化管理大盘**
  - 支持标准模式与网页反代模式在线一键无缝切换。
  - 在线热更新并保存 Google Cookie 和 Project ID。
  - **智能 Cookie 解析**：支持直接粘贴 `Cookie-Editor` 等浏览器插件导出的 JSON 或 Header String 格式的 Cookie，系统会自动解析。
  - **Project ID 自动识别**：可直接将含有 `?project=xxx` 或 `/projects/xxx` 的控制台整条 URL 粘贴到输入框中，系统会自动解析并提取干净的 Project ID。
  - **实时监控与图表**：实时运行日志推流展示、服务健康度比例分析图表（成功、错误、拥堵重试次数）、输入/输出/总计 Token 算力消耗量统计。
- **Gemini 原有能力与优化**
  - 普通文本对话、流式（SSE）和非流式响应。
  - OpenAI tools / function calling 到 Gemini function calling 的适配转换。
  - Google Search 谷歌搜索增强模型别名：支持在模型名后添加 `-search` 后缀开启。
  - 自动识别并保留最新的 Gemini 思考过程（Thinking Process），并以 `reasoning_content` 的形式在流中返回。
  - 生图模型配置，包括图片输入压缩、比例解析、图片输出 Markdown data URL 转换。
  - **自动退避重试**：无论标准 API 还是网页直连模式，均内置了 429 限流/拥堵自动退避重试方案（最大重试 3 次，延迟 2s, 4s, 8s），显著降低 429 报错发生频率。
- **中文运行日志**
  - 密钥轮询、模型配置、上游调用、重试退避、权限报错、Token 统计等信息均使用中文实时说明。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `API_KEY` | 是 | `123456` | 保护本代理服务的 API Key。客户端请求本服务时使用 `Authorization: Bearer <API_KEY>`。 |
| `VERTEX_EXPRESS_API_KEY` | 否 | 空 | Gemini Express Mode API Key。多个 Key 用英文逗号分隔。标准模式下使用。 |
| `ROUNDROBIN` | 否 | `false` | `true` 表示多个 Express Key 按顺序轮询；`false` 表示随机选择。 |
| `FAKE_STREAMING` | 否 | `false` | `true` 时先用非流式请求上游，再向客户端模拟流式输出；图片模型会自动启用假流式保护。 |
| `FAKE_STREAMING_INTERVAL` | 否 | `1.0` | 假流式等待期间发送 keep-alive chunk 的间隔秒数。 |
| `MODELS_CONFIG_URL` | 否 | 仓库 `vertexModels.json` | 远程模型列表地址；默认从仓库 `vertexModels.json` 拉取，修改远程文件后无需重新部署即可刷新模型列表。 |
| `SAFETY_SCORE` | 否 | `false` | 是否把 Gemini safety ratings 附加到输出中。 |
| `PROXY_URL` | 否 | 空 | 上游 HTTP/HTTPS/SOCKS 代理。 |
| `SSL_CERT_FILE` | 否 | 空 | 自定义证书路径。 |
| `GOOGLE_COOKIE` | 否 | 空 | 网页反代直连模式下的 Google Cookie 字符串（初始化使用，后续可在后台随时更新）。 |
| `GOOGLE_PROJECT_ID` | 否 | 空 | 网页反代直连模式下的 Google Cloud 项目 ID（初始化使用，后续可在后台随时更新）。 |

---

## 网页反代直连模式配置指引 (支持手机与电脑)

如果您在管理大盘中切换到 **Agent Platform Studio (Cookie 直连反代)**，需要粘贴配置 **Cookie** 与 **Project ID**：

### 1. 获取完整的 Google Cookie
因为关键的会话校验凭证（如 `__Secure-1PSIDTS`、`__Secure-1PSID`）带有 `HttpOnly` 安全属性，无法通过一般的书签脚本提取，必须通过以下方式获取：
- **电脑端浏览器**：
  1. 打开并登录 [Google Cloud Console](https://console.cloud.google.com)。
  2. 按下 **F12** 键打开开发者工具，切换到 **Network (网络)** 标签页。
  3. 刷新一下页面，在左侧的请求列表中点击任意一个成功的请求。
  4. 找到 **Request Headers (请求头)** 中的 `Cookie:` 字段，复制其整段长字符串，粘贴到大盘对应输入框中。
- **手机端浏览器**：
  1. iOS (Safari) 或 Android (Kiwi Browser) 上安装免费插件 `Cookie-Editor`。
  2. 登录控制台后，点击 `Cookie-Editor` 插件，选择 **Export** 导出为 **Header String** 或 **JSON**。
  3. 直接将导出的字符串或 JSON 完整地粘贴到大盘 Cookie 输入框中，系统会自动解析和识别。

### 2. 获取 Google Project ID
- 在控制台顶部的项目选择器中复制项目 ID，或者直接复制您当前的浏览器地址栏网址 URL（形如：`https://console.cloud.google.com/vertex-ai/studio/multimodal?project=your-project-id`）。
- 粘贴到大盘 Project ID 输入框中，系统会自动提取出干净的项目 ID。

> ⚠️ **提示**：Google Cookie 通常较为持久——只要不退出登录、不修改密码、Google 未主动失效会话，一般可维持**数周甚至更久**（实测可用一个月以上），并非只有 1~2 小时。仅当接口确实报 `Permission Denied` / `predict denied` 时，重新获取并到大盘保存激活即可。

---

## 本地 Docker 运行

编辑 `docker-compose.yml`，设置所需的初始环境变量，例如：

```yaml
environment:
  - API_KEY=your_adapter_api_key
  - VERTEX_EXPRESS_API_KEY=your_vertex_express_api_key
```

启动服务：

```bash
docker compose up -d
```

默认将宿主机的 `8050` 端口映射到容器内 `7860`，通过以下地址访问控制大盘：

```text
http://localhost:8050
```

---

## 调用示例

### 查询模型

```bash
curl http://localhost:8050/v1/models \
  -H "Authorization: Bearer your_adapter_api_key"
```

### 非流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "用一句话介绍 Gemini Express Mode。"}
    ],
    "stream": false
  }'
```

### 流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "写一首短诗。"}
    ],
    "stream": true
  }'
```

### Google Search 搜索增强

只需在模型名后添加 `-search` 后缀即可开启：

```json
{
  "model": "gemini-2.5-flash-search",
  "messages": [
    {"role": "user", "content": "今天有哪些 Gemini API 相关更新？"}
  ]
}
```

---

## 模型列表配置

默认模型列表在远程 `MODELS_CONFIG_URL` 或本地 `vertexModels.json` 中配置：

```json
{
  "models": [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
  ]
}
```

调用 `/v1/models` 会自动为非生图的 Gemini 模型生成带有 `-search` 后缀的别名。

---

## 关于 429 报错与并发控制

429 (Resource Exhausted) 常由于上游限额不足或请求频率过高导致。本项目已内置请求退避重试方案，但建议：
- 控制客户端并发请求频率。
- 适当减少最大输出 Token 的大小。
- 采用多 Key 轮询机制：配置多个有效 API Key。
- 及时排查并更新已失效或权限受限的 Google Cookie。

---

## 本地开发与检查

常用检查命令：

```bash
python -m compileall app
```

如需本地手动启动开发环境：

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## 模型参数与控制台

- 控制台（浏览器打开根路径 `/`，仅需输入密码，即 `API_KEY`）提供"模型参数"面板，可在线调整：思考强度（3.x 档位 / 2.5 预算）、生图分辨率与默认比例、采样默认值、输入图压缩、重试次数与退避、假流式/轮询/安全分显示、预填充兼容模式。
- **参数优先级**：单次请求（OpenAI 请求体字段，如 `reasoning_effort`、`thinking_budget`、`image_size`、`aspect_ratio`，以及标准 `temperature`/`max_tokens` 等） > 控制台设置 > 内置默认。
- **按模型能力自动裁剪**（`app/model_capabilities.py`）：不同模型支持的参数不同，代理会自动为每个模型剥离其不支持的参数，避免上游报错。依据官方文档：
  - 自 **Gemini 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）**，`temperature`/`top_p`/`top_k` 已废弃（现忽略、未来 400），会被自动移除；更早的 3.x 仍可用但建议保持默认。
  - **所有 Gemini 3.x** 不支持 `candidate_count`；改用 `thinking_level`（而非 `thinking_budget`）。
  - **两个生图模型比例白名单不同**（pro-image 10 种 / flash-image 15 种），控制台按所选模型过滤，后端亦会校验并对不支持的比例回退为"由模型决定"。
  - **预填充**：Gemini 3.6+ 拒绝以 model 轮次结尾（400）；代理内置"预填充智能兼容"，与模型名无关，自动处理。

## 后续升级与扩展

- **新增模型**：只需把模型 ID 加入 `vertexModels.json`（或远程 `MODELS_CONFIG_URL`）。`model_capabilities.py` 按**家族/版本模式**自动归类（思考方式、采样裁剪、生图比例/分辨率、预填充），**未知/未来型号按"最新代"前向安全处理**（自动移除已废弃采样参数、要求 user 结尾）。因此加新模型基本即插即用，无需改代码。
- **迁移到 Interactions API**：代码按上游通道解耦（`app/upstreams/` 下各类实现 `BaseUpstream`；能力判定、消息转换、参数构建均为可复用模块）。待 Agent Platform 支持基础 Gemini 模型的 Interactions API 后，新增一个 `InteractionsUpstream`（实现 `BaseUpstream`）并在路由层接入即可，能力矩阵与预填充/参数逻辑可直接复用。当前 Agent Platform（Vertex）尚未对基础 Gemini 模型开放 Interactions（仅少数 agent），故暂未内置该通道。
