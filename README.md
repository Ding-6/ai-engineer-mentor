# AI Engineer Mentor

面向 AI 工程师的智能课程助手。集成**多模型管理、意图识别、知识库问答、文件解析、联网搜索**于一体；前端采用**豆包式**消息交互（复制整段 / 重新生成 / 点赞点踩 / 消息可编辑重发 / 停止生成）。

---

## 项目结构

```
myai/
├── app.py             # Flask 主程序（路由 / 流式 / 多 provider 调度）
├── run.py             # PyInstaller 打包入口（自动打开浏览器）
├── intent.py          # 意图识别（规则 + LLM）
├── rag.py             # RAG 向量检索（ChromaDB + API Embedding）
├── file_parser.py     # 多格式文件解析（PDF / DOCX / IPYNB / XLSX / CSV / PY）
├── search.py          # 网络搜索（Tavily API）
├── build.spec         # PyInstaller 配置
├── build.bat          # Windows 一键打包脚本
├── .env               # 环境变量
├── requirements.txt   # 依赖
├── mentor.db          # SQLite（自动生成）
├── chroma_db/         # 向量库（自动生成）
└── templates/
    ├── login.html     # 登录 / 注册
    ├── chat.html      # 主聊天界面（白底 + 模型下拉 + 豆包式消息操作）
    └── admin.html     # 管理后台（含「清空所有知识库」）
```

---

## 快速启动（开发模式）

```bash
# 1. 安装 conda 环境 + 依赖
bash setup.sh                     # 或在 Windows 上手动 conda create -n myai python=3.11 -y && conda run -n myai pip install -r requirements.txt

# 2. 编辑 .env，填入 AI_API_KEY 等
# 3. 启动
conda activate myai
python app.py
```

浏览器访问 **http://127.0.0.1:7860**
演示账号：`demo / demo123` · 管理员：`admin / admin888`

---

## 多模型管理（新增核心能力）

不再依赖单一 `.env`。**进入「⚙ 设置」即可添加任意 OpenAI 兼容端点**：

- 任意 `base_url`：OpenAI / DeepSeek / 智谱 / 通义千问 / Ollama / 自建网关 …
- 分别填写 `对话模型` 和 `Embedding 模型`
- 点「�� 从服务端拉取模型列表」自动读 `/models` 接口拿到全部可选模型
- 点击聊天顶部「�� 模型名」下拉即可即时切换
- 切换 Embedding 模型时，RAG 引擎会在下一次入库/检索前自动同步

每个用户互不影响，每个用户独立保存多套 API 配置。

---

## 豆包式消息交互

| 操作 | 触发方式 |
|------|---------|
| �� 复制整段 AI 回复 | hover AI 消息 → 点 �� |
| �� 重新生成 | hover AI 消息 → 点 ��（自动删最后一条 AI 再用相同上下文重发）|
| �� / �� 反馈 | hover AI 消息 → 点对应按钮 |
| ✏️ 编辑用户消息并重发 | hover 用户消息 → 点 ✏️ |
| �� 删除用户消息及其后全部 | hover 用户消息 → 点 �� |
| ⏹ 停止生成 | 生成中点输入框右侧按钮 |
| �� 切换模型 | 顶部「�� 模型名」下拉 |
| ⚙ 新增 / 编辑 / 删除模型 | 顶部「⚙ 设置」 |

每条 AI 消息下方还会显示当前使用的**模型 + 时间**。

---

## 知识库管理（管理员）

后台左侧导航新增「�� 知识库」入口：

- 顶部状态条显示 **向量集合数 / 累计片段 / 影响用户数**
- 一键清空所有用户知识库（需要输入大写 `YES` 二次确认）
- 仅清空向量库，不影响 SQLite 中的对话 / 消息 / 用户数据

---

## 打包成电脑安装包（Windows）

```bat
conda activate myai
pip install pyinstaller
build.bat
```

产物在 `dist\AIEngineerMentor\AIEngineerMentor.exe`，整个 `dist\AIEngineerMentor\` 目录可压缩发给任何人：

- 双击 `.exe` 自动启动 + 自动打开浏览器到 `http://127.0.0.1:7860`
- 关闭控制台窗口即停止服务
- 数据库与向量库存放在 exe 同级目录

---

## 功能一览

| 模块 | 能力 |
|------|------|
| 多模型管理 | 自定义 OpenAI 兼容端点 / 多套配置 / 顶部下拉切换 / 远程拉取模型列表 |
| 对话 | 多会话 · 自动命名 · 按今天/昨天/更早分组 |
| 消息 | 流式输出 / 复制整段 / 重新生成 / ���� / 消息可编辑 / 消息可删除 / 停止生成 |
| 知识库 | ChromaDB · 用户隔离 · 文件 / URL / 文本入库 · cosine 检索 · 来源标注 |
| 文件解析 | PDF / DOCX / IPYNB / PY / XLSX / CSV / TXT / MD · 预览卡片 |
| 联网搜索 | Tavily API · 顶栏开关 · 来源链接徽章 |
| 意图识别 | 规则匹配 → LLM 分类 · 5 类意图 + 自适应 system prompt |
| 管理后台 | 用户列表 / 封禁 / 重置密码 / 删除 / 查看对话 / **一键清空知识库** |

---

## API 端点速查

| 方法 | 路由 | 说明 |
|------|------|------|
| POST | `/api/login`、`/api/register`、`/api/logout` | 账号 |
| GET  | `/api/me` | 当前用户信息 |
| GET/POST/PUT/DELETE | `/api/providers[/{id}]` | 多模型配置 CRUD |
| POST | `/api/providers/{id}/activate` | 切换激活 provider |
| GET  | `/api/providers/{id}/models` | 拉取服务端模型列表 |
| GET/POST/DELETE | `/api/conversations[/{id}]` | 对话管理 |
| POST | `/api/conversations/{id}/stream` | 流式对话（核心） |
| POST | `/api/conversations/{id}/regenerate` | 重新生成最后一条 |
| POST | `/api/conversations/{id}/stop` | 主动停止生成 |
| GET  | `/api/conversations/{id}/messages` | 获取历史消息 |
| PUT/DELETE | `/api/messages/{id}` | 编辑 / 删除消息 |
| POST/GET | `/api/messages/{id}/feedback` | 反馈 |
| GET/POST/DELETE | `/api/rag/...` | 知识库 CRUD |
| POST | `/api/chat-upload` | 聊天内文件解析预览 |
| 管理员 | `/api/admin/...` | 用户 / 统计 / RAG |

---

## 常见问题

**Q: 切换 embedding 模型后旧的向量还在吗？**
A: 还在。不同 embedding 模型产生的向量不可比较，建议在「⚙ 设置 → 管理后台 → �� 知识库」里清空后重新入库。

**Q: 我能把 .env 删了吗？**
A: 第一次启动会自动把 .env 里的配置写入数据库（标为「默认」）。之后你可以在「⚙ 设置」里维护，不再需要 .env。

**Q: 打包后体积多大？**
A: 约 600 MB（含 chromadb / onnxruntime / tokenizers）。如需瘦身可删 `build.spec` 中不需要的 hiddenimports。

**Q: 流式中途想停止怎么办？**
A: 直接点输入框右侧的 ⏹