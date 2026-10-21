"""
AI Engineer Mentor - Flask App
Step 6: 网络搜索模块（Tavily API + AI 自主决策 + 来源融入回答）
"""
import sqlite3, json, os, hashlib, secrets, urllib.request, urllib.error, threading
from datetime import datetime
from flask import (Flask, render_template, request, jsonify,
                   session, Response, stream_with_context, redirect)

# ── .env 加载 ──────────────────────────────────────────────────────────────────
def load_dotenv(path=".env"):
    fp = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(fp): return
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

# ── 配置 ───────────────────────────────────────────────────────────────────────
AI_API_KEY   = os.environ.get("AI_API_KEY", "")
AI_BASE_URL  = os.environ.get("AI_BASE_URL", "https://vg.v1api.cc/v1").rstrip("/")
AI_MODEL     = os.environ.get("AI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "text-embedding-ada-002")
TAVILY_KEY   = os.environ.get("TAVILY_API_KEY", "")
CHROMA_DIR   = os.path.join(os.path.dirname(__file__), "chroma_db")
DB_PATH      = os.path.join(os.path.dirname(__file__), "mentor.db")

SYSTEM_PROMPT = """你是一位专业的 AI Engineer Mentor（AI 工程师导师），专注于帮助学习者掌握 AI 工程技能。

你的专业领域包括：机器学习、深度学习、Python 编程、数据工程、MLOps、LLM 应用开发、系统设计、技术面试备考。

回答风格：中文回答（英文提问则英文回答）、简洁清晰、适当用代码示例（markdown 格式）、鼓励动手实践。"""

RAG_SYSTEM_PROMPT = """你是一位专业的 AI Engineer Mentor。

【知识库检索结果】
以下是从用户知识库中检索到的相关内容，请优先基于这些内容回答，并在回答末尾标注参考来源：

{context}

【回答要求】
- 优先使用上方知识库内容作答
- 如知识库内容不足，可补充你自身的知识
- 在回答末尾用 📚 标注参考的来源文件名
- 中文回答，代码用 markdown 格式"""

# ── RAG 引擎（懒加载）─────────────────────────────────────────────────────────
_rag = None

def get_rag():
    """懒加载 RAG 引擎。若已登录用户配置了不同的 embed 模型，每次调用前同步过去。"""
    global _rag
    if _rag is None:
        try:
            from rag import RAGEngine
            _rag = RAGEngine(
                persist_dir=CHROMA_DIR,
                api_key=AI_API_KEY,
                base_url=AI_BASE_URL,
                embed_model=EMBED_MODEL,
            )
        except ImportError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)
    # 如果当前已激活 provider 的 embed 模型与 RAG 引擎不一致，按需同步
    try:
        if current_user():
            p = get_active_provider(current_user())
            if p and (p["embed_model"] != _rag.embed_model
                      or p["base_url"].rstrip("/") != _rag.base_url
                      or p["api_key"] != _rag.api_key):
                _rag.configure(p["api_key"], p["base_url"], p["embed_model"])
    except Exception:
        pass
    return _rag, None

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
# 文件上传最大 32MB
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ── 数据库 ─────────────────────────────────────────────────────────────────────
def get_db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            is_admin   INTEGER DEFAULT 0,
            is_banned  INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT DEFAULT '新对话',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        CREATE TABLE IF NOT EXISTS model_providers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            name         TEXT NOT NULL,
            base_url     TEXT NOT NULL,
            api_key      TEXT NOT NULL,
            chat_model   TEXT NOT NULL,
            embed_model  TEXT NOT NULL,
            is_default   INTEGER DEFAULT 0,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS message_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            feedback    TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(message_id, user_id),
            FOREIGN KEY (message_id) REFERENCES messages(id),
            FOREIGN KEY (user_id)    REFERENCES users(id)
        );
    """)
    # 兼容旧库：安全地补列（忽略已存在错误）
    for col, definition in [
        ("is_admin",  "INTEGER DEFAULT 0"),
        ("is_banned", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
        except Exception:
            pass

    # 首次为每个用户写入一条「默认」提供方（沿用 .env 配置），确保有可用模型
    for uid in [r[0] for r in conn.execute("SELECT id FROM users").fetchall()]:
        if not conn.execute(
            "SELECT 1 FROM model_providers WHERE user_id=? AND is_default=1", (uid,)
        ).fetchone():
            conn.execute(
                """INSERT INTO model_providers
                   (user_id,name,base_url,api_key,chat_model,embed_model,is_default)
                   VALUES (?,?,?,?,?,?,1)""",
                (uid, "默认（.env）", AI_BASE_URL, AI_API_KEY or "",
                 AI_MODEL, EMBED_MODEL)
            )

    pw_demo  = hash_pw("demo123")
    pw_admin = hash_pw("admin888")    # ← 管理员默认密码，上线前请修改
    conn.execute("INSERT OR IGNORE INTO users (username,password,is_admin) VALUES (?,?,0)", ("demo",  pw_demo))
    conn.execute("INSERT OR IGNORE INTO users (username,password,is_admin) VALUES (?,?,1)", ("admin", pw_admin))
    conn.commit(); conn.close()

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

# ── Auth ───────────────────────────────────────────────────────────────────────
def current_user():    return session.get("user_id")
def is_admin_session():return bool(session.get("is_admin"))

def require_login(fn):
    from functools import wraps
    @wraps(fn)
    def w(*a, **kw):
        if not current_user(): return jsonify({"error":"Unauthorized"}), 401
        return fn(*a, **kw)
    return w

def require_admin(fn):
    from functools import wraps
    @wraps(fn)
    def w(*a, **kw):
        if not current_user():      return jsonify({"error":"Unauthorized"}), 401
        if not is_admin_session():  return jsonify({"error":"Forbidden"}), 403
        return fn(*a, **kw)
    return w

# ── 页面 ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("chat.html") if current_user() else render_template("login.html")

# ── Auth API ───────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    d = request.json or {}
    u, p = (d.get("username") or "").strip(), (d.get("password") or "").strip()
    if not u or not p: return jsonify({"error":"用户名和密码不能为空"}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username,password) VALUES (?,?)", (u, hash_pw(p)))
        conn.commit(); return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        return jsonify({"error":"用户名已存在"}), 409
    finally: conn.close()

@app.route("/api/login", methods=["POST"])
def login():
    d = request.json or {}
    u, p = (d.get("username") or "").strip(), (d.get("password") or "").strip()
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=? AND password=?",
                       (u, hash_pw(p))).fetchone()
    conn.close()
    if not row:  return jsonify({"error":"用户名或密码错误"}), 401
    if row["is_banned"]: return jsonify({"error":"账号已被禁用，请联系管理员"}), 403
    session["user_id"]  = row["id"]
    session["username"] = row["username"]
    session["is_admin"] = bool(row["is_admin"])
    return jsonify({"ok": True, "username": row["username"],
                    "is_admin": bool(row["is_admin"])})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear(); return jsonify({"ok": True})

@app.route("/api/me")
def me():
    if not current_user(): return jsonify({"error":"Not logged in"}), 401
    return jsonify({"username": session.get("username"),
                    "is_admin": is_admin_session()})

@app.route("/admin")
def admin_page():
    if not current_user():     return redirect("/")
    if not is_admin_session(): return redirect("/")
    return render_template("admin.html")

# ── 对话管理 ───────────────────────────────────────────────────────────────────
@app.route("/api/conversations")
@require_login
def list_conversations():
    conn = get_db()
    rows = conn.execute(
        "SELECT id,title,created_at FROM conversations WHERE user_id=? ORDER BY created_at DESC",
        (current_user(),)
    ).fetchall(); conn.close()
    now = datetime.now()
    out = []
    for r in rows:
        d = (now - datetime.fromisoformat(r["created_at"])).days
        g = "今天" if d==0 else "昨天" if d==1 else "最近7天" if d<=7 else "更早"
        out.append({"id":r["id"],"title":r["title"],"group":g})
    return jsonify(out)

@app.route("/api/conversations", methods=["POST"])
@require_login
def create_conversation():
    conn = get_db()
    c = conn.execute("INSERT INTO conversations (user_id,title) VALUES (?,?)",
                     (current_user(),"新对话"))
    conn.commit(); cid = c.lastrowid; conn.close()
    return jsonify({"id":cid,"title":"新对话"})

@app.route("/api/conversations/<int:cid>", methods=["DELETE"])
@require_login
def delete_conversation(cid):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    conn.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, current_user()))
    conn.commit(); conn.close(); return jsonify({"ok":True})

@app.route("/api/conversations/<int:cid>/messages")
@require_login
def get_messages(cid):
    conn = get_db()
    rows = conn.execute(
        "SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,)
    ).fetchall(); conn.close()
    return jsonify([dict(r) for r in rows])

# ── 流式对话（意图识别 + 搜索决策 + RAG）────────────────────────────────────
@app.route("/api/conversations/<int:cid>/stream", methods=["POST"])
@require_login
def stream_message(cid):
    data     = request.json or {}
    user_msg = (data.get("message") or "").strip()
    use_rag  = data.get("use_rag", True)
    regen    = bool(data.get("regenerate", False))   # 重新生成模式：不再追加用户消息

    if not user_msg: return jsonify({"error":"消息不能为空"}), 400

    # ── 选择当前激活的 provider ─────────────────────────────────────────────
    provider = get_active_provider(current_user())
    if not provider:
        def _err():
            yield "data: " + json.dumps({"error":"⚠️ 请先在「⚙ 设置」中添加模型配置"}) + "\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")
    api_key    = provider["api_key"]
    base_url   = provider["base_url"].rstrip("/")
    chat_model = provider["chat_model"]

    if not api_key or api_key == "your_api_key_here":
        def _err():
            yield "data: " + json.dumps({"error":"⚠️ 当前模型未配置 API Key，请在「⚙ 设置」中补充"}) + "\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    # ── 保存用户消息 & 自动命名（regenerate 模式不再插入用户消息）─────────
    conn = get_db()
    if not regen:
        conn.execute("INSERT INTO messages (conversation_id,role,content) VALUES (?,'user',?)",
                     (cid, user_msg))
        cnt = conn.execute("SELECT COUNT(*) as c FROM messages WHERE conversation_id=?",
                           (cid,)).fetchone()["c"]
        if cnt <= 1:
            title = user_msg[:20] + ("…" if len(user_msg)>20 else "")
            conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
        conn.commit()

    hist = conn.execute(
        "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 20",
        (cid,)
    ).fetchall(); conn.close()
    hist_list = [{"role": r["role"], "content": r["content"]} for r in reversed(hist)]

    # ── Step 1：意图识别（使用当前 provider 的 chat_model）─────────────────────
    from intent import detect_intent
    intent_result = detect_intent(
        message  = user_msg,
        api_key  = api_key,
        base_url = base_url,
        model    = chat_model,
        history  = hist_list[:-1],
    )

    # ── Step 2：网络搜索（由前端开关控制，不做 AI 决策）─────────────────────
    from search import tavily_search, _extract_query
    use_search = data.get("use_search", False)   # 前端开关传入
    search_ctx = None
    if use_search:
        if TAVILY_KEY and TAVILY_KEY != "your_tavily_key_here":
            search_ctx = tavily_search(
                query        = _extract_query(user_msg),
                api_key      = TAVILY_KEY,
                max_results  = 5,
                search_depth = "basic",
            )
        # 无 Key 时 search_ctx 保持 None，generate() 里会推送警告

    # ── Step 3：RAG 检索（qa / learning / file_analysis，且未搜索时优先用 RAG）
    rag_chunks, rag_error = [], None
    rag_intents = {"qa", "learning", "file_analysis"}
    # 如果已经搜索到结果，RAG 作为补充；如果搜索失败，RAG 作为主力
    if use_rag and intent_result.intent in rag_intents:
        rag, err = get_rag()
        if rag:
            try:
                rag_chunks = rag.retrieve(current_user(), user_msg, top_k=3)
            except Exception as e:
                rag_error = str(e)
        elif err:
            rag_error = err

    # ── Step 4：构建 System Prompt（搜索 > RAG > 意图专属）────────────────────
    sys_prompt = _build_system_prompt(
        intent_result, search_ctx, rag_chunks
    )

    messages_payload = [{"role":"system","content":sys_prompt}]
    for row in hist_list:
        messages_payload.append({"role":row["role"],"content":row["content"]})

    temp_map = {"code":0.2, "chat":0.9, "learning":0.6, "qa":0.4, "file_analysis":0.4}
    temperature = temp_map.get(intent_result.intent, 0.7)

    # ── 取消事件：重新生成或新一次发送前先清空旧标志 ──────────────────────────
    cancel_ev = _cancel_event(current_user(), cid)
    cancel_ev.clear()

    # ── SSE 生成器 ─────────────────────────────────────────────────────────────
    def generate():
        # 1. 意图标签 + 当前模型信息
        yield "data: " + json.dumps({
            "intent":  intent_result.to_dict(),
            "model":   {"id": provider["id"], "name": provider["name"],
                        "chat_model": chat_model},
        }) + "\n\n"

        # 2. 搜索状态通知
        if use_search:
            if not TAVILY_KEY or TAVILY_KEY == "your_tavily_key_here":
                yield "data: " + json.dumps({
                    "search_warn": "TAVILY_API_KEY 未配置，请在 .env 中填入后重启"
                }) + "\n\n"
            elif search_ctx and search_ctx.error:
                yield "data: " + json.dumps({"search_warn": search_ctx.error}) + "\n\n"
            elif search_ctx and search_ctx.results:
                yield "data: " + json.dumps({
                    "search_results": search_ctx.to_dict()
                }) + "\n\n"

        # 3. RAG 来源引用
        if rag_chunks:
            refs = [{"source":c["source"],"score":c["score"]} for c in rag_chunks]
            yield "data: " + json.dumps({"rag_refs": refs}) + "\n\n"
        if rag_error:
            yield "data: " + json.dumps({"rag_warn": rag_error}) + "\n\n"

        # 4. 流式生成回答
        full_reply = []
        stopped = False
        try:
            payload = json.dumps({
                "model":       chat_model,
                "messages":    messages_payload,
                "stream":      True,
                "temperature": temperature,
                "max_tokens":  2048,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={"Content-Type":"application/json",
                         "Authorization":f"Bearer {api_key}"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw in resp:
                    if cancel_ev.is_set():
                        stopped = True
                        break
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data:"): continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]": break
                    try:
                        obj   = json.loads(chunk)
                        token = obj["choices"][0].get("delta",{}).get("content","")
                        if token:
                            full_reply.append(token)
                            yield "data: " + json.dumps({"token":token}) + "\n\n"
                    except: continue

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            yield "data: " + json.dumps({"error":f"API 错误({e.code}): {body[:300]}"}) + "\n\n"
            return
        except Exception as e:
            yield "data: " + json.dumps({"error":f"连接异常: {e}"}) + "\n\n"
            return

        # 5. 存完整回复 & 结束信号
        complete = "".join(full_reply)
        assistant_mid = None
        if complete:
            db = get_db()
            cur = db.execute(
                "INSERT INTO messages (conversation_id,role,content) VALUES (?,'assistant',?)",
                (cid, complete)
            )
            assistant_mid = cur.lastrowid
            db.commit(); db.close()

        if stopped:
            yield "data: " + json.dumps({"stopped": True,
                                          "message_id": assistant_mid}) + "\n\n"

        db2 = get_db()
        t = db2.execute("SELECT title FROM conversations WHERE id=?", (cid,)).fetchone()
        db2.close()
        yield "data: " + json.dumps({
            "done": True,
            "title": t["title"] if t else "新对话",
            "message_id": assistant_mid,
            "stopped": stopped,
        }) + "\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ── 重新生成最后一条 assistant 消息 ─────────────────────────────────────────
@app.route("/api/conversations/<int:cid>/regenerate", methods=["POST"])
@require_login
def regenerate_last(cid):
    """复用流式接口：删除最后一条 assistant 消息，再用同样的对话历史重新生成。"""
    conn = get_db()
    last = conn.execute(
        """SELECT id FROM messages WHERE conversation_id=? AND role='assistant'
           ORDER BY id DESC LIMIT 1""", (cid,)
    ).fetchone()
    if not last:
        conn.close(); return jsonify({"error":"当前对话没有可重新生成的回答"}), 400
    last_user = conn.execute(
        """SELECT content FROM messages WHERE conversation_id=? AND role='user'
           ORDER BY id DESC LIMIT 1""", (cid,)
    ).fetchone()
    conn.execute("DELETE FROM messages WHERE id=?", (last["id"],))
    conn.commit(); conn.close()
    if not last_user:
        return jsonify({"error":"未找到对应的用户消息"}), 400
    with app.test_request_context(
        f"/api/conversations/{cid}/stream",
        method="POST",
        json={"message": last_user["content"], "regenerate": True}
    ):
        return stream_message(cid)


# ── System Prompt 构建（搜索 > RAG > 意图专属）────────────────────────────────
def _build_system_prompt(intent_result, search_ctx, rag_chunks) -> str:
    parts = [intent_result.system_prompt]

    # 搜索结果块（最高优先级）
    if search_ctx and not search_ctx.error and search_ctx.results:
        parts.append("\n\n" + search_ctx.to_prompt_block())

    # RAG 知识库块（补充）
    if rag_chunks:
        ctx_parts = [
            f"[{i}] 来源：{c['source']}（相似度 {c['score']}）\n{c['content']}"
            for i, c in enumerate(rag_chunks, 1)
        ]
        rag_block = (
            "\n\n【知识库补充内容】\n"
            "以下是从本地知识库检索到的相关内容，可作为补充参考：\n\n"
            + "\n\n---\n\n".join(ctx_parts)
            + "\n\n请在回答末尾用 📚 标注引用的知识库来源。"
        )
        parts.append(rag_block)

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 模型配置中心
#   每个用户可以保存多套「API 提供方」(base_url + api_key + 模型组合)，
#   调用时按当前选中的 provider 走；用户在聊天顶部下拉里切换。
# ═══════════════════════════════════════════════════════════════════════════════
def list_providers(user_id: int):
    conn = get_db()
    rows = conn.execute(
        """SELECT id,name,base_url,api_key,chat_model,embed_model,is_default
           FROM model_providers WHERE user_id=? ORDER BY is_default DESC, id ASC""",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_active_provider(user_id: int) -> dict | None:
    """获取当前激活的 provider。优先级：session → DB 中 is_default=1 → 第一条 → None"""
    pid = session.get("active_provider_id")
    conn = get_db()
    if pid:
        row = conn.execute(
            "SELECT * FROM model_providers WHERE id=? AND user_id=?",
            (pid, user_id)
        ).fetchone()
        if row:
            conn.close(); return dict(row)
    row = conn.execute(
        """SELECT * FROM model_providers
           WHERE user_id=? ORDER BY is_default DESC, id ASC LIMIT 1""",
        (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def mask_key(k: str) -> str:
    if not k: return ""
    if len(k) <= 8: return "•" * len(k)
    return k[:4] + "…" + k[-4:]

@app.route("/api/providers", methods=["GET"])
@require_login
def api_list_providers():
    providers = list_providers(current_user())
    # API Key 返回给前端时仅显示前缀，避免泄露完整密钥
    for p in providers:
        p["api_key_masked"] = mask_key(p["api_key"])
        # 完整密钥仍保存在服务端，前端用 GET 列表时不返回原文
        p.pop("api_key", None)
    active = get_active_provider(current_user())
    return jsonify({
        "providers": providers,
        "active_id": active["id"] if active else None,
    })

@app.route("/api/providers", methods=["POST"])
@require_login
def api_create_provider():
    d = request.json or {}
    name       = (d.get("name") or "自定义").strip()
    base_url   = (d.get("base_url") or "").strip().rstrip("/")
    api_key    = (d.get("api_key") or "").strip()
    chat_model = (d.get("chat_model") or "").strip()
    embed_model= (d.get("embed_model") or "").strip()
    if not base_url or not api_key or not chat_model or not embed_model:
        return jsonify({"error":"base_url / api_key / chat_model / embed_model 均必填"}), 400

    conn = get_db()
    cur = conn.execute(
        """INSERT INTO model_providers
           (user_id,name,base_url,api_key,chat_model,embed_model,is_default)
           VALUES (?,?,?,?,?,?,0)""",
        (current_user(), name, base_url, api_key, chat_model, embed_model)
    )
    pid = cur.lastrowid
    conn.commit(); conn.close()
    return jsonify({"ok": True, "id": pid})

@app.route("/api/providers/<int:pid>", methods=["PUT"])
@require_login
def api_update_provider(pid):
    d = request.json or {}
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM model_providers WHERE id=? AND user_id=?",
        (pid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"配置不存在"}), 404
    fields = {
        "name":       (d.get("name")       or row["name"]).strip(),
        "base_url":   (d.get("base_url")   or row["base_url"]).strip().rstrip("/"),
        "api_key":    (d.get("api_key")    or row["api_key"]).strip(),
        "chat_model": (d.get("chat_model") or row["chat_model"]).strip(),
        "embed_model":(d.get("embed_model")or row["embed_model"]).strip(),
    }
    conn.execute(
        """UPDATE model_providers
           SET name=:name, base_url=:base_url, api_key=:api_key,
               chat_model=:chat_model, embed_model=:embed_model
           WHERE id=:id""", {**fields, "id": pid}
    )
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/providers/<int:pid>", methods=["DELETE"])
@require_login
def api_delete_provider(pid):
    conn = get_db()
    row = conn.execute(
        "SELECT is_default FROM model_providers WHERE id=? AND user_id=?",
        (pid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"配置不存在"}), 404
    if row["is_default"]:
        conn.close(); return jsonify({"error":"默认配置不可删除，请先切换或新建一个默认"}), 400
    conn.execute("DELETE FROM model_providers WHERE id=?", (pid,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/providers/<int:pid>/activate", methods=["POST"])
@require_login
def api_activate_provider(pid):
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM model_providers WHERE id=? AND user_id=?",
        (pid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"配置不存在"}), 404
    # 取消所有默认 → 选中这条设为默认
    conn.execute("UPDATE model_providers SET is_default=0 WHERE user_id=?", (current_user(),))
    conn.execute("UPDATE model_providers SET is_default=1 WHERE id=?", (pid,))
    conn.commit(); conn.close()
    session["active_provider_id"] = pid
    return jsonify({"ok": True, "active_id": pid})

@app.route("/api/providers/<int:pid>/models", methods=["GET"])
@require_login
def api_provider_models(pid):
    """拉取 provider 的 /models 列表（OpenAI 兼容），便于用户挑选模型。"""
    conn = get_db()
    row = conn.execute(
        "SELECT base_url,api_key FROM model_providers WHERE id=? AND user_id=?",
        (pid, current_user())
    ).fetchone()
    conn.close()
    if not row:
        return jsonify({"error":"配置不存在"}), 404
    base_url = row["base_url"].rstrip("/")
    api_key  = row["api_key"]
    try:
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "models": []})


# ═══════════════════════════════════════════════════════════════════════════════
# 知识库管理 — 管理后台一键清空所有用户知识库
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/admin/rag/clear", methods=["POST"])
@require_admin
def admin_rag_clear():
    """删除 chroma_db 下所有 user_* 集合（保留 sqlite 用户/对话/消息）。"""
    rag, err = get_rag()
    if not rag:
        return jsonify({"error": f"RAG 不可用: {err}"}), 503
    try:
        cleared = rag.clear_all()
        return jsonify({"ok": True, "cleared": cleared})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/rag/summary", methods=["GET"])
@require_admin
def admin_rag_summary():
    """返回知识库概况：集合数 / 累计片段 / 涉及用户数。"""
    rag, err = get_rag()
    if not rag:
        return jsonify({"error": f"RAG 不可用: {err}", "collections": 0,
                        "chunks": 0, "users": 0}), 200
    try:
        cols = rag.client.list_collections()
        user_collections = [c for c in cols if c.name.startswith("user_")]
        chunks = 0
        for c in user_collections:
            try:
                chunks += c.count()
            except Exception:
                pass
        # 涉及的 sqlite 用户数（有 chroma 集合的不一定都还有 sqlite 用户，仅作参考）
        user_ids = [int(c.name.split("_", 1)[1]) for c in user_collections
                    if c.name.split("_", 1)[1].isdigit()]
        return jsonify({
            "collections": len(user_collections),
            "chunks": chunks,
            "users": len(user_ids),
        })
    except Exception as e:
        return jsonify({"error": str(e), "collections": 0,
                        "chunks": 0, "users": 0}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# 消息级操作 — 反馈 / 停止生成 / 编辑重发
# ═══════════════════════════════════════════════════════════════════════════════
# 跨请求持有「取消标记」：key=(user_id, conv_id)；前端调 /api/conversations/<cid>/stop
# 来停止正在生成的 SSE 流。
_STREAM_FLAGS: dict[tuple[int, int], threading.Event] = {}

def _cancel_event(uid: int, cid: int) -> threading.Event:
    key = (uid, cid)
    ev = _STREAM_FLAGS.get(key)
    if ev is None:
        ev = threading.Event()
        _STREAM_FLAGS[key] = ev
    return ev

@app.route("/api/messages/<int:mid>/feedback", methods=["POST"])
@require_login
def api_message_feedback(mid):
    d = request.json or {}
    fb = (d.get("feedback") or "").strip().lower()
    if fb not in ("like", "dislike", "clear"):
        return jsonify({"error":"feedback 必须是 like / dislike / clear"}), 400
    conn = get_db()
    row = conn.execute(
        """SELECT m.id FROM messages m
           JOIN conversations c ON c.id = m.conversation_id
           WHERE m.id=? AND c.user_id=?""",
        (mid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"消息不存在"}), 404
    if fb == "clear":
        conn.execute("DELETE FROM message_feedback WHERE message_id=? AND user_id=?",
                     (mid, current_user()))
    else:
        conn.execute(
            """INSERT INTO message_feedback(message_id,user_id,feedback)
               VALUES(?,?,?)
               ON CONFLICT(message_id,user_id) DO UPDATE SET feedback=excluded.feedback""",
            (mid, current_user(), fb)
        )
    conn.commit(); conn.close()
    return jsonify({"ok": True, "feedback": fb})

@app.route("/api/messages/<int:mid>/feedback", methods=["GET"])
@require_login
def api_get_feedback(mid):
    conn = get_db()
    row = conn.execute(
        """SELECT feedback FROM message_feedback
           WHERE message_id=? AND user_id=?""", (mid, current_user())
    ).fetchone()
    conn.close()
    return jsonify({"feedback": row["feedback"] if row else None})

@app.route("/api/messages/<int:mid>", methods=["PUT"])
@require_login
def api_edit_message(mid):
    """编辑用户消息；用于「消息可编辑 / 删了重发」。"""
    d = request.json or {}
    new_content = (d.get("content") or "").strip()
    if not new_content:
        return jsonify({"error":"内容不能为空"}), 400
    conn = get_db()
    row = conn.execute(
        """SELECT m.* FROM messages m
           JOIN conversations c ON c.id = m.conversation_id
           WHERE m.id=? AND c.user_id=? AND m.role='user'""",
        (mid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"消息不存在或无权编辑"}), 404
    conn.execute("UPDATE messages SET content=? WHERE id=?", (new_content, mid))
    # 删除该消息之后所有消息（让前端重新生成）
    conn.execute("DELETE FROM messages WHERE conversation_id=? AND id>?",
                 (row["conversation_id"], mid))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/messages/<int:mid>", methods=["DELETE"])
@require_login
def api_delete_message(mid):
    """删除某条用户消息及其后续所有消息。"""
    conn = get_db()
    row = conn.execute(
        """SELECT m.* FROM messages m
           JOIN conversations c ON c.id = m.conversation_id
           WHERE m.id=? AND c.user_id=?""",
        (mid, current_user())
    ).fetchone()
    if not row:
        conn.close(); return jsonify({"error":"消息不存在"}), 404
    conn.execute("DELETE FROM messages WHERE conversation_id=? AND id>=?",
                 (row["conversation_id"], mid))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/conversations/<int:cid>/stop", methods=["POST"])
@require_login
def api_stop_stream(cid):
    """通知正在进行的 SSE 流停止生成。"""
    ev = _cancel_event(current_user(), cid)
    ev.set()
    return jsonify({"ok": True})



    parts = [intent_result.system_prompt]

    # 搜索结果块（最高优先级）
    if search_ctx and not search_ctx.error and search_ctx.results:
        parts.append("\n\n" + search_ctx.to_prompt_block())

    # RAG 知识库块（补充）
    if rag_chunks:
        ctx_parts = [
            f"[{i}] 来源：{c['source']}（相似度 {c['score']}）\n{c['content']}"
            for i, c in enumerate(rag_chunks, 1)
        ]
        rag_block = (
            "\n\n【知识库补充内容】\n"
            "以下是从本地知识库检索到的相关内容，可作为补充参考：\n\n"
            + "\n\n---\n\n".join(ctx_parts)
            + "\n\n请在回答末尾用 📚 标注引用的知识库来源。"
        )
        parts.append(rag_block)

    return "".join(parts)


# ── 搜索状态 API ────────────────────────────────────────────────────────────────
@app.route("/api/search/status")
@require_login
def search_status():
    return jsonify({
        "tavily_configured": bool(TAVILY_KEY and TAVILY_KEY != "your_tavily_key_here"),
        "tavily_key_prefix": TAVILY_KEY[:8] + "…" if TAVILY_KEY else "",
    })


# ── RAG 知识库 API ─────────────────────────────────────────────────────────────

@app.route("/api/rag/status")
@require_login
def rag_status():
    """返回 RAG 依赖状态和知识库统计"""
    from rag import check_deps
    deps  = check_deps()
    rag, err = get_rag()
    count = rag.total_chunks(current_user()) if rag else 0
    docs  = rag.list_documents(current_user()) if rag else []
    return jsonify({
        "deps": deps,
        "error": err,
        "total_chunks": count,
        "documents": docs,
        "embed_model": EMBED_MODEL,
    })

@app.route("/api/rag/upload", methods=["POST"])
@require_login
def rag_upload():
    """上传文件 → 深度解析 → 入知识库，返回解析预览"""
    rag, err = get_rag()
    if not rag: return jsonify({"error": f"RAG 不可用: {err}"}), 503

    f = request.files.get("file")
    if not f: return jsonify({"error":"未收到文件"}), 400

    filename = f.filename or "未知文件"
    raw = f.read()

    try:
        from file_parser import parse_file
        result = parse_file(raw, filename)
        if not result.full_text:
            return jsonify({"error": "文件内容为空，无法解析"}), 400

        rag_result = rag.add_document(
            current_user(), result.full_text, filename, result.file_type
        )
        return jsonify({
            "ok": True,
            "parse": result.to_dict(),
            **rag_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 文件解析端点（仅解析，不入库）──────────────────────────────────────────────
@app.route("/api/parse", methods=["POST"])
@require_login
def parse_only():
    """解析文件，返回结构化预览卡片数据（不入库）"""
    f = request.files.get("file")
    if not f: return jsonify({"error":"未收到文件"}), 400
    filename = f.filename or "未知文件"
    raw = f.read()
    try:
        from file_parser import parse_file
        result = parse_file(raw, filename)
        return jsonify({"ok": True, "parse": result.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 聊天内文件上传（仅解析 + 预览卡片，不入知识库）──────────────────────────
@app.route("/api/chat-upload", methods=["POST"])
@require_login
def chat_upload():
    """
    聊天输入框旁的上传入口：
    - 仅解析文件，生成预览卡片数据
    - 不自动入 RAG 知识库（如需入库请在「知识库」面板手动操作）
    """
    f = request.files.get("file")
    if not f: return jsonify({"error":"未收到文件"}), 400
    filename = f.filename or "未知文件"
    raw = f.read()

    try:
        from file_parser import parse_file
        result = parse_file(raw, filename)
    except Exception as e:
        return jsonify({"error": f"解析失败: {e}"}), 500

    return jsonify({
        "ok": True,
        "parse": result.to_dict(),
        "rag": None,   # 不入库，前端不显示知识库徽章
    })

@app.route("/api/rag/url", methods=["POST"])
@require_login
def rag_url():
    """抓取网页 URL 入库"""
    rag, err = get_rag()
    if not rag: return jsonify({"error": f"RAG 不可用: {err}"}), 503

    url = (request.json or {}).get("url", "").strip()
    if not url: return jsonify({"error":"URL 不能为空"}), 400

    try:
        from rag import RAGEngine
        content = RAGEngine.fetch_url(url)
        result  = rag.add_document(current_user(), content, url, "url")
        return jsonify({"ok":True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/text", methods=["POST"])
@require_login
def rag_text():
    """手动文本入库"""
    rag, err = get_rag()
    if not rag: return jsonify({"error": f"RAG 不可用: {err}"}), 503

    d       = request.json or {}
    content = (d.get("content") or "").strip()
    title   = (d.get("title") or "手动输入").strip()
    if not content: return jsonify({"error":"内容不能为空"}), 400

    try:
        result = rag.add_document(current_user(), content, title, "text")
        return jsonify({"ok":True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/documents/<doc_id>", methods=["DELETE"])
@require_login
def rag_delete(doc_id):
    rag, err = get_rag()
    if not rag: return jsonify({"error": f"RAG 不可用: {err}"}), 503
    try:
        rag.delete_document(current_user(), doc_id)
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 管理员 API ─────────────────────────────────────────────────────────────────

@app.route("/api/admin/stats")
@require_admin
def admin_stats():
    conn = get_db()
    stats = {
        "total_users":    conn.execute("SELECT COUNT(*) FROM users WHERE is_admin=0").fetchone()[0],
        "total_convs":    conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
        "total_messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "banned_users":   conn.execute("SELECT COUNT(*) FROM users WHERE is_banned=1").fetchone()[0],
    }
    conn.close()
    return jsonify(stats)


@app.route("/api/admin/users")
@require_admin
def admin_users():
    conn = get_db()
    rows = conn.execute("""
        SELECT u.id, u.username, u.is_admin, u.is_banned, u.created_at,
               COUNT(DISTINCT c.id)  AS conv_count,
               COUNT(DISTINCT m.id)  AS msg_count
        FROM users u
        LEFT JOIN conversations c ON c.user_id = u.id
        LEFT JOIN messages m      ON m.conversation_id = c.id
        WHERE u.is_admin = 0
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/users/<int:uid>/conversations")
@require_admin
def admin_user_convs(uid):
    conn = get_db()
    rows = conn.execute("""
        SELECT c.id, c.title, c.created_at,
               COUNT(m.id) AS msg_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.created_at DESC
    """, (uid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/conversations/<int:cid>/messages")
@require_admin
def admin_conv_messages(cid):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/users/<int:uid>/ban", methods=["POST"])
@require_admin
def admin_ban_user(uid):
    conn = get_db()
    # 不能封禁管理员
    row = conn.execute("SELECT is_admin FROM users WHERE id=?", (uid,)).fetchone()
    if not row:            conn.close(); return jsonify({"error":"用户不存在"}), 404
    if row["is_admin"]:   conn.close(); return jsonify({"error":"不能封禁管理员"}), 400
    conn.execute("UPDATE users SET is_banned=1 WHERE id=?", (uid,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:uid>/unban", methods=["POST"])
@require_admin
def admin_unban_user(uid):
    conn = get_db()
    conn.execute("UPDATE users SET is_banned=0 WHERE id=?", (uid,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:uid>/reset-password", methods=["POST"])
@require_admin
def admin_reset_password(uid):
    d = request.json or {}
    new_pw = (d.get("password") or "").strip()
    if len(new_pw) < 6: return jsonify({"error":"密码至少 6 位"}), 400
    conn = get_db()
    conn.execute("UPDATE users SET password=? WHERE id=?", (hash_pw(new_pw), uid))
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/admin/users/<int:uid>", methods=["DELETE"])
@require_admin
def admin_delete_user(uid):
    conn = get_db()
    row = conn.execute("SELECT is_admin FROM users WHERE id=?", (uid,)).fetchone()
    if not row:          conn.close(); return jsonify({"error":"用户不存在"}), 404
    if row["is_admin"]:  conn.close(); return jsonify({"error":"不能删除管理员"}), 400
    # 级联删除对话和消息
    conv_ids = [r[0] for r in conn.execute(
        "SELECT id FROM conversations WHERE user_id=?", (uid,)).fetchall()]
    for cid in conv_ids:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    conn.execute("DELETE FROM conversations WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})


@app.route("/api/admin/users", methods=["POST"])
@require_admin
def admin_create_user():
    d = request.json or {}
    u  = (d.get("username") or "").strip()
    p  = (d.get("password") or "").strip()
    if not u or len(p) < 6: return jsonify({"error":"用户名不能为空，密码至少 6 位"}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username,password) VALUES (?,?)", (u, hash_pw(p)))
        conn.commit()
        return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        return jsonify({"error":"用户名已存在"}), 409
    finally:
        conn.close()


# ── 启动 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    key_ok = AI_API_KEY and AI_API_KEY != "your_api_key_here"
    print("=" * 58)
    print("  🤖  AI Engineer Mentor  —  Step 6: Web Search")
    print("=" * 58)
    print(f"  API Key    : {'✅ 已配置' if key_ok else '❌ 未配置（请编辑 .env）'}")
    print(f"  Base URL   : {AI_BASE_URL}")
    print(f"  Chat Model : {AI_MODEL}")
    print(f"  Embed Model: {EMBED_MODEL}")
    search_ok = bool(TAVILY_KEY and TAVILY_KEY != "your_tavily_key_here")
    print(f"  Tavily     : {'✅ 已配置' if search_ok else '❌ 未配置（可选，填入后开启网络搜索）'}")
    print(f"  ChromaDB   : {CHROMA_DIR}")
    print(f"  地址       : http://127.0.0.1:7860")
    print(f"  演示账号   : demo / demo123")
    print("=" * 58)
    app.run(host="0.0.0.0", port=7860, debug=False)
