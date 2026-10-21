"""
batch_import.py — 批量导入论文到 RAG 知识库

使用方法：
1. 将论文 PDF 文件放入 rag_papers 文件夹
2. 运行：python batch_import.py

注意：需要先配置好 .env 文件中的 API Key
"""
import os
import time
from file_parser import parse_file
from rag import RAGEngine

# ── 加载环境变量 ────────────────────────────────────────────────────────────────
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
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "text-embedding-ada-002")
CHROMA_DIR   = os.path.join(os.path.dirname(__file__), "chroma_db")

# ── 初始化 RAG 引擎 ────────────────────────────────────────────────────────────
rag = RAGEngine(
    persist_dir=CHROMA_DIR,
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL,
    embed_model=EMBED_MODEL,
)
user_id = 1  # 使用 demo 用户 ID

# ── 批量导入 ────────────────────────────────────────────────────────────────────
papers_dir = "rag_papers"

if not os.path.exists(papers_dir):
    print(f"❌ 文件夹不存在: {papers_dir}")
    print("请先创建 rag_papers 文件夹并放入 PDF 文件")
    exit(1)

pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
if not pdf_files:
    print(f"❌ 在 {papers_dir} 中没有找到 PDF 文件")
    exit(1)

print(f"📚 找到 {len(pdf_files)} 个 PDF 文件")
print("=" * 50)

success_count = 0
for idx, filename in enumerate(pdf_files, 1):
    file_path = os.path.join(papers_dir, filename)
    print(f"[{idx}/{len(pdf_files)}] 正在处理: {filename}")
    
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            parse_result = parse_file(data, filename)
        
        if not parse_result.full_text:
            print(f"  ⚠️ 文件内容为空，跳过")
            continue
        
        # 重试逻辑：最多重试 2 次
        for attempt in range(1, 3):
            try:
                result = rag.add_document(
                    user_id=user_id,
                    content=parse_result.full_text,
                    source=filename,
                    doc_type="pdf"
                )
                print(f"  ✅ 入库成功，文档ID: {result['doc_id']}，片段数: {result['chunks']}")
                success_count += 1
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  ⚠️ 第{attempt}次尝试失败，等待后重试...")
                    time.sleep(15)
                else:
                    raise
        
        time.sleep(10)  # 让 ChromaDB 有足够时间完成 compaction
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")

print("=" * 50)
print(f"🎉 导入完成！成功: {success_count}/{len(pdf_files)}")
