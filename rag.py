"""
RAG engine
- Vector store: ChromaDB persistent collections
- Embeddings: OpenAI-compatible /embeddings API
- Input sources: text, files, URLs
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Dict, List

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    try:
        import pypdf as PyPDF2
        HAS_PDF = True
    except ImportError:
        HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class RAGEngine:
    """Each user gets an isolated Chroma collection."""

    def __init__(self, persist_dir: str, api_key: str, base_url: str, embed_model: str):
        if not HAS_CHROMA:
            raise ImportError("请先安装 chromadb: pip install chromadb")

        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.embed_model = embed_model

        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

    # 让 RAG 引擎在切换 embedding 模型/提供方后使用新配置（注意：collection 内的向量不会自动重算，仅影响后续入库/检索）
    def configure(self, api_key: str, base_url: str, embed_model: str):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.embed_model = embed_model

    def clear_all(self) -> int:
        """删除所有 user_* 集合，返回被删除的集合数。供管理后台使用。"""
        deleted = 0
        try:
            for col in self.client.list_collections():
                if col.name.startswith("user_"):
                    try:
                        self.client.delete_collection(col.name)
                        deleted += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return deleted

    def _collection(self, user_id: int):
        return self.client.get_or_create_collection(
            name=f"user_{user_id}",
            metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 128},
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("AI_API_KEY 未配置，无法生成 embedding")
        if not self.base_url:
            raise ValueError("AI_BASE_URL 未配置，无法调用 embeddings 接口")
        if not self.embed_model:
            raise ValueError("EMBED_MODEL 未配置，无法生成 embedding")

        import time as _time
        
        # 分批 embedding，每批最多 5 个，避免超时
        BATCH_SIZE = 5
        all_embeddings = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(total_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            batch = texts[start:end]

            payload = json.dumps({
                "model": self.embed_model,
                "input": batch,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )

            # 重试逻辑：最多重试 3 次，间隔递增
            data = None
            for attempt in range(1, 4):
                try:
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"embedding API 错误({e.code}): {body[:300]}")
                except urllib.error.URLError as e:
                    reason = str(e.reason)
                    if "timed out" in reason.lower():
                        if attempt < 3:
                            wait_time = 10 * attempt
                            print(f"    ⚠️ 超时，等待 {wait_time} 秒后重试...", flush=True)
                            _time.sleep(wait_time)
                            continue
                        raise TimeoutError("embedding 请求超时，文档内容可能过多，请减少文件大小后重试")
                    raise RuntimeError(f"embedding 服务连接失败: {e.reason}")
                except Exception as e:
                    if attempt < 3:
                        wait_time = 10 * attempt
                        print(f"    ⚠️ 错误: {e}，等待 {wait_time} 秒后重试...", flush=True)
                        _time.sleep(wait_time)
                        continue
                    raise

            if data is None:
                raise RuntimeError("embedding 请求失败")

            if "data" not in data:
                raise RuntimeError("embedding 响应缺少 data 字段")

            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
            print(f"  ⏳ Embedding progress: {end}/{len(texts)} chunks", flush=True)

        return all_embeddings

    @staticmethod
    def _chunk(text: str, size: int = 800, overlap: int = 100) -> List[str]:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        if size <= 0:
            raise ValueError("chunk size 必须大于 0")
        if overlap < 0:
            overlap = 0
        if overlap >= size:
            overlap = max(0, size // 5)

        separators = ["。", "！", "？", "；", "\n", ". ", "! ", "? ", "; ", ", ", "，"]
        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + size, len(text))
            actual_end = end
            chunk = text[start:actual_end]

            if end < len(text):
                for sep in separators:
                    pos = chunk.rfind(sep)
                    if pos > size // 2:
                        actual_end = start + pos + len(sep)
                        chunk = text[start:actual_end]
                        break

            chunk = chunk.strip()
            if chunk and (len(chunk) > 20 or not chunks):
                chunks.append(chunk)

            next_start = actual_end - overlap
            if next_start <= start:
                next_start = end
            if next_start <= start:
                break
            start = next_start

        return chunks

    def add_document(self, user_id: int, content: str, source: str, doc_type: str) -> Dict:
        chunks = self._chunk(content)
        if not chunks:
            raise ValueError("文档解析后内容为空，无法入库")

        embeddings = self._embed(chunks)
        col = self._collection(user_id)
        doc_id = str(uuid.uuid4())

        # 分批写入 ChromaDB，避免 HNSW 索引 compaction 错误
        # 批次大小必须足够小，否则 HNSW 索引在 compaction 时会损坏
        import time as _time
        CHROMA_BATCH = 5
        for batch_start in range(0, len(chunks), CHROMA_BATCH):
            batch_end = min(batch_start + CHROMA_BATCH, len(chunks))
            col.add(
                ids=[f"{doc_id}_{i}" for i in range(batch_start, batch_end)],
                embeddings=embeddings[batch_start:batch_end],
                documents=chunks[batch_start:batch_end],
                metadatas=[
                    {
                        "source": source,
                        "type": doc_type,
                        "doc_id": doc_id,
                        "chunk_idx": i,
                    }
                    for i in range(batch_start, batch_end)
                ],
            )
            if batch_end < len(chunks):
                _time.sleep(1)

        return {
            "doc_id": doc_id,
            "source": source,
            "type": doc_type,
            "chunks": len(chunks),
        }

    def retrieve(self, user_id: int, query: str, top_k: int = 4) -> List[Dict]:
        col = self._collection(user_id)
        if col.count() == 0:
            return []

        q_vec = self._embed([query])[0]
        n = min(top_k, col.count())
        res = col.query(
            query_embeddings=[q_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict] = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            meta = meta or {}
            if not doc:
                continue
            if dist < 0.45:
                results.append({
                    "content": doc,
                    "source": meta.get("source", "未知来源"),
                    "type": meta.get("type", ""),
                    "score": round(1 - dist, 3),
                })
        return results

    def list_documents(self, user_id: int) -> List[Dict]:
        col = self._collection(user_id)
        try:
            if col.count() == 0:
                return []
        except Exception:
            return []

        res = col.get(include=["metadatas"])
        seen: Dict[str, Dict] = {}
        for meta in res["metadatas"]:
            if not meta:
                continue
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", "未知"),
                    "type": meta.get("type", ""),
                    "chunks": 0,
                }
            seen[doc_id]["chunks"] += 1
        return list(seen.values())

    def delete_document(self, user_id: int, doc_id: str):
        col = self._collection(user_id)
        res = col.get(where={"doc_id": doc_id})
        if res["ids"]:
            col.delete(ids=res["ids"])

    def total_chunks(self, user_id: int) -> int:
        try:
            return self._collection(user_id).count()
        except Exception:
            return 0

    @staticmethod
    def parse_txt(data: bytes) -> str:
        for enc in ("utf-8", "gbk", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def parse_pdf(data: bytes) -> str:
        if not HAS_PDF:
            raise ImportError("请安装 PyPDF2: pip install PyPDF2")
        import io

        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def parse_docx(data: bytes) -> str:
        if not HAS_DOCX:
            raise ImportError("请安装 python-docx: pip install python-docx")
        import io

        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def fetch_url(url: str) -> str:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=1000) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if "timed out" in reason.lower() or "time" in reason.lower():
                raise TimeoutError(f"请求超时，请检查网址是否可访问或稍后再试: {url}")
            raise ConnectionError(f"无法连接到 {url}: {e.reason}")
        except TimeoutError:
            raise TimeoutError(f"请求超时，请检查网址是否可访问或稍后再试: {url}")

        html = raw.decode("utf-8", errors="replace")
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        return re.sub(r"<[^>]+>", " ", html)


def check_deps() -> Dict[str, bool]:
    return {
        "chromadb": HAS_CHROMA,
        "PyPDF2": HAS_PDF,
        "python-docx": HAS_DOCX,
        "beautifulsoup4": HAS_BS4,
    }
