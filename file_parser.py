"""
file_parser.py — 多格式文件深度解析模块
支持PDF / DOCX / Python / IPYNB / Excel / CSV / TXT / MD
输出统一的 ParseResult 结构，用于预览卡片 + RAG 入库
"""
from __future__ import annotations
import io, os, re, json, csv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# ── 可选依赖检测 ──────────────────────────────────────────────────────────────
def _try_import(module: str):
    try:
        return __import__(module)
    except ImportError:
        return None

# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class TableInfo:
    title: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)   # 最多预览 5 行
    total_rows: int = 0

@dataclass
class CodeCell:
    cell_type: str = "code"   # code / markdown
    source: str = ""
    output_summary: str = ""  # 输出摘要

@dataclass
class ParseResult:
    filename: str = ""
    file_type: str = ""           # pdf / docx / ipynb / py / xlsx / csv / txt / md
    file_size_kb: int = 0

    # 统计
    char_count: int = 0
    page_count: int = 0           # PDF 页数
    section_count: int = 0        # DOCX 标题数 / IPYNB cell 数
    table_count: int = 0
    image_count: int = 0          # PDF/DOCX 图片数

    # 摘要
    summary: str = ""             # 内容前 500 字预览
    headings: List[str] = field(default_factory=list)   # 最多 10 个标题
    tables: List[TableInfo] = field(default_factory=list)  # 最多 3 张表

    # 代码专用
    code_cells: List[CodeCell] = field(default_factory=list)  # 最多 10 个 cell
    language: str = ""

    # 数据文件专用
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)  # 列统计

    # 用于 RAG 入库的完整文本
    full_text: str = ""

    # 解析状态
    success: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # tables → list of dicts
        d["tables"] = [asdict(t) for t in self.tables]
        d["code_cells"] = [asdict(c) for c in self.code_cells]
        return d


# ── 主解析入口 ────────────────────────────────────────────────────────────────

def parse_file(data: bytes, filename: str) -> ParseResult:
    """根据扩展名分派到对应解析器"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    res = ParseResult(filename=filename, file_type=ext,
                      file_size_kb=round(len(data) / 1024, 1))
    try:
        if ext == "pdf":
            _parse_pdf(data, res)
        elif ext == "docx":
            _parse_docx(data, res)
        elif ext == "ipynb":
            _parse_ipynb(data, res)
        elif ext == "py":
            _parse_python(data, res)
        elif ext in ("xlsx", "xls"):
            _parse_excel(data, res)
        elif ext == "csv":
            _parse_csv(data, res)
        elif ext in ("txt", "md", "html", "rst"):
            _parse_text(data, res)
        else:
            _parse_text(data, res)
    except Exception as e:
        res.success = False
        res.warnings.append(f"解析异常: {str(e)}")
        # 仍尝试提取纯文本
        try:
            res.full_text = data.decode("utf-8", errors="replace")
            res.char_count = len(res.full_text)
            res.summary = res.full_text[:500]
        except:
            pass
    return res


# ── PDF 解析 ──────────────────────────────────────────────────────────────────

def _parse_pdf(data: bytes, res: ParseResult):
    # 优先用 pdfplumber（更好的表格），回退 PyPDF2
    pdfplumber = _try_import("pdfplumber")
    if pdfplumber:
        _parse_pdf_plumber(data, res, pdfplumber)
    else:
        _parse_pdf_pypdf2(data, res)

def _parse_pdf_plumber(data: bytes, res: ParseResult, pdfplumber):
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        res.page_count = len(pdf.pages)
        full_parts = []
        tables_found = []

        for i, page in enumerate(pdf.pages):
            # 文字
            text = page.extract_text() or ""
            if text.strip():
                full_parts.append(f"[第{i+1}页]\n{text}")

            # 表格（最多收集 3 张）
            if len(tables_found) < 3:
                for tbl in page.extract_tables() or []:
                    if not tbl or len(tbl) < 2:
                        continue
                    headers = [str(c or "") for c in tbl[0]]
                    rows = [[str(c or "") for c in r] for r in tbl[1:6]]
                    tables_found.append(TableInfo(
                        title=f"第{i+1}页表格",
                        headers=headers,
                        rows=rows,
                        total_rows=len(tbl) - 1
                    ))

        res.full_text = "\n\n".join(full_parts)
        res.char_count = len(res.full_text)
        res.tables = tables_found
        res.table_count = len(tables_found)
        res.summary = _summarize(res.full_text)

        # 尝试提取标题行（全大写或短句）
        res.headings = _extract_pdf_headings(res.full_text)

def _parse_pdf_pypdf2(data: bytes, res: ParseResult):
    try:
        import PyPDF2
    except ImportError:
        try:
            import pypdf as PyPDF2
        except ImportError:
            raise ImportError("请安装 PyPDF2 或 pdfplumber: pip install PyPDF2")

    reader = PyPDF2.PdfReader(io.BytesIO(data))
    res.page_count = len(reader.pages)
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[第{i+1}页]\n{text}")

    res.full_text = "\n\n".join(parts)
    res.char_count = len(res.full_text)
    res.summary = _summarize(res.full_text)
    res.headings = _extract_pdf_headings(res.full_text)
    res.warnings.append("未安装 pdfplumber，表格提取不可用（pip install pdfplumber）")

def _extract_pdf_headings(text: str, max_n: int = 10) -> List[str]:
    headings = []
    for line in text.split("\n"):
        line = line.strip()
        is_heading = (
            line.isupper() or
            bool(re.match(r'^(\d+[\.\、]|第.章|Chapter\s)', line)) or
            (line.endswith(('：', ':')) and len(line) < 40)
        )
        if 4 < len(line) < 60 and is_heading:
            headings.append(line)
            if len(headings) >= max_n:
                break
    return headings


# ── DOCX 解析 ─────────────────────────────────────────────────────────────────

def _parse_docx(data: bytes, res: ParseResult):
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")

    doc = Document(io.BytesIO(data))
    parts = []
    headings = []
    image_count = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if "Heading" in style or style.startswith("标题"):
            headings.append(text)
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)

    # 表格
    tables_found = []
    for i, tbl in enumerate(doc.tables[:3]):
        if not tbl.rows:
            continue
        headers = [c.text.strip() for c in tbl.rows[0].cells]
        rows = [[c.text.strip() for c in r.cells] for r in tbl.rows[1:6]]
        tables_found.append(TableInfo(
            title=f"表格 {i+1}",
            headers=headers,
            rows=rows,
            total_rows=len(tbl.rows) - 1
        ))
        # 也加入全文
        parts.append("\n[表格]\n" + _table_to_text(headers, rows))

    # 统计图片（通过 XML 检测 drawing 元素）
    try:
        from docx.oxml.ns import qn
        image_count = len(doc.element.findall('.//' + qn('a:blip')))
    except:
        pass

    res.full_text = "\n".join(parts)
    res.char_count = len(res.full_text)
    res.headings = headings[:10]
    res.section_count = len(headings)
    res.tables = tables_found
    res.table_count = len(doc.tables)
    res.image_count = image_count
    res.summary = _summarize(res.full_text)


# ── IPYNB 解析 ────────────────────────────────────────────────────────────────

def _parse_ipynb(data: bytes, res: ParseResult):
    text = data.decode("utf-8", errors="replace")
    nb = json.loads(text)
    cells_raw = nb.get("cells", [])
    res.language = (nb.get("metadata", {})
                      .get("kernelspec", {})
                      .get("language", "python"))
    res.section_count = len(cells_raw)

    code_cells: List[CodeCell] = []
    full_parts = []
    md_headings = []

    for i, cell in enumerate(cells_raw):
        ctype  = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            # 提取 # 标题
            for line in source.split("\n"):
                if line.startswith("#"):
                    md_headings.append(line.lstrip("#").strip())
            full_parts.append(source)
            if len(code_cells) < 10:
                code_cells.append(CodeCell("markdown", source[:300], ""))

        elif ctype == "code":
            # 输出摘要
            outputs = cell.get("outputs", [])
            out_summary = ""
            for o in outputs[:2]:
                otype = o.get("output_type", "")
                if otype in ("stream", "execute_result", "display_data"):
                    txt = o.get("text", o.get("data", {}).get("text/plain", []))
                    if isinstance(txt, list):
                        txt = "".join(txt)
                    out_summary = str(txt)[:120]
                    break
                elif otype == "error":
                    out_summary = f"❌ {o.get('ename','Error')}: {o.get('evalue','')}"[:120]

            full_parts.append(f"# Cell {i+1}\n```python\n{source}\n```")
            if len(code_cells) < 10:
                code_cells.append(CodeCell("code", source[:400], out_summary))

    res.full_text = "\n\n".join(full_parts)
    res.char_count = len(res.full_text)
    res.code_cells = code_cells
    res.headings = md_headings[:10]
    res.summary = _summarize(res.full_text)


# ── Python 文件解析 ───────────────────────────────────────────────────────────

def _parse_python(data: bytes, res: ParseResult):
    text = data.decode("utf-8", errors="replace")
    res.full_text = text
    res.char_count = len(text)
    res.language = "python"

    lines = text.split("\n")
    functions, classes, imports = [], [], []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^(async\s+)?def\s+\w+', stripped):
            name = re.match(r'^(?:async\s+)?def\s+(\w+)', stripped).group(1)
            functions.append(f"def {name}() [L{i+1}]")
        elif re.match(r'^class\s+\w+', stripped):
            name = re.match(r'^class\s+(\w+)', stripped).group(1)
            classes.append(f"class {name} [L{i+1}]")
        elif stripped.startswith(("import ", "from ")):
            imports.append(stripped[:80])

    headings = (
        [f"📦 {c}" for c in classes[:5]] +
        [f"🔧 {f}" for f in functions[:8]]
    )
    res.headings = headings[:10]
    res.section_count = len(functions) + len(classes)

    # 用 CodeCell 展示结构
    summary_parts = []
    if imports:
        summary_parts.append("依赖：" + ", ".join(imports[:6]))
    if classes:
        summary_parts.append("类：" + ", ".join([c.split()[1] for c in classes]))
    if functions:
        summary_parts.append("函数：" + ", ".join([f.split()[1] for f in functions[:8]]))

    res.summary = "\n".join(summary_parts) or text[:400]

    # 分函数/类为 CodeCell
    # 用正则提取每个 def/class 块（前 10 个）
    blocks = list(re.finditer(r'^(?:async\s+)?(?:def|class)\s+\w+', text, re.MULTILINE))
    for j, match in enumerate(blocks[:10]):
        start = match.start()
        end = blocks[j+1].start() if j+1 < len(blocks) else start + 800
        snippet = text[start:end][:400]
        res.code_cells.append(CodeCell("code", snippet, ""))


# ── Excel 解析 ────────────────────────────────────────────────────────────────

def _parse_excel(data: bytes, res: ParseResult):
    # .xls (旧格式) 用 xlrd，.xlsx 用 openpyxl
    if res.file_type == "xls":
        _parse_xls(data, res)
        return

    openpyxl = _try_import("openpyxl")
    if not openpyxl:
        raise ImportError("请安装 openpyxl: pip install openpyxl")

    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as e:
        # openpyxl 失败时尝试用 xlrd 兜底（部分 .xlsx 实为 xls）
        res.warnings.append(f"openpyxl 读取失败，尝试 xlrd 兜底: {e}")
        _parse_xls(data, res)
        return
    _fill_excel_from_wb(wb, res)


def _parse_xls(data: bytes, res: ParseResult):
    """使用 xlrd 解析 .xls（老格式 Excel）"""
    xlrd = _try_import("xlrd")
    if not xlrd:
        raise ImportError(
            "解析 .xls 需要安装 xlrd：pip install xlrd\n"
            "建议将文件另存为 .xlsx 格式后重新上传"
        )
    wb = xlrd.open_workbook(file_contents=data)
    full_parts = []
    for sheet_idx in range(min(3, wb.nsheets)):
        ws = wb.sheet_by_index(sheet_idx)
        sheet_name = wb.sheet_names()[sheet_idx]
        if ws.nrows == 0:
            continue
        headers = [str(ws.cell_value(0, c)) for c in range(ws.ncols)]
        data_rows = []
        for r in range(1, min(ws.nrows, 1001)):
            data_rows.append([str(ws.cell_value(r, c)) for c in range(ws.ncols)])
        res.row_count += len(data_rows)

        # 数值统计
        col_stats = {}
        for ci, col_name in enumerate(headers):
            vals = []
            for r in data_rows:
                try:
                    vals.append(float(r[ci]))
                except:
                    pass
            if vals:
                col_stats[col_name] = {
                    "count": len(vals),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "mean": round(sum(vals) / len(vals), 4),
                }
        if col_stats:
            res.stats[sheet_name] = col_stats

        res.tables.append(TableInfo(
            title=sheet_name,
            headers=headers,
            rows=data_rows[:5],
            total_rows=len(data_rows),
        ))
        res.columns.extend(headers)
        full_parts.append(f"[Sheet: {sheet_name}]\n" + "\t".join(headers))
        for r in data_rows[:200]:
            full_parts.append("\t".join(r))

    res.full_text = "\n".join(full_parts)
    res.char_count = len(res.full_text)
    res.table_count = len(res.tables)
    res.section_count = wb.nsheets
    res.summary = _excel_summary(res)


def _fill_excel_from_wb(wb, res: ParseResult):
    """openpyxl workbook → ParseResult（.xlsx 专用）"""
    full_parts = []
    for sheet_name in wb.sheetnames[:3]:
        ws = wb[sheet_name]
        rows_data = []
        for row in ws.iter_rows(max_row=1000, values_only=True):
            if any(c is not None for c in row):
                rows_data.append([str(c) if c is not None else "" for c in row])
        if not rows_data:
            continue
        headers = rows_data[0]
        data_rows = rows_data[1:]
        res.row_count += len(data_rows)
        col_stats = {}
        for ci, col_name in enumerate(headers):
            vals = []
            for r in data_rows:
                if ci < len(r):
                    try:
                        vals.append(float(r[ci]))
                    except:
                        pass
            if vals:
                col_stats[col_name] = {
                    "count": len(vals),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "mean": round(sum(vals) / len(vals), 4),
                }
        if col_stats:
            res.stats[sheet_name] = col_stats
        res.tables.append(TableInfo(
            title=sheet_name, headers=headers,
            rows=[r for r in data_rows[:5]], total_rows=len(data_rows)
        ))
        res.columns.extend(headers)
        full_parts.append(f"[Sheet: {sheet_name}]\n" + "\t".join(headers))
        for r in data_rows[:200]:
            full_parts.append("\t".join(r))

    res.full_text = "\n".join(full_parts)
    res.char_count = len(res.full_text)
    res.table_count = len(res.tables)
    res.section_count = len(wb.sheetnames)
    res.summary = _excel_summary(res)


# ── CSV 解析 ──────────────────────────────────────────────────────────────────

def _parse_csv(data: bytes, res: ParseResult):
    text = data.decode("utf-8", errors="replace")

    # 自动检测分隔符
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(text[:2048])
        delimiter = dialect.delimiter
    except:
        delimiter = ","

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    all_rows = list(reader)
    if not all_rows:
        res.warnings.append("CSV 文件为空")
        return

    headers = all_rows[0]
    data_rows = all_rows[1:]
    res.columns = headers
    res.row_count = len(data_rows)

    # 数值列统计
    for ci, col_name in enumerate(headers):
        vals = []
        for r in data_rows:
            if ci < len(r):
                try:
                    vals.append(float(r[ci]))
                except:
                    pass
        if vals and len(vals) > len(data_rows) * 0.5:
            res.stats[col_name] = {
                "count": len(vals),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "mean": round(sum(vals)/len(vals), 4),
            }

    # 预览表格
    res.tables.append(TableInfo(
        title=res.filename,
        headers=headers,
        rows=[r for r in data_rows[:5]],
        total_rows=len(data_rows)
    ))

    # 全文（用于 RAG，最多 500 行）
    lines = [",".join(headers)]
    for r in data_rows[:500]:
        lines.append(",".join(r))
    res.full_text = "\n".join(lines)
    res.char_count = len(res.full_text)
    res.summary = _csv_summary(res)


# ── TXT / MD 解析 ─────────────────────────────────────────────────────────────

def _parse_text(data: bytes, res: ParseResult):
    for enc in ("utf-8", "gbk", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")

    res.full_text = text
    res.char_count = len(text)

    # 提取 Markdown 标题
    if res.file_type in ("md", "markdown"):
        for line in text.split("\n"):
            if line.startswith("#"):
                res.headings.append(line.lstrip("#").strip())
                if len(res.headings) >= 10:
                    break

    res.section_count = len(res.headings)
    res.summary = _summarize(text)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _summarize(text: str, max_len: int = 500) -> str:
    """取文本前 max_len 字符作为摘要，在句子边界截断"""
    text = text.strip()
    if len(text) <= max_len:
        return text
    chunk = text[:max_len]
    for sep in ["。", "！", "？", "\n", ". "]:
        pos = chunk.rfind(sep)
        if pos > max_len // 2:
            return chunk[:pos+1] + "…"
    return chunk + "…"

def _table_to_text(headers: List[str], rows: List[List[str]]) -> str:
    lines = [" | ".join(headers)]
    lines.append("-" * 40)
    for r in rows:
        lines.append(" | ".join(r))
    return "\n".join(lines)

def _excel_summary(res: ParseResult) -> str:
    parts = [f"共 {res.row_count} 行数据，{len(res.columns)} 列"]
    if res.columns:
        parts.append("列名：" + "、".join(res.columns[:8]))
    for sheet, cols in list(res.stats.items())[:2]:
        col_summaries = []
        for col, s in list(cols.items())[:4]:
            col_summaries.append(f"{col}(均值:{s['mean']})")
        if col_summaries:
            parts.append(f"{sheet}：" + "，".join(col_summaries))
    return "\n".join(parts)

def _csv_summary(res: ParseResult) -> str:
    parts = [f"共 {res.row_count} 行 × {len(res.columns)} 列"]
    if res.columns:
        parts.append("字段：" + "、".join(res.columns[:10]))
    num_stats = [(c, s) for c, s in list(res.stats.items())[:4]]
    if num_stats:
        parts.append("数值列统计：")
        for col, s in num_stats:
            parts.append(f"  {col}: 均值={s['mean']}, 范围=[{s['min']}, {s['max']}]")
    return "\n".join(parts)
