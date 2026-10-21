"""
search.py — 网络搜索模块

【模块概述】
本模块实现了网络搜索功能，使用 Tavily Search API 获取实时网络信息。
这是 AI Mentor 系统的"联网能力"组件，让 AI 能够回答需要实时信息的问题。

【核心功能】
1. 搜索决策：判断用户问题是否需要联网搜索
2. 搜索执行：调用 Tavily API 执行搜索
3. 结果格式化：将搜索结果转换为可注入 System Prompt 的格式

【搜索决策策略】
采用两阶段决策：
1. 规则快速判断：通过关键词模式匹配快速决策
2. LLM 精准判断：复杂场景下调用 AI 进行判断

【Tavily API 说明】
- 官网：https://tavily.com
- 免费额度：每月 1000 次搜索
- 特点：专为 AI 应用设计，返回结构化结果

【工作流程】
用户问题 → 规则判断 → (需要则)调用Tavily → 格式化结果 → 注入System Prompt
"""
from __future__ import annotations
import re, json, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """
    单条搜索结果
    
    属性：
        title: 搜索结果标题
        url: 结果链接地址
        content: 内容摘要（Tavily 返回的 snippet）
        score: 相关性分数（0-1，越高越相关）
    """
    title:   str
    url:     str
    content: str
    score:   float = 0.0


@dataclass
class SearchDecision:
    """
    搜索决策结果
    
    属性：
        should_search: 是否需要搜索
        query: 优化后的搜索词
        reason: 决策原因（用于调试）
        method: 决策方法（rule=规则，llm=大模型）
    """
    should_search: bool
    query:         str = ""
    reason:        str = ""
    method:        str = ""


@dataclass
class SearchContext:
    """
    搜索上下文（包含搜索结果）
    
    属性：
        query: 搜索查询词
        results: 搜索结果列表
        error: 错误信息（如果有）
    
    主要方法：
        to_prompt_block(): 将结果格式化为可注入 System Prompt 的文本
        to_dict(): 转换为字典格式，用于 JSON 响应
    """
    query:    str
    results:  List[SearchResult] = field(default_factory=list)
    error:    Optional[str] = None

    def to_prompt_block(self) -> str:
        """
        将搜索结果格式化为注入 System Prompt 的文本块
        
        格式示例：
            【网络搜索结果】查询：Python 最新版本
            
            [1] Python 3.12 Released
            URL: https://python.org/...
            Python 3.12 brings significant performance improvements...
            
            【使用说明】
            - 请基于以上搜索结果回答...
        """
        if self.error or not self.results:
            return ""
        lines = [f"【网络搜索结果】查询：{self.query}\n"]
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"[{i}] {r.title}\n"
                f"URL: {r.url}\n"
                f"{r.content[:400]}\n"
            )
        lines.append(
            "\n【使用说明】\n"
            "- 请基于以上搜索结果回答，信息来自互联网实时数据\n"
            "- 在回答末尾用「参考来源」部分列出引用的链接，格式：[标题](URL)\n"
            "- 如搜索结果与问题不完全相关，补充你自身的知识"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """将搜索上下文转换为字典格式，用于 API 响应"""
        return {
            "query":   self.query,
            "error":   self.error,
            "results": [
                {"title": r.title, "url": r.url,
                 "content": r.content[:200], "score": r.score}
                for r in self.results
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 搜索决策规则
# ═══════════════════════════════════════════════════════════════════════════════

_STRONG_PATTERNS = [
    r"(最新|最近|今年|今天|现在|当前|目前)(的|有什么|是什么|怎么样)",
    r"(2024|2025|2026)\s*年",
    r"(最新版本|新版|更新了|发布了|release|changelog)",
    r"(新闻|资讯|动态|进展|趋势)",
    r"(股价|价格|汇率|天气)",
    r"(刚刚|昨天|上周|上个月)(发生|发布|更新|推出)",
    r"(gpt-4o?|claude|gemini|llama|mistral|deepseek).*(版本|价格|更新|发布)",
    r"(openai|anthropic|google|meta|nvidia).*(发布|公告|新闻)",
]

_WEAK_PATTERNS = [
    r"(怎么安装|如何安装|安装教程)",
    r"(官方文档|官网|文档链接|下载地址)",
    r"(比较|对比|哪个更好|区别).*(框架|库|工具|模型)",
    r"(推荐|有没有好用的)(工具|框架|库|平台|网站)",
    r"paper|论文|arXiv",
]

def _rule_need_search(message: str) -> Optional[SearchDecision]:
    """
    基于规则的搜索决策
    
    工作原理：
    1. 检查消息是否匹配强触发模式（几乎一定需要搜索）
    2. 如果匹配，提取搜索词并返回决策
    
    参数：
        message: 用户消息
    
    返回：
        SearchDecision 或 None（需要 LLM 进一步判断）
    """
    msg = message.lower()
    for pat in _STRONG_PATTERNS:
        if re.search(pat, msg, re.IGNORECASE):
            query = _extract_query(message)
            return SearchDecision(
                should_search=True, query=query,
                reason="强规则命中：需要实时信息", method="rule"
            )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 搜索决策
# ═══════════════════════════════════════════════════════════════════════════════

_DECISION_PROMPT = """你是一个搜索决策器。判断用户的问题是否需要联网搜索获取实时/最新信息。

需要搜索的情况：
- 问题涉及最新版本、近期新闻、实时数据
- 问题涉及具体产品的当前价格/状态
- 问题需要2024年后的信息
- 问题明确要求"最新"、"现在"、"今天"等时效性内容

不需要搜索的情况：
- 概念解释、原理问答（不依赖时效）
- 代码编写、算法实现
- 学习路径规划
- 日常对话

请只返回 JSON（不要任何其他内容）：
{"need_search": true/false, "query": "优化后的搜索词（如不搜索则空字符串）", "reason": "一句话原因"}"""

def decide_search(
    message: str,
    api_key: str,
    base_url: str,
    model:   str,
    intent:  str = "qa",
) -> SearchDecision:
    """
    搜索决策主入口函数
    
    决策流程：
    1. chat/code 意图直接跳过搜索
    2. 规则强触发 → 直接搜索
    3. 其他情况 → LLM 判断
    
    参数：
        message: 用户消息
        api_key: AI API 密钥
        base_url: AI API 基础 URL
        model: 模型名称
        intent: 当前识别的意图
    
    返回：
        SearchDecision: 搜索决策结果
    
    示例：
        decision = decide_search(
            message="GPT-4 最新版本是什么",
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
            intent="qa"
        )
        print(decision.should_search)  # True
    """
    if intent in ("chat", "code"):
        return SearchDecision(should_search=False, reason="意图不需要实时信息", method="rule")

    rule_result = _rule_need_search(message)
    if rule_result:
        return rule_result

    if api_key and api_key != "your_api_key_here":
        try:
            return _llm_decide(message, api_key, base_url, model)
        except Exception as e:
            pass

    return SearchDecision(should_search=False, reason="默认不搜索", method="fallback")


def _llm_decide(message: str, api_key: str, base_url: str, model: str) -> SearchDecision:
    """
    使用 LLM 进行搜索决策
    
    工作流程：
    1. 构建决策提示词
    2. 调用 AI API
    3. 解析 JSON 响应
    
    参数：
        message: 用户消息
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称
    
    返回：
        SearchDecision: 决策结果
    """
    payload = json.dumps({
        "model":       model,
        "messages":    [
            {"role": "system", "content": _DECISION_PROMPT},
            {"role": "user",   "content": f"用户问题：{message}"},
        ],
        "temperature": 0.0,
        "max_tokens":  80,
        "stream":      False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw = data["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    obj = json.loads(raw)

    need   = bool(obj.get("need_search", False))
    query  = str(obj.get("query", message))
    reason = str(obj.get("reason", ""))
    return SearchDecision(should_search=need, query=query or message,
                          reason=reason, method="llm")


# ═══════════════════════════════════════════════════════════════════════════════
# Tavily 搜索 API
# ═══════════════════════════════════════════════════════════════════════════════

TAVILY_API_URL = "https://api.tavily.com/search"

def tavily_search(
    query:      str,
    api_key:    str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> SearchContext:
    """
    调用 Tavily Search API 执行搜索
    
    Tavily 是专为 AI 应用设计的搜索 API：
    - 返回结构化的搜索结果
    - 自动提取网页核心内容
    - 支持两种搜索深度：basic（快速）和 advanced（深入）
    
    参数：
        query: 搜索查询词
        api_key: Tavily API 密钥
        max_results: 最大返回结果数（默认 5）
        search_depth: 搜索深度（"basic" 或 "advanced"）
    
    返回：
        SearchContext: 包含搜索结果或错误信息
    
    示例：
        ctx = tavily_search(
            query="Python 3.12 新特性",
            api_key="tvly-xxx",
            max_results=5
        )
        for r in ctx.results:
            print(f"{r.title}: {r.url}")
    
    API 文档：
        https://docs.tavily.com/docs/tavily-api/rest_api
    """
    if not api_key or api_key == "your_tavily_key_here":
        return SearchContext(
            query=query,
            error="⚠️ 请在 .env 中配置 TAVILY_API_KEY（https://tavily.com 免费注册）"
        )

    payload = json.dumps({
        "api_key":      api_key,
        "query":        query,
        "max_results":  max_results,
        "search_depth": search_depth,
        "include_answer": False,
        "include_raw_content": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        TAVILY_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return SearchContext(query=query, error=f"Tavily API 错误({e.code}): {body[:200]}")
    except Exception as e:
        return SearchContext(query=query, error=f"搜索请求失败: {e}")

    results = []
    for item in data.get("results", []):
        results.append(SearchResult(
            title   = item.get("title", "无标题"),
            url     = item.get("url", ""),
            content = item.get("content", ""),
            score   = float(item.get("score", 0.0)),
        ))

    return SearchContext(query=query, results=results)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_query(message: str) -> str:
    """
    从用户消息中提取搜索词
    
    处理逻辑：
    1. 移除常见的疑问词前缀
    2. 移除末尾的标点符号
    3. 限制长度为 100 字符
    
    参数：
        message: 原始用户消息
    
    返回：
        str: 清理后的搜索词
    
    示例：
        _extract_query("请问你知道Python最新版本吗？")
        # 返回：Python最新版本
    """
    query = re.sub(r"(请问|你知道|帮我|我想知道|告诉我)[，,]?\s*", "", message)
    query = re.sub(r"[？?！!。，,]+$", "", query).strip()
    return query[:100] or message[:100]
