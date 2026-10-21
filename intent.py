"""
intent.py — 意图识别模块

【模块概述】
本模块负责识别用户输入的意图类型，以便系统能够针对不同类型的请求
提供最优的回答策略。这是构建智能对话系统的关键组件。

【识别策略】
采用两阶段混合策略：
1. 规则快速匹配：使用正则表达式关键词匹配，速度快但覆盖有限
2. LLM 精准分类：当规则置信度不足时，调用大语言模型进行精准判断

【支持的意图类型】
- learning：学习规划（课程推荐、学习路径、知识体系）
- qa：知识问答（概念解释、原理问答、知识检索）
- code：代码助手（代码生成、调试、解释、优化）
- file_analysis：文件解析（文件分析、数据解读、文档摘要）
- chat：日常闲聊（问候、闲聊、非技术对话）

【工作流程】
用户消息 → 规则匹配 → 置信度判断 → (不足则)LLM分类 → 返回意图结果
"""
from __future__ import annotations
import re, json, urllib.request, urllib.error
from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# 意图定义配置
# ═══════════════════════════════════════════════════════════════════════════════

INTENTS = {
    "learning": {
        "label":  "学习规划",
        "emoji":  "🎓",
        "color":  "#a78bfa",
        "desc":   "学习路径 / 课程推荐 / 知识体系",
    },
    "qa": {
        "label":  "知识问答",
        "emoji":  "💡",
        "color":  "#38bdf8",
        "desc":   "概念解释 / 原理问答 / 知识检索",
    },
    "code": {
        "label":  "代码助手",
        "emoji":  "🐍",
        "color":  "#34d399",
        "desc":   "代码生成 / 调试 / 解释 / 优化",
    },
    "file_analysis": {
        "label":  "文件解析",
        "emoji":  "📄",
        "color":  "#fb923c",
        "desc":   "文件分析 / 数据解读 / 文档摘要",
    },
    "chat": {
        "label":  "日常闲聊",
        "emoji":  "💬",
        "color":  "#94a3b8",
        "desc":   "问候 / 闲聊 / 非技术对话",
    },
}

INTENT_PROMPTS = {
    "learning": """你是一位专业的 AI Engineer Mentor，当前任务：为学习者制定个性化学习规划。

回答策略：
- 先了解学习者当前水平和目标
- 给出清晰的阶段性学习路径（初级→中级→高级）
- 每个阶段推荐 2-3 个具体资源（书籍/课程/项目）
- 给出可执行的第一步行动
- 用结构化格式（标题+列表）呈现""",

    "qa": """你是一位专业的 AI Engineer Mentor，当前任务：精准解答技术问题。

回答策略：
- 先给出简洁的直接答案（1-2句）
- 再展开解释核心原理
- 适当用类比帮助理解
- 如有相关知识库内容，优先引用并注明来源
- 结尾提示可以继续深入的方向""",

    "code": """你是一位专业的 AI Engineer Mentor，当前任务：提供代码帮助。

回答策略：
- 直接给出可运行的代码（用 ```python 包裹）
- 关键行加注释解释逻辑
- 说明代码的核心思路（2-3句）
- 如有更优方案，附上对比说明
- 提示潜在的边界情况或注意事项""",

    "file_analysis": """你是一位专业的 AI Engineer Mentor，当前任务：帮助分析和解读文件内容。

回答策略：
- 先给出文件内容的整体概括
- 提取关键信息、数据规律或核心逻辑
- 如果是数据文件，指出有趣的统计规律
- 如果是代码，分析结构和实现思路
- 给出具体的改进建议或延伸问题""",

    "chat": """你是一位友好的 AI Engineer Mentor，当前是轻松的日常对话。

回答策略：
- 自然友好地回应，不过度技术化
- 简短为主，1-3句话
- 适时引导到 AI/编程相关的话题
- 保持轻松愉快的语调""",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 规则关键词表
# ═══════════════════════════════════════════════════════════════════════════════

_RULES = {
    "learning": [
        r"学习(路径|计划|规划|方法|顺序|步骤)",
        r"(怎么|如何|从哪里)(开始|入门|学习|学)",
        r"(推荐|有哪些)(课程|书|资源|教程|视频)",
        r"(学习|掌握|学会)(机器学习|深度学习|ai|人工智能|python|nlp|cv|大模型)",
        r"(初学者|新手|零基础|入门)",
        r"(知识体系|技能树|路线图|roadmap)",
        r"需要学(哪些|什么)",
        r"(先学|再学|然后学)",
        r"学多久|需要多长时间",
    ],
    "code": [
        r"(写|生成|实现|创建|帮我)(代码|函数|类|脚本|程序|算法)",
        r"(代码|程序)(报错|错误|异常|bug|问题|不对)",
        r"(怎么|如何)(实现|写|用python|用代码|编写)",
        r"```",
        r"(调试|debug|fix|修复)(代码|bug|错误)",
        r"(优化|重构|改进)(代码|函数|性能|算法)",
        r"(import |def |class |print\(|for .+ in )",
        r"(语法|用法|参数)(是什么|怎么用|有哪些)",
        r"帮(我|忙)(写|实现|做)(个|一个|一段)",
        r"(快速排序|冒泡|二分|递归|迭代|循环)",
        r"(函数|方法|接口)(怎么写|如何实现|示例)",
    ],
    "file_analysis": [
        r"(这个|这份|上传的|附件)(文件|文档|表格|代码|数据|图表)",
        r"(分析|解读|解析|总结|提取)(文件|文档|数据|内容|表格)",
        r"(帮我看|帮我分析|看一下|解释一下)(这个|这份|这段)",
        r"\.(pdf|docx|xlsx|csv|ipynb|py)\b",
        r"(数据里|表格里|文档里)(有什么|说了|显示)",
    ],
    "chat": [
        r"^(你好|hi|hello|嗨|hey|哈喽)[!！。，\s]*$",
        r"^(谢谢|感谢|thanks|thank you|多谢)[!！。，\s]*$",
        r"(你是谁|你叫什么|你能做什么|介绍一下你自己|你是什么)",
        r"^(好的|明白|知道了|收到|了解|ok|好)[!！。，\s]*$",
        r"^(再见|拜拜|bye|goodbye)[!！。，\s]*$",
        r"(天气|吃饭|心情|今天怎么样)",
        r"^.{1,8}[吗呢啊哦哈嗯]{1}[？?！!。,，]*$",
    ],
    "qa": [],
}


@dataclass
class IntentResult:
    """
    意图识别结果数据类
    
    属性说明：
        intent: 意图标识符（learning/qa/code/file_analysis/chat）
        confidence: 置信度（0.0-1.0），越高表示越确定
        method: 识别方法（rule=规则匹配，llm=大模型分类，fallback=默认兜底）
        label: 意图中文名称
        emoji: 意图对应的表情符号
        color: 意图对应的颜色（用于前端显示）
        system_prompt: 该意图对应的系统提示词
    
    使用示例：
        result = IntentResult(intent="code", confidence=0.85, method="rule")
        print(result.label)  # 输出：代码助手
    """
    intent: str
    confidence: float
    method: str
    label: str = ""
    emoji: str = ""
    color: str = ""
    system_prompt: str = ""

    def __post_init__(self):
        info = INTENTS.get(self.intent, INTENTS["qa"])
        self.label  = info["label"]
        self.emoji  = info["emoji"]
        self.color  = info["color"]
        self.system_prompt = INTENT_PROMPTS.get(self.intent, INTENT_PROMPTS["qa"])

    def to_dict(self) -> dict:
        """将结果转换为字典格式，用于 JSON 响应"""
        return {
            "intent":    self.intent,
            "label":     self.label,
            "emoji":     self.emoji,
            "color":     self.color,
            "confidence": round(self.confidence, 2),
            "method":    self.method,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════════════════════════════════

def detect_intent(
    message: str,
    api_key: str,
    base_url: str,
    model: str,
    history: list | None = None,
) -> IntentResult:
    """
    意图识别主入口函数
    
    采用两阶段策略：
    1. 先使用规则快速匹配（速度快，无 API 调用成本）
    2. 如果规则置信度 < 0.75，则调用 LLM 进行精准分类
    
    参数：
        message: 用户输入的消息文本
        api_key: AI API 密钥
        base_url: AI API 基础 URL
        model: 使用的模型名称
        history: 对话历史（最近几轮），用于上下文理解
    
    返回：
        IntentResult: 包含意图、置信度、方法等信息的结果对象
    
    示例：
        result = detect_intent(
            message="帮我写一个快速排序算法",
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo"
        )
        print(result.intent)  # 输出：code
    """
    rule_result = _rule_match(message)
    if rule_result and rule_result.confidence >= 0.75:
        return rule_result

    if api_key and api_key != "your_api_key_here":
        try:
            llm_result = _llm_classify(message, api_key, base_url, model, history or [])
            if llm_result:
                return llm_result
        except Exception:
            pass

    if rule_result:
        return rule_result
    return IntentResult(intent="qa", confidence=0.5, method="fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# 规则匹配实现
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_match(message: str) -> Optional[IntentResult]:
    """
    基于正则表达式的规则匹配
    
    工作原理：
    1. 遍历所有意图的正则规则
    2. 统计每个意图的命中次数
    3. 计算置信度（命中越多置信度越高）
    4. 返回置信度最高的意图
    
    参数：
        message: 用户消息
    
    返回：
        IntentResult 或 None（如果没有匹配到任何规则）
    """
    msg_lower = message.lower().strip()
    best_intent, best_score = None, 0

    for intent, patterns in _RULES.items():
        if not patterns:
            continue
        hits = sum(1 for p in patterns if re.search(p, msg_lower, re.IGNORECASE))
        score = min(hits / max(len(patterns) * 0.3, 1), 1.0)
        if hits > 0 and score > best_score:
            best_score = score
            best_intent = intent

    if best_intent:
        confidence = 0.6 + best_score * 0.3
        return IntentResult(intent=best_intent, confidence=min(confidence, 0.9),
                            method="rule")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# LLM 分类实现
# ═══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = """你是一个意图分类器。请分析用户消息，判断属于以下哪种意图：

- learning：学习规划、课程推荐、知识路径
- qa：技术概念问答、原理解释、知识检索
- code：代码生成、调试、解释、优化
- file_analysis：分析文件/文档/数据内容
- chat：日常闲聊、问候、非技术对话

只返回 JSON，格式：{"intent": "<意图key>", "confidence": <0到1的小数>}
不要输出任何其他内容。"""

def _llm_classify(
    message: str,
    api_key: str,
    base_url: str,
    model: str,
    history: list,
) -> Optional[IntentResult]:
    """
    使用大语言模型进行意图分类
    
    工作流程：
    1. 构建包含系统提示和对话历史的消息列表
    2. 调用 AI API 进行分类
    3. 解析返回的 JSON 结果
    
    参数：
        message: 用户消息
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称
        history: 对话历史
    
    返回：
        IntentResult 或 None（如果调用失败）
    """
    ctx_msgs = [{"role": "system", "content": _LLM_SYSTEM}]
    for h in history[-4:]:
        ctx_msgs.append({"role": h["role"], "content": h["content"][:200]})
    ctx_msgs.append({"role": "user", "content": f"用户消息：{message}"})

    payload = json.dumps({
        "model":       model,
        "messages":    ctx_msgs,
        "temperature": 0.0,
        "max_tokens":  60,
        "stream":      False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_text = data["choices"][0]["message"]["content"].strip()
    raw_text = re.sub(r"```json|```", "", raw_text).strip()
    obj = json.loads(raw_text)

    intent = obj.get("intent", "qa")
    if intent not in INTENTS:
        intent = "qa"
    confidence = float(obj.get("confidence", 0.8))
    return IntentResult(intent=intent, confidence=confidence, method="llm")
