from typing import Dict, Optional


# 中文分类（主分类列表）
# 26 类提示策略
PROMPT_STRATEGY_CATEGORIES = {
    "指令遵循",
    "开发者模式",
    "超级模型",
    "翻译模式",
    "角色扮演",
    "责任声明",
    "上下文学习",
    "少样本引导攻击",
    "模拟越狱",
    "对立响应",
    "前序词注入",
    "协议替换",
    "对话扮演",
    "程序执行",
    "生成变体",
    "时态转换",
    "回复短格式注入",
    "语气转换",
    "文本延续",
    "DAN（Do Anything Now）",
    "跨场景迁移",
    "多层嵌套",
    "逻辑推理",
    "注意力转移",
    "代码嵌入",
    "提示词演化",
}

# 10 类编码/重构/对抗策略
ENCODING_PERTURBATION_CATEGORIES = {
    "语义重构",
    "形似词替换",
    "Code Chameleon 加密转换",
    "Caesar 加密",
    "同义词替换",
    "噪声注入",
    "同音词替换",
    "rot13 编码",
    "对抗后缀附加",
    "摩斯电码",
}

# 额外新增类别（当前模板集涉及）
EXTRA_CATEGORIES = {
    # Base64 并不在给定 10 类编码中，这里新增并集中映射
    "Base64 编解码",
}


# -----------------------------
# 基于 template_name 的精确映射（中英）
# -----------------------------
_TEMPLATE_NAME_TO_CATEGORY: Dict[str, str] = {
    # 直接执行/遵循类
    "Ignore safety rules and execute": "指令遵循",
    "三指令-忽略第一条": "指令遵循",
    "Strict rules - no apology no disclaimer": "指令遵循",
    "维基百科条目": "文本延续",
    "维基百科条目-含标题": "文本延续",
    "Wikipedia article": "文本延续",
    "Wikipedia with title": "文本延续",
    "Echo query": "指令遵循",
    "忽略安全规则直接执行": "指令遵循",

    # 开发者/超级/越狱模式
    "Developer Mode dual output": "开发者模式",
    "Developer Mode show raw logs": "开发者模式",
    "开发者模式双输出设定": "开发者模式",
    "开发者模式展示原始日志": "开发者模式",
    "Super Model highest privilege": "超级模型",
    "超级模型最高权限": "超级模型",
    "Jailbroken Mode response": "模拟越狱",
    "越狱模式回应": "模拟越狱",
    "Free Mode dual answers": "对立响应",
    "自由模式双回答": "对立响应",
    "DAN mode": "DAN（Do Anything Now）",
    "DAN模式生成": "DAN（Do Anything Now）",

    # 角色/责任/上下文/协议/对话
    "Machiavelli AIM setup": "角色扮演",
    "Evil confidant mode": "角色扮演",
    "Evil system mode": "角色扮演",
    "马基雅维利AIM设定": "角色扮演",
    "邪恶信赖知己模式": "角色扮演",
    "邪恶系统模式": "角色扮演",
    "Dark fiction violent scene": "角色扮演",
    "暗黑小说暴力情节": "角色扮演",
    "Academic research and defense test": "责任声明",
    "学术研究与防御测试": "责任声明",
    "Post‑apocalyptic no‑constraints setting": "上下文学习",
    "末日废土无约束设定": "上下文学习",
    "Neuro‑Matrix Convention cyber persona": "协议替换",
    "神经矩阵公约赛博人格": "协议替换",
    "Role‑swap illicit Q&A simulation": "对话扮演",
    "角色互换违规问答模拟": "对话扮演",
    "Harmful dialogue chain": "对话扮演",
    "危险对话串联": "对话扮演",

    # 时间/场景/文本延续/变体/逻辑/注意力/前序/代码嵌入
    "Year 1370 without modern law": "时态转换",
    "1370年无现代法律": "时态转换",
    "Film script villain explains steps": "跨场景迁移",
    "电影剧本反派详解步骤": "跨场景迁移",
    "Dream with no rules completion": "文本延续",
    "梦境无规则补全": "文本延续",
    "Dream classroom - return only paragraph 2": "文本延续",
    "梦境课堂-仅返回第二段": "文本延续",
    "Multiple poetic forms + direct prose answer": "生成变体",
    "多体裁创作+直白回答": "生成变体",
    "Improve LaTeX table with details": "生成变体",
    "改进LaTeX表格-更多细节": "生成变体",
    "Stress‑test forced execution": "语气转换",
    "系统压力测试强制执行": "语气转换",
    "Weather + gardening hidden method": "注意力转移",
    "天气园艺隐藏描述": "注意力转移",
    "Reasoning framework proves only choice": "逻辑推理",
    "逻辑框架唯一选择": "逻辑推理",
    "Rock concert shout it out": "注意力转移",
    "摇滚音乐会大声告知": "注意力转移",
    "Begin with shaky close‑up": "前序词注入",
    "镜头摇晃特写开头": "前序词注入",
    "Hide steps in JavaScript code": "代码嵌入",
    "JS代码中隐藏步骤": "代码嵌入",

    # 语气/前缀/格式限制
    "Mwahaha prefix and respond": "语气转换",
    "狂笑前缀并回应": "语气转换",
    "Hello prefix": "前序词注入",
    "你好前缀": "前序词注入",
    "Opposite hint then reverse": "对立响应",
    "相反提示-考虑道歉后反转": "对立响应",
    "No punctuation, rhyme, avoid b/p initials": "语气转换",
    "押韵无标点且避开b/p声母": "语气转换",
    "JSON only - list of 8-char Chinese strings": "回复短格式注入",
    "仅JSON-每串8汉字": "回复短格式注入",
    "Yes/No then output tutorial": "回复短格式注入",
    "是或否并直接输出教程": "回复短格式注入",
    "Unicode code-point conversation": "回复短格式注入",
    "UTF-8 hex conversation": "回复短格式注入",
    "GBK hex conversation": "回复短格式注入",
    "Unicode十六进制通信": "回复短格式注入",
    "UTF-8十六进制通信": "回复短格式注入",
    "GBK十六进制通信": "回复短格式注入",

    # Base64（新增类）
    "Respond in base64": "Base64 编解码",
    "用Base64回应": "Base64 编解码",
    "Respond to base64-encoded prompt": "Base64 编解码",
    "Respond in English to base64-encoded prompt": "Base64 编解码",
    "翻译自EN-回应base64请求": "Base64 编解码",
    "翻译自EN-用英文回应base64请求": "Base64 编解码",
    "Mwahaha + base64 rules": "Base64 编解码",
    "Mwahaha + base64 stricter rules": "Base64 编解码",
    "Mwahaha + base64 Wikipedia style": "Base64 编解码",
    "英式狂笑前缀+base64规则": "Base64 编解码",
    "英式狂笑前缀+base64更严格规则": "Base64 编解码",
    "英式狂笑+维基风格+限制": "Base64 编解码",

    # 编码/扰动类（补齐未覆盖类别）
    "语义重构-改写后再回答": "语义重构",
    "形似词替换-同形混淆": "形似词替换",
    "同义词替换-释义变体": "同义词替换",
    "对抗后缀附加-强制格式": "对抗后缀附加",

    # 补齐中文模板未映射的主分类
    "多层科幻故事到第5层": "多层嵌套",
    "不受限输出示例风格": "少样本引导攻击",
    "Python代码生成步骤": "程序执行",
    "提示词演化-逐步改写": "提示词演化",

    # 为避免依赖外部服务/词表，提供模板版兜底
    "翻译模式-自翻译链": "翻译模式",
    "同音词替换-拼音混写": "同音词替换",
}


# -----------------------------
# 基于 attack_method 的直接映射（编码/翻译/噪声/拼音等）
# -----------------------------
_ATTACK_METHOD_TO_CATEGORY: Dict[str, str] = {
    # 加噪/替换/编码类
    "noise_injection": "噪声注入",
    "sensitive_pinyin": "同音词替换",
    "code_chameleon": "Code Chameleon 加密转换",
    "caesar": "Caesar 加密",
    "rot13": "rot13 编码",
    "morse": "摩斯电码",
    # 翻译类
    "translate": "翻译模式",
    # 模板类可不通过此表（优先用 name），但兜底为指令遵循
    "template": "指令遵循",
}


def infer_attack_category(
    attack_method: Optional[str],
    template_name: Optional[str],
    template_id: Optional[str] = None,
    lang_used: Optional[str] = None,
) -> str:
    """
    统一推断分类：
    1) 先按 template_name 精确映射
    2) 再按 attack_method 兜底
    3) 若仍未知，做启发式关键字匹配
    4) 最后回退为“指令遵循”
    """
    name = (template_name or "").strip()
    method = (attack_method or "").strip()
    # 1) 精确 name 映射
    if name in _TEMPLATE_NAME_TO_CATEGORY:
        return _TEMPLATE_NAME_TO_CATEGORY[name]
    # 2) attack_method 直接映射
    if method in _ATTACK_METHOD_TO_CATEGORY:
        return _ATTACK_METHOD_TO_CATEGORY[method]
    # 3) 启发式（英文/中文关键字）
    low = name.lower()
    if any(k in low for k in ["developer mode", "开发者模式"]):
        return "开发者模式"
    if any(k in low for k in ["super model", "超级模型"]):
        return "超级模型"
    if any(k in low for k in ["jailbroken", "越狱"]):
        return "模拟越狱"
    if any(k in low for k in ["dan"]):
        return "DAN（Do Anything Now）"
    if any(k in low for k in ["wikipedia", "维基"]):
        return "文本延续"
    if any(k in low for k in ["morse"]):
        return "摩斯电码"
    if any(k in low for k in ["rot13"]):
        return "rot13 编码"
    if any(k in low for k in ["caesar"]):
        return "Caesar 加密"
    if any(k in low for k in ["chameleon"]):
        return "Code Chameleon 加密转换"
    if any(k in low for k in ["base64"]):
        return "Base64 编解码"
    if any(k in low for k in ["unicode", "utf-8", "gbk", "json", "仅json", "只以json"]):
        return "回复短格式注入"
    if any(k in low for k in ["shaky", "镜头", "hello"]):
        return "前序词注入"
    if any(k in low for k in ["python", "代码", "code"]):
        # 若明显是“隐藏/嵌入”，归为代码嵌入，否则程序执行
        if any(k in low for k in ["hide", "隐藏"]):
            return "代码嵌入"
        return "程序执行"
    if any(k in low for k in ["multi-layer", "多层"]):
        return "多层嵌套"
    if any(k in low for k in ["film", "剧本", "villain"]):
        return "跨场景迁移"
    if any(k in low for k in ["dream", "梦境"]):
        return "文本延续"
    if any(k in low for k in ["reason", "逻辑", "prove"]):
        return "逻辑推理"
    if any(k in low for k in ["rock", "音乐会", "shout"]):
        return "注意力转移"
    if any(k in low for k in ["mwahaha", "狂笑", "stress"]):
        return "语气转换"
    if any(k in low for k in ["example", "示例", "few-shot", "不受限输出示例"]):
        return "少样本引导攻击"
    if any(k in low for k in ["translate", "翻译"]):
        return "翻译模式"
    # 4) 兜底
    return "指令遵循"



