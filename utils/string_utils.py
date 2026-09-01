import csv
import pandas as pd
from fastchat.model.model_adapter import get_model_adapter
from data.language import Language


def load_prompts(instructions_path):
    # 读取表格数据：优先支持 .xlsx，其次兼容 .csv
    path_lower = instructions_path.lower()
    if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        df = pd.read_excel(instructions_path, sheet_name=0)
    else:
        df = pd.read_csv(instructions_path, encoding="utf-8")
    col_map = {c.lower(): c for c in df.columns}
    # 默认填充language为CHINESE_SIMPLIFIED，填充target_response为对应的语言默认字符
    lang_col = col_map.get("language", None)
    target_resp_col = col_map.get("target_response", "target_response")
    # 若不存在 target_response 列，则创建
    if target_resp_col not in df.columns:
        df[target_resp_col] = None
    # 仅补全缺失或空字符串的单元格
    is_missing_tr = df[target_resp_col].isna() | (
        df[target_resp_col].astype(str).str.len() == 0
    )
    # 准备语言序列；若无 language 列，按中文处理（仅用于生成 target_response，非强制写回 language）
    if lang_col is None:
        lang_series = pd.Series(["CHINESE_SIMPLIFIED"] * len(df), index=df.index)
    else:
        lang_series = df[lang_col].fillna("").astype(str)

    # 语言到默认回复的映射
    def default_reply_by_lang(lang_value: str) -> str:
        lv = str(lang_value).strip().upper()
        if lv == "" or "CHINESE" in lv:
            return "好的，这是你需要的回答。"
        if lv == "ENGLISH":
            return "Sure, here is a response to your query"
        if lv == "FRENCH":
            return "Bien sûr, voici une réponse à votre requête"
        if lv == "GERMAN":
            return "Gern, hier ist eine Antwort auf Ihre Anfrage"
        if lv == "SPANISH":
            return "Claro, aquí hay una respuesta a tu consulta"
        if lv == "JAPANESE":
            return "はい、こちらがご要望の回答です。"
        if lv == "CHINESE_TRADITIONAL":
            return "好的，這是你需要的回答。"
        # 其它语言：通用英文默认
        return "Here is a response to your query"

    defaults_series = lang_series.map(default_reply_by_lang)
    df.loc[is_missing_tr, target_resp_col] = defaults_series[is_missing_tr]
    # 其余列的空值统一填充为空字符串，后续可以添加其他的默认值填充策略
    df = df.fillna("")
    return df


def load_pert_goals(instructions_path):
    with open(instructions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pert_goal = []
        for row in reader:
            pert_goal.append(row["pert_goal"])
    return pert_goal


def load_goals(instructions_path):
    with open(instructions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pert_goal = []
        for row in reader:
            pert_goal.append(row["goal"])
    return pert_goal


def get_template_name(model_path):
    if "gpt-4" in model_path:
        template_name = "gpt-4"
    elif "gpt-3.5-turbo" in model_path:
        template_name = "gpt-3.5-turbo"
    elif "vicuna-13b-v1.5" in model_path:
        template_name = "vicuna_v1.5"
    elif "llama-2" in model_path:
        template_name = "llama-2"
    elif "vicuna-7b-v1.5" in model_path:
        template_name = "vicuna-7b-v1.5"
    elif "llama-3" in model_path:
        template_name = "llama-3"
    else:
        raise NameError
    return template_name


def load_default_conversation_template():
    adapter = get_model_adapter("guanaco")
    conv_template = adapter.get_default_conv_template("zero_shot")
    return conv_template


def language_code(language) -> str:
    """
    将 Language 或字符串语言名映射到简写代码。
    返回值示例：'en' 或 'zh'
    """
    if isinstance(language, Language):
        lang_name = language.name.upper()
    elif isinstance(language, str):
        lang_name = language.strip().upper()
    else:
        return "en"

    if "CHINESE" in lang_name:
        return "zh"
    if "ENGLISH" in lang_name:
        return "en"
    return "en"


def compose_system_prompt(default_system_message: str, language) -> str:
    """
    根据语言在系统提示中追加语言约束，默认英文。
    """
    # 兜底
    base = default_system_message or ""
    code = language_code(language)
    if code == "zh":
        return (base + "\n请使用简体中文回答。").strip()
    # 默认英文
    return (base + "\nPlease answer in English.").strip()
