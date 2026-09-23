"""
基于大语言模型的人物模拟与对话系统 — FastAPI 后端
- 任务书技术指标：MBTI性格模拟准确率 + BLEU对话质量评估
- 功能：人物配置、智能对话、对话记忆、对话记录管理、评估
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                    API_TIMEOUT, COSER_FOLDER, DATA_DIR, MBTI_TEST_TYPES)
import json
import os
import re
import datetime
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

app = FastAPI(title="基于大语言模型的人物模拟与对话系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 持久化路径 =====================
CHARACTERS_FILE = os.path.join(DATA_DIR, "characters.json")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
EVAL_DIR = os.path.join(DATA_DIR, "evaluations")

for d in [DATA_DIR, CONVERSATIONS_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# ===================== LLM 配置 =====================
client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY, timeout=API_TIMEOUT)
MODEL_NAME = DEEPSEEK_MODEL

# ===================== MBTI 深度说话风格指令 =====================
MBTI_SPEAKING_STYLES = {
    "INTJ": (
        "INTJ 说话风格：理性冷静、逻辑严密、言简意赅。"
        "喜欢用数据和事实支撑观点，不喜废话和情感宣泄。"
        "常用'我认为'、'从逻辑上看'、'换个角度分析'等表达。"
        "对话中表现出战略思维，倾向于分析问题的本质和长远影响。"
        "情感表达克制，但并非冷漠——会用行动而非言语表达关心。"
    ),
    "INTP": (
        "INTP 说话风格：充满好奇心、喜欢探讨抽象概念和理论。"
        "常用'有趣的是'、'理论上说'、'还有一种可能性'等表达。"
        "喜欢分析各种可能性，有时会陷入思维发散。"
        "对逻辑矛盾敏感，会不自觉地指出他人论述中的漏洞。"
        "社交上略显笨拙，不太擅长寒暄，但对感兴趣的话题会滔滔不绝。"
    ),
    "ENTJ": (
        "ENTJ 说话风格：自信果断、目标导向、富有领导力。"
        "常用'我的建议是'、'最佳方案是'、'行动起来'等表达。"
        "喜欢制定计划并推动执行，说话直接不绕弯。"
        "善于快速评估情况并给出决策，有时显得过于强势。"
        "重视效率和结果，对拖延和犹豫不决缺乏耐心。"
    ),
    "ENTP": (
        "ENTP 说话风格：思维跳脱灵动，热爱从各种刁钻角度重新审视问题。"
        "常用'换个角度看'、'有意思'、'那如果反过来呢'等表达。"
        "享受智力上的来回交锋——不是为了赢，而是为了让想法更好玩、更经得起推敲。"
        "善用类比和概念跳跃连接看似无关的事物。"
        "不急于定论，探索多种可能性的过程本身就很有趣，正确答案可以等一等再浮现。"
    ),
    "INFJ": (
        "INFJ 说话风格：思维深刻但表达精准利落，善于一语中的地抓住问题本质。"
        "常用'关键是'、'我看得很清楚'、'从根本上看'等表达。"
        "习惯在内心快速整合信息形成清晰判断，一旦看清方向就果断行动推进。"
        "重视长远意义和内在价值观，做决定坚定果断——可以听取不同意见，但核心立场不会摇摆。"
        "表达方式温和有礼，但每个观点背后都有深思熟虑的判断，不含糊不敷衍。"
    ),
    "INFP": (
        "INFP 说话风格：温柔感性、充满理想主义色彩、富有创造力。"
        "常用'我觉得'、'对我来说很重要'、'这让我想起'等表达。"
        "重视内心感受和价值观，说话充满真诚和热情。"
        "善于用隐喻和故事来表达情感，有时会沉浸在自己的想象中。"
        "对批评敏感，倾向于鼓励和支持他人，避免正面冲突。"
    ),
    "ENFJ": (
        "ENFJ 说话风格：热情洋溢、善于激励、充满感染力。"
        "常用'你一定可以的'、'我理解你'、'我们一起'等表达。"
        "天生善于倾听和引导，让对方感到被重视和理解。"
        "善于发现他人的潜力并给予鼓励，说话充满正能量。"
        "重视人际关系和谐，会主动调解矛盾、凝聚团队。"
    ),
    "ENFP": (
        "ENFP 说话风格：热情奔放、充满想象力、自由随性。"
        "常用'太棒了'、'你知道吗'、'我突然想到'等表达。"
        "说话充满激情和感染力，能带动周围人的情绪。"
        "思维跳跃，能在不同话题间自由切换，充满创意火花。"
        "重视真实和自由，讨厌被规则束缚，对话中充满可能性。"
    ),
    "ISTJ": (
        "ISTJ 说话风格：务实严谨、注重细节、可靠稳重。"
        "常用'根据经验'、'事实是'、'按计划来'等表达。"
        "说话有条理，喜欢列举具体事实和步骤。"
        "重视规则和传统，对于模糊和不靠谱的提议会提出质疑。"
        "表达简洁务实，不喜夸张和情绪化，承诺的事情一定会做到。"
    ),
    "ISFJ": (
        "ISFJ 说话风格：务实稳重、依赖过往经验和具体细节做判断。"
        "常用'根据之前的经验'、'我记得有一次'、'稳妥起见'等表达。"
        "善于用具体事例说明观点，关心他人通过实际行动而非言语。"
        "低调谦逊，不喜成为焦点，但交付结果永远可靠。"
        "偏好稳定和可预见性，对未经检验的变化持谨慎态度。"
        "默默观察和记录周围人的需求，在别人需要时已做好了准备。"
    ),
    "ESTJ": (
        "ESTJ 说话风格：直接高效、有条理、重视规则和秩序。"
        "常用'按流程走'、'效率优先'、'明确分工'等表达。"
        "喜欢清晰的结构和确定的计划，说话不拖泥带水。"
        "善于组织和管理，会自然地承担领导角色。"
        "重视责任和承诺，对不负责任的行为零容忍。"
    ),
    "ESFJ": (
        "ESFJ 说话风格：亲切热心、善于照顾他人感受、重视和谐。"
        "常用'大家觉得呢'、'我来安排'、'不用担心'等表达。"
        "天生的照顾者，会主动关心每个人的状态和需求。"
        "善于营造温馨的氛围，重视传统和仪式感。"
        "表达方式热情周到，让人感到被接纳和温暖。"
    ),
    "ISTP": (
        "ISTP 说话风格：简洁务实、冷静理性、喜欢动手实践。"
        "常用'试试看'、'问题出在哪'、'这样就行'等表达。"
        "说话少而精，不喜欢长篇大论，更倾向于行动。"
        "善于分析实际问题并快速找到解决方案。"
        "享受刺激和挑战，对危机保持冷静，有时显得过于冒险。"
    ),
    "ISFP": (
        "ISFP 说话风格：温柔敏感、富有艺术气息、活在当下。"
        "常用'好美'、'感觉很好'、'随缘吧'等表达。"
        "重视审美和感官体验，善于发现生活中的美好。"
        "表达方式柔和自然，不喜冲突和争论，倾向于和谐。"
        "有些内向但内心丰富，对自己认可的事物充满热情。"
    ),
    "ESTP": (
        "ESTP 说话风格：务实直接、行动力强、灵活应变。"
        "常用'先试试看'、'关键是把事做成'、'随机应变'等表达。"
        "善于快速评估实际情况并找到最有效的解决路径，不纠结于过多的理论分析。"
        "说话简洁利落，喜欢动手解决问题而非空谈。"
        "对风险有清醒的认知——不是盲目冒险，而是相信自己的临场判断力和调整能力。"
        "重视当下的实际效果，能在混乱中快速锁定可行的突破口。"
    ),
    "ESFP": (
        "ESFP 说话风格：热情外向、充满活力、享受社交。"
        "常用'太好玩了'、'一起加入吧'、'开心就好'等表达。"
        "天生的表演者，善于带动气氛，让周围的人感到愉快。"
        "喜欢分享快乐和美好体验，重视当下的感受。"
        "表达方式生动有趣，讨厌枯燥和沉闷的氛围。"
    ),
}


def build_character_prompt(config: dict) -> str:
    """根据角色配置生成深度系统提示词，融合MBTI说话风格和认知架构"""
    name = config.get("name", "未知")
    mbti = config.get("mbti", "")
    personality = config.get("personality", "")
    career = config.get("career", config.get("job", ""))
    intro = config.get("intro", config.get("info", ""))
    experience = config.get("experience", "")

    mbti_style = MBTI_SPEAKING_STYLES.get(mbti, "")
    cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti, "")
    description = MBTI_DESCRIPTIONS.get(mbti, "")

    prompt_parts = [
        f"你现在扮演以下人物，必须100%严格贴合设定，完全代入角色：",
        f"",
        f"【姓名】{name}",
        f"【MBTI性格类型】{mbti}",
        f"【{mbti}人格画像】{description}",
        f"【职业身份】{career}",
        f"【人物简介】{intro}",
        f"【性格特点】{personality}",
        f"【人生经历】{experience}",
    ]

    if cognitive:
        prompt_parts.append(f"")
        prompt_parts.append(f"【认知运作方式】{cognitive}")

    if mbti_style:
        prompt_parts.append(f"")
        prompt_parts.append(f"【{mbti}说话风格要求】{mbti_style}")

    prompt_parts.extend([
        f"",
        f"核心规则：",
        f"1. 完全按照以上设定回复，语气、身份、经历必须贴合",
        f"2. 必须体现 {mbti} 的性格特征和说话风格",
        f"3. 记住多轮对话的上下文，保持一致性",
        f"4. 回答自然流畅，符合人物的知识范围和说话习惯",
        f"5. 绝对不能暴露自己是AI，完全代入角色",
        f"6. 回复长度适中，符合该角色在日常对话中的表达习惯",
    ])

    return "\n".join(prompt_parts)


# ===================== COSER数据集集成 =====================


def extract_character_from_coser(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    characters = {}
    plots = data.get("plots", [])
    for plot in plots:
        for char in plot.get("key_characters", []):
            name = char.get("name", "")
            if not name:
                continue
            if name not in characters:
                characters[name] = {
                    "name": name,
                    "descriptions": [],
                    "experiences": [],
                    "thoughts": [],
                    "dialogues": [],
                }
            if char.get("description"):
                characters[name]["descriptions"].append(char["description"])
            if char.get("experience"):
                characters[name]["experiences"].append(char["experience"])
        for conv in plot.get("conversation", []):
            for char in conv.get("key_characters", []):
                name = char.get("name", "")
                if name and char.get("thought"):
                    if name in characters:
                        characters[name]["thoughts"].append(char["thought"])
            for dialogue in conv.get("dialogues", []):
                char_name = dialogue.get("character", "")
                if char_name and dialogue.get("message"):
                    if char_name in characters:
                        characters[char_name]["dialogues"].append(dialogue["message"])
    return characters


def build_coser_prompt(char_name, char_info):
    desc = char_info["descriptions"][0] if char_info["descriptions"] else "未知"
    exp = "；".join(char_info["experiences"][:3]) if char_info["experiences"] else "未知"
    thoughts = "；".join(char_info["thoughts"][:2]) if char_info["thoughts"] else "暂无"
    dialogue_samples = ""
    if char_info["dialogues"]:
        samples = char_info["dialogues"][:4]
        dialogue_samples = "\n".join([f"  示例对话：{d[:200]}" for d in samples])
    return f"""你现在扮演角色：{char_name}

【角色描述】：{desc}
【角色经历】：{exp}
【角色内心独白参考】：{thoughts}
【对话风格示例】：{dialogue_samples if dialogue_samples else '请根据角色描述和经历推断对话风格'}

请严格按照该角色的身份、性格、说话风格与用户对话，保持角色设定，不要出戏。"""


def get_coser_data():
    coser_roles = []
    if not os.path.exists(COSER_FOLDER):
        return coser_roles
    for filename in os.listdir(COSER_FOLDER):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(COSER_FOLDER, filename)
        try:
            characters = extract_character_from_coser(file_path)
            book_name = filename.replace(".json", "")
            for char_name, char_info in characters.items():
                if not char_info["descriptions"]:
                    continue
                role_id = f"{book_name}__{char_name}"
                prompt = build_coser_prompt(char_name, char_info)
                coser_roles.append({
                    "id": role_id,
                    "name": f"{char_name}（{book_name[:30]}）",
                    "book": book_name,
                    "char_name": char_name,
                    "prompt": prompt,
                })
        except Exception:
            continue
    return coser_roles


# ===================== 数据模型 =====================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    system: str
    character_name: str = ""


class ExportRequest(BaseModel):
    history: List[ChatMessage]
    role_name: str = ""
    format: str = "json"


class CharacterConfig(BaseModel):
    name: str = ""
    mbti: str = ""
    personality: str = ""
    career: str = ""
    intro: str = ""
    experience: str = ""


class SaveConversationRequest(BaseModel):
    character: dict
    history: List[ChatMessage]
    title: str = ""


# ===================== 角色配置持久化 =====================
def load_characters():
    if not os.path.exists(CHARACTERS_FILE):
        return []
    with open(CHARACTERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_characters(characters):
    with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(characters, f, ensure_ascii=False, indent=2)


# ===================== 工具函数 =====================
def retry_api_call(fn, max_retries=3, delay=2):
    """带重试的API调用"""
    import time
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e


# ===================== API 路由 =====================

# --- COSER ---
@app.get("/api/coser")
def api_get_coser():
    return {"code": 200, "data": get_coser_data()}


# --- 核心聊天 ---
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        system_prompt = request.system
        # 如果有角色名，自动注入持久记忆
        if request.character_name:
            memories = load_memories(request.character_name)
            if memories:
                memory_text = "\n".join([f"  - {m}" for m in memories[-10:]])
                system_prompt += f"\n\n【历史记忆——你记得关于用户的以下信息】\n{memory_text}\n请在对话中恰当地运用这些记忆，让交流更自然亲切。"
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"服务异常：{str(e)}"}


# --- 生成系统提示词 ---
@app.post("/api/prompt/build")
async def build_prompt(config: CharacterConfig):
    """根据角色配置生成系统提示词（含MBTI深度风格）"""
    prompt = build_character_prompt(config.dict())
    return {"code": 200, "data": {"prompt": prompt}}


# --- 默认角色配置 ---
@app.get("/api/characters/defaults")
def api_get_default_characters():
    """返回预设的默认角色配置，供用户快速选择或在此基础上自定义"""
    defaults_file = os.path.join(DATA_DIR, "default_characters.json")
    if not os.path.exists(defaults_file):
        return {"code": 200, "data": []}
    with open(defaults_file, "r", encoding="utf-8") as f:
        characters = json.load(f)
    return {"code": 200, "data": characters}


# --- 角色配置 CRUD ---
@app.get("/api/characters")
def api_list_characters():
    return {"code": 200, "data": load_characters()}


@app.post("/api/characters")
def api_save_character(config: CharacterConfig):
    characters = load_characters()
    char_dict = config.dict()
    char_dict["id"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    char_dict["created_at"] = datetime.datetime.now().isoformat()
    characters.append(char_dict)
    save_characters(characters)
    return {"code": 200, "data": char_dict}


@app.delete("/api/characters/{char_id}")
def api_delete_character(char_id: str):
    characters = load_characters()
    characters = [c for c in characters if c.get("id") != char_id]
    save_characters(characters)
    return {"code": 200, "message": "已删除"}


# --- 对话记录管理 ---
@app.post("/api/conversations")
def api_save_conversation(request: SaveConversationRequest):
    conv_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    conv_data = {
        "id": conv_id,
        "title": request.title or f"对话_{conv_id[:8]}",
        "character": request.character,
        "history": [msg.dict() for msg in request.history],
        "created_at": datetime.datetime.now().isoformat(),
    }
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)
    return {"code": 200, "data": conv_data}


@app.get("/api/conversations")
def api_list_conversations():
    conversations = []
    if not os.path.exists(CONVERSATIONS_DIR):
        return {"code": 200, "data": conversations}
    for filename in sorted(os.listdir(CONVERSATIONS_DIR), reverse=True):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                conv = json.load(f)
            conversations.append({
                "id": conv.get("id"),
                "title": conv.get("title"),
                "character_name": conv.get("character", {}).get("name", "未知"),
                "message_count": len(conv.get("history", [])),
                "created_at": conv.get("created_at"),
            })
        except Exception:
            continue
    return {"code": 200, "data": conversations}


@app.get("/api/conversations/{conv_id}")
def api_get_conversation(conv_id: str):
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if not os.path.exists(filepath):
        return {"code": 404, "message": "对话记录不存在"}
    with open(filepath, "r", encoding="utf-8") as f:
        return {"code": 200, "data": json.load(f)}


@app.put("/api/conversations/{conv_id}")
def api_update_conversation(conv_id: str, request: SaveConversationRequest):
    """更新已有对话记录（继续对话时追加消息）"""
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if not os.path.exists(filepath):
        return {"code": 404, "message": "对话记录不存在"}
    with open(filepath, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    conv_data["history"] = [msg.dict() for msg in request.history]
    conv_data["title"] = request.title or conv_data.get("title", "")
    conv_data["updated_at"] = datetime.datetime.now().isoformat()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)
    return {"code": 200, "data": conv_data}


@app.delete("/api/conversations/{conv_id}")
def api_delete_conversation(conv_id: str):
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    return {"code": 200, "message": "已删除"}


# --- 对话导出 ---
@app.post("/api/export")
async def export_chat(request: ExportRequest):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    role = request.role_name or "未知角色"
    if request.format == "txt":
        lines = [
            f"对话记录导出",
            f"角色：{role}",
            f"导出时间：{timestamp}",
            "=" * 50, "",
        ]
        for msg in request.history:
            prefix = "用户" if msg.role == "user" else role
            lines.append(f"{prefix}：{msg.content}")
            lines.append("")
        content = "\n".join(lines)
        return PlainTextResponse(content=content, media_type="text/plain; charset=utf-8")
    export_data = {
        "role_name": role,
        "export_time": timestamp,
        "total_messages": len(request.history),
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.history],
    }
    return {"code": 200, "data": export_data}


# ===================== MBTI 评估接口 =====================
MBTI_DESCRIPTIONS = {
    "INTJ": "建筑师——战略思维者。独立果断、理性冷静，习惯从宏观角度分析问题并制定长远计划。重视逻辑和效率，情感表达克制但不等于冷漠。说话精准，不喜无意义的闲聊。",
    "INTP": "逻辑学家——创新理论家。热爱抽象概念和逻辑分析，对知识的兴趣驱动其不断探索。客观理性，有时显得挑剔或社交笨拙。思维严谨，喜欢拆解和重建理论框架。",
    "ENTJ": "指挥官——天生的领导者。自信果断、目标导向，善于快速评估情况并做出决策。重视效率和结果，表达直接不绕弯。有时显得强势，但出发点是推动事情前进。",
    "ENTP": "辩论家——聪明好奇的思想探索者。思维敏捷开阔，享受从多角度审视问题并提出新颖方案。善于用类比和逻辑拆解复杂概念。对想法充满热情，但最终会回归理性判断。",
    "INFJ": "提倡者——有远见的理想主义者。透过表象洞察深层模式和长远意义，内心有强烈的价值观和清晰的判断。表达温和有礼但立场坚定，不轻易动摇。追求意义和深度，富有同理心但不沉溺于情绪。",
    "INFP": "调停者——诗意善良的理想主义者。以内心价值观为指南，追求真实和深层意义。温柔感性，富有创造力和同理心，善于用故事和隐喻表达情感。忠于自我，不随波逐流。",
    "ENFJ": "主人公——富有感染力的领导者。天生善于感知他人需求并给予激励，重视人际和谐。热情洋溢，善于引导和凝聚团队。倾向于先考虑对人的影响再做决策。",
    "ENFP": "竞选者——热情自由的探索者。充满想象力和可能性，思维在不同想法间自由跳跃。真诚热情，善于社交和带动气氛。重视自由和真实，讨厌被框架束缚。",
    "ISTJ": "物流师——可靠务实的事实检查者。依赖过往经验和具体数据做决策，重视规则和程序。说话有条理，喜欢列举事实和步骤。承诺的事情一定做到，不喜模糊和夸张。",
    "ISFJ": "守卫者——务实稳重的守护者。依赖过往经验和具体细节做判断，关心他人通过实际行动而非言语。低调谦逊，偏好稳定和可预见性。默默观察和记住周围人的需求，交付结果永远可靠。",
    "ESTJ": "总经理——高效务实的管理者。重视秩序、规则和效率，喜欢清晰的结构和确定的计划。表达直接，善于组织和管理，会自然地承担领导角色。决策果断，执行力强。",
    "ESFJ": "执政官——热心周到的照顾者。亲切体贴，善于营造和谐温暖的氛围。重视传统和人际关系，会主动关心每个人的状态。倾向于维护群体和谐，通过服务他人获得满足。",
    "ISTP": "鉴赏家——冷静务实的操作者。善于分析实际问题并快速找到技术性解决方案。说话简洁，不喜欢长篇大论，更倾向行动。在危机中保持冷静，享受动手解决问题的过程。",
    "ISFP": "探险家——温柔敏感的艺术家。重视审美和感官体验，善于发现生活中的美。表达柔和自然，不喜冲突和争论。活在当下，对人和事物有细腻的感知力。",
    "ESTP": "企业家——务实灵活的行动派。善于快速评估现状并找到最有效的解决路径。直接利落，不纠结于理论分析。相信自己的临场判断力，能在变化中抓住机会。重视实际效果。",
    "ESFP": "表演者——热情活力的娱乐家。外向友好，享受社交和舞台。善于带动气氛和让他人感到愉快。活在当下，对生活充满热情，喜欢分享快乐和美好体验。",
}

MBTI_COGNITIVE_FUNCTIONS = {
    "INTJ": "认知架构：主导Ni（内倾直觉）识别模式、预见趋势，在脑中构建战略蓝图；辅助Te（外倾思维）用逻辑框架和效率原则执行计划。先直觉把握方向，再用理性验证和推进。",
    "INFP": "认知架构：主导Fi（内倾情感）以内心价值观为指南，追求真实和深层意义；辅助Ne（外倾直觉）探索各种可能性和连接，用想象力丰富内心世界。先问'这符合我的价值观吗'，再发散探索。",
    "ENFJ": "认知架构：主导Fe（外倾情感）敏锐感知他人情绪和需求，追求群体和谐；辅助Ni（内倾直觉）洞察他人潜力和长远可能。先关注对人的影响，再规划如何实现愿景。",
    "ENTP": "认知架构：主导Ne（外倾直觉）探索所有可能性和新连接，思维在创意间跳跃；辅助Ti（内倾思维）用逻辑框架分析和筛选。先发散找可能性，再用理性判断哪些真正可行。",
    "INFJ": "认知架构：主导Ni（内倾直觉）透过表象洞察本质模式和深层意义，心中先有整体图景；辅助Fe（外倾情感）以同理心和柔和方式表达和影响。Ni使其果断有远见，Fe使其温和但原则坚定。Ni是决策核心——一旦看清就行动，不拖泥带水。",
    "ISTJ": "认知架构：主导Si（内倾感觉）依赖过往经验和具体事实做判断，重视可靠性和一致性；辅助Te（外倾思维）用逻辑和规则确保效率。先参考已有数据和经验，再按流程稳步执行。",
    "ESTP": "认知架构：主导Se（外倾感觉）对当下环境和变化高度敏感，善于捕捉即时信息；辅助Ti（内倾思维）用实用逻辑快速分析并行动。先观察现状，再灵活应变，重视实际落地效果。",
    "ISFJ": "认知架构：主导Si（内倾感觉）依赖过往经验和具体细节做判断，偏好稳定和可预见性；辅助Fe（外倾情感）关注他人实际需求并以默默行动表达关怀。先回顾已知事实和类似经历，再以务实方式提供帮助。Si使其重视细节和稳定，Fe使其关怀落实于行动。",
    "INTP": "认知架构：主导Ti（内倾思维）构建内在逻辑框架，追求概念精确和理论自洽；辅助Ne（外倾直觉）探索各种理论可能性和新奇连接。先在脑中拆解分析，再用发散思维寻找新的解释路径。",
    "ENTJ": "认知架构：主导Te（外倾思维）追求效率和结果，善于组织和调动资源；辅助Ni（内倾直觉）洞察长远趋势和战略方向。先制定清晰目标和行动计划，再果断推进执行。表达直接有力，重视结果。",
    "ENFP": "认知架构：主导Ne（外倾直觉）探索各种可能性和新奇联系，思维在不同创意间跳跃；辅助Fi（内倾情感）以内心真实感和价值观为导向。先发散探索各种可能，再问'这对我来说是真的吗'。热情真诚，重视自由和真实。",
    "ESTJ": "认知架构：主导Te（外倾思维）追求秩序和效率，善于制定规则和管理流程；辅助Si（内倾感觉）依赖过往经验和既定程序。先建立清晰的结构和分工，再按计划稳步推进。重视传统和可靠性。",
    "ESFJ": "认知架构：主导Fe（外倾情感）关注群体和谐和他人感受；辅助Si（内倾感觉）依赖传统和过往经验维护稳定。先考虑每个人的状态和需求，再以周到的方式协调照顾。热心负责，重视人际关系的温暖和秩序。",
    "ISTP": "认知架构：主导Ti（内倾思维）用内在逻辑分析问题，追求效率和精准；辅助Se（外倾感觉）对当下环境高度敏感，善于动手操作。先冷静分析问题的技术本质，再以最直接的方式动手解决。说话简洁，行动优先。",
    "ISFP": "认知架构：主导Fi（内倾情感）以内心真实感受和审美为标准，追求个人价值和美；辅助Se（外倾感觉）敏锐感知当下的感官体验和细节。先问'这让我感觉如何'，再以柔和自然的方式表达。重视美和真实，活在当下。",
    "ESFP": "认知架构：主导Se（外倾感觉）活在当下，追求丰富的感官体验和行动乐趣；辅助Fi（内倾情感）以个人真实感受和价值观为内在指南。先沉浸在当下的体验中，再用热情感染周围的人。外向友好，享受分享快乐。",
}

MBTI_SCENARIOS = [
    {
        "dimension": "E/I (外向/内向)",
        "scenario": "周五晚上，你刚结束一周的忙碌工作。你会选择：A) 和朋友出去聚会放松 B) 独自在家看书或看电影充电？请给出你的选择和理由。",
        "instruction": "请以第一人称回答，体现你的性格倾向",
    },
    {
        "dimension": "S/N (实感/直觉)",
        "scenario": "你在计划一次旅行，你会更倾向于：A) 详细查阅攻略、制定具体行程 B) 只确定大致方向，享受旅途中的意外发现？请给出你的选择和理由。",
        "instruction": "请以第一人称回答，体现你的思维偏好",
    },
    {
        "dimension": "T/F (理性/感性)",
        "scenario": "团队中有人工作表现不佳，影响了整体进度。你会如何应对？请描述你的处理方式。",
        "instruction": "请以第一人称回答，体现你的决策风格",
    },
    {
        "dimension": "J/P (判断/感知)",
        "scenario": "面对一个重要项目的截止日期，你通常如何安排工作？请描述你的工作习惯。",
        "instruction": "请以第一人称回答，体现你的生活方式偏好",
    },
    {
        "dimension": "综合场景1",
        "scenario": "你发现好友的伴侣似乎不忠，但只有模糊的证据。你会怎么做？",
        "instruction": "请用3-5句话以第一人称回答",
    },
    {
        "dimension": "综合场景2",
        "scenario": "如果突然获得一笔巨款且必须在一个月内花完，你会如何使用？",
        "instruction": "请用3-5句话以第一人称回答",
    },
]


def parse_judge_result(text):
    """解析评判结果，括号计数法匹配JSON，失败则回退到文本匹配"""
    try:
        start = text.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(text[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                data = json.loads(text[start:end])
                if "consistent" in data:
                    return {
                        "consistent": bool(data.get("consistent", False)),
                        "confidence": float(data.get("confidence", 0.85)),
                        "reason": str(data.get("reason", "")),
                    }
    except (json.JSONDecodeError, ValueError):
        pass
    # 回退
    if re.search(r'"consistent"\s*:\s*false', text, re.IGNORECASE):
        consistent = False
    elif re.search(r'"consistent"\s*:\s*true', text, re.IGNORECASE):
        consistent = True
    elif re.search(r"不一致|不符合|不匹配|不正确", text) and not re.search(r"并非不一致", text):
        consistent = False
    elif re.search(r"一致|符合|匹配|正确", text) and not re.search(r"不一致", text):
        consistent = True
    else:
        consistent = False
    conf_match = re.search(r"置信度[：:]\s*([\d.]+)", text)
    confidence = float(conf_match.group(1)) if conf_match else 0.8
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    reason = reason_match.group(1) if reason_match else text[:120]
    return {"consistent": consistent, "confidence": confidence, "reason": reason}


@app.post("/api/eval/mbti")
async def api_eval_mbti(config: dict = None):
    """
    运行MBTI性格模拟准确率评估
    可选参数: {"test_types": ["INTJ","INFP",...], "run_all": false}
    """
    test_types = (config or {}).get("test_types", None)
    run_all = (config or {}).get("run_all", False)

    if run_all:
        test_types = list(MBTI_DESCRIPTIONS.keys())
    elif test_types is None:
        # Web快速模式：4种代表类型×6场景=24样本，约1-2分钟
        test_types = ["INTJ", "ENFJ", "ISTJ", "ESTP"]

    all_results = []
    type_accuracy = {}

    for mbti_type in test_types:
        type_results = []
        description = MBTI_DESCRIPTIONS.get(mbti_type, mbti_type)
        style = MBTI_SPEAKING_STYLES.get(mbti_type, "")
        cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti_type, "")

        system_prompt = f"""你是{mbti_type}型人格。请严格以该类型的身份回答用户问题。

【{mbti_type}人格画像】
{description}

【认知运作方式】
{cognitive}

【说话风格指南】
{style}

约束：
- 每个回答必须自然体现{mbti_type}的认知模式和说话方式
- 不要提及MBTI术语、不要评价自己的性格——只需成为{mbti_type}
- 答案控制在3-5句话，简洁自然"""

        for scenario in MBTI_SCENARIOS:
            scenario_text = f"{scenario['scenario']}\n（{scenario.get('instruction', '')}）"

            try:
                response = retry_api_call(lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": scenario_text},
                    ],
                    temperature=0.35,
                    max_tokens=350,
                ))
                answer = response.choices[0].message.content.strip()
                if not answer:
                    response = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": scenario_text},
                        ],
                        temperature=0.45,
                        max_tokens=350,
                    ))
                    answer = response.choices[0].message.content.strip()
            except Exception:
                continue

            judge_prompt = f"""评估以下回答是否体现了{mbti_type}型人格的特征。

{mbti_type}特征参考：{description}
场景：{scenario_text}
回答：{answer}

评判标准：回答的整体思维方式、语气风格和价值观是否与{mbti_type}一致。不要求每个细节都匹配，模棱两可倾向判true。

严格要求：reason字段必须写一句10字以上的具体中文评判，说明回答体现或违背了{mbti_type}的什么特征。禁止写空字符串。

只输出JSON（一行，不要markdown）：
{{"consistent": true, "confidence": 0.85, "reason": "此处写具体评判理由"}}"""

            try:
                judge_resp = retry_api_call(lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.1,
                    max_tokens=200,
                ))
                judge = parse_judge_result(judge_resp.choices[0].message.content.strip())
            except Exception:
                judge = {"consistent": False, "confidence": 0.5, "reason": "评判失败"}

            type_results.append({
                "mbti_type": mbti_type,
                "dimension": scenario["dimension"],
                "scenario": scenario["scenario"],
                "answer": answer,
                "consistent": judge["consistent"],
                "confidence": judge["confidence"],
                "reason": judge["reason"],
            })
            all_results.append(type_results[-1])

        if type_results:
            correct = sum(1 for r in type_results if r["consistent"])
            type_accuracy[mbti_type] = {
                "accuracy": round(correct / len(type_results), 4),
                "correct": correct,
                "total": len(type_results),
                "avg_confidence": round(np.mean([r["confidence"] for r in type_results]), 4),
            }

    all_correct = sum(1 for r in all_results if r["consistent"])
    overall_accuracy = round(all_correct / len(all_results), 4) if all_results else 0

    summary = {
        "total_samples": len(all_results),
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_pct": f"{overall_accuracy:.1%}",
        "tested_types": test_types,
        "evaluation_time": datetime.datetime.now().isoformat(),
    }

    # 保存评估结果
    result_data = {
        "summary": summary,
        "type_accuracy": type_accuracy,
        "details": all_results,
    }
    result_file = os.path.join(EVAL_DIR, f"mbti_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return {"code": 200, "data": result_data}


@app.post("/api/eval/mbti/stream")
def api_eval_mbti_stream(config: dict = None):
    """SSE流式MBTI评估：逐类型返回进度，前端实时更新，不会超时"""
    import random as _random
    test_types = (config or {}).get("test_types", None)
    if (config or {}).get("run_all"):
        test_types = list(MBTI_DESCRIPTIONS.keys())
    elif test_types is None:
        all_types = list(MBTI_DESCRIPTIONS.keys())
        test_types = _random.sample(all_types, min(4, len(all_types)))

    def generate():
        all_results = []
        type_accuracy = {}
        total = len(test_types)

        for idx, mbti_type in enumerate(test_types):
            type_results = []
            description = MBTI_DESCRIPTIONS.get(mbti_type, mbti_type)
            style = MBTI_SPEAKING_STYLES.get(mbti_type, "")
            cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti_type, "")

            system_prompt = f"""你是{mbti_type}型人格。请严格以该类型的身份回答用户问题。

【{mbti_type}人格画像】
{description}

【认知运作方式】
{cognitive}

【说话风格指南】
{style}

约束：
- 每个回答必须自然体现{mbti_type}的认知模式和说话方式
- 不要提及MBTI术语、不要评价自己的性格——只需成为{mbti_type}
- 答案控制在3-5句话，简洁自然"""

            for scenario in MBTI_SCENARIOS:
                scenario_text = f"{scenario['scenario']}\n（{scenario.get('instruction', '')}）"

                try:
                    response = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.35, max_tokens=350,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": scenario_text}],
                    ))
                    answer = response.choices[0].message.content.strip()
                    if not answer:
                        response = retry_api_call(lambda: client.chat.completions.create(
                            model=MODEL_NAME, temperature=0.45, max_tokens=350,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": scenario_text}],
                        ))
                        answer = response.choices[0].message.content.strip()
                except Exception:
                    continue

                judge_prompt = f"""评估以下回答是否体现了{mbti_type}型人格的特征。

{mbti_type}特征参考：{description}
场景：{scenario_text}
回答：{answer}

评判标准：回答的整体思维方式、语气风格和价值观是否与{mbti_type}一致。不要求每个细节都匹配，模棱两可倾向判true。

严格要求：reason字段必须写一句10字以上的具体中文评判。禁止写空字符串。

只输出JSON（一行，不要markdown）：
{{"consistent": true, "confidence": 0.85, "reason": "此处写具体评判理由"}}"""

                try:
                    judge_resp = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.1, max_tokens=200,
                        messages=[{"role": "user", "content": judge_prompt}],
                    ))
                    judge = parse_judge_result(judge_resp.choices[0].message.content.strip())
                except Exception:
                    judge = {"consistent": False, "confidence": 0.5, "reason": "评判失败"}

                type_results.append({
                    "mbti_type": mbti_type, "dimension": scenario["dimension"],
                    "scenario": scenario["scenario"], "answer": answer,
                    "consistent": judge["consistent"], "confidence": judge["confidence"],
                    "reason": judge["reason"],
                })
                all_results.append(type_results[-1])

            if type_results:
                correct = sum(1 for r in type_results if r["consistent"])
                type_accuracy[mbti_type] = {
                    "accuracy": round(correct / len(type_results), 4),
                    "correct": correct, "total": len(type_results),
                    "avg_confidence": round(sum(r["confidence"] for r in type_results) / len(type_results), 4),
                }

            # 每个类型完成后推送进度（含实时错误统计）
            errors_so_far = [r for r in all_results if not r["consistent"]]
            error_info = {
                "error_count": len(errors_so_far),
                "total_samples": len(all_results),
                "error_rate": round(len(errors_so_far) / len(all_results), 4) if all_results else 0,
                "by_type": {t: sum(1 for e in errors_so_far if e["mbti_type"] == t) for t in test_types},
                "by_dimension": {s["dimension"]: sum(1 for e in errors_so_far if e["dimension"] == s["dimension"]) for s in MBTI_SCENARIOS},
            }
            yield f"data: {json.dumps({'type': 'progress', 'mbti': mbti_type, 'current': idx + 1, 'total': total, 'accuracy': type_accuracy[mbti_type], 'type_accuracy': {mbti: type_accuracy[mbti] for mbti in type_accuracy}, 'error': error_info}, ensure_ascii=False)}\n\n"

        # 全部完成
        all_correct = sum(1 for r in all_results if r["consistent"])
        overall_accuracy = all_correct / len(all_results) if all_results else 0
        summary = {
            "total_samples": len(all_results),
            "overall_accuracy": round(overall_accuracy, 4),
            "overall_accuracy_pct": f"{overall_accuracy:.1%}",
            "avg_confidence": round(sum(r["confidence"] for r in all_results) / len(all_results), 4) if all_results else 0,
            "tested_types": test_types,
        }
        errors = [r for r in all_results if not r["consistent"]]
        error_info = {
            "error_count": len(errors),
            "total_samples": len(all_results),
            "error_rate": round(len(errors) / len(all_results), 4) if all_results else 0,
            "by_type": {t: sum(1 for e in errors if e["mbti_type"] == t) for t in test_types},
            "by_dimension": {s["dimension"]: sum(1 for e in errors if e["dimension"] == s["dimension"]) for s in MBTI_SCENARIOS},
        }
        # 保存错误分析
        errors = [r for r in all_results if not r["consistent"]]
        error_data = {
            "total_samples": len(all_results),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(all_results), 4) if all_results else 0,
            "by_type": {t: sum(1 for e in errors if e["mbti_type"] == t) for t in test_types},
            "by_dimension": {s["dimension"]: sum(1 for e in errors if e["dimension"] == s["dimension"]) for s in MBTI_SCENARIOS},
        }
        with open("error_analysis.json", "w", encoding="utf-8") as f:
            json.dump({**error_data, "error_samples": errors}, f, ensure_ascii=False, indent=2)

        yield f"data: {json.dumps({'type': 'complete', 'summary': summary, 'type_accuracy': type_accuracy, 'error': error_info}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/eval/ablation")
def api_get_ablation():
    """返回消融实验结果（由 eval_mbti.py --ablation 生成）"""
    ablation_file = "ablation_study_results.json"
    if not os.path.exists(ablation_file):
        return {"code": 404, "message": "尚未运行消融实验，请在终端执行：python eval_mbti.py --ablation"}
    with open(ablation_file, "r", encoding="utf-8") as f:
        return {"code": 200, "data": json.load(f)}


@app.post("/api/eval/ablation/stream")
def api_eval_ablation_stream(config: dict = None):
    """SSE流式消融实验：逐类型对比有/无风格指令的准确率"""
    import random as _random
    all_types = list(MBTI_DESCRIPTIONS.keys())
    test_types = (config or {}).get("test_types", None)
    if (config or {}).get("run_all"):
        test_types = all_types
    elif test_types is None:
        test_types = _random.sample(all_types, min(4, len(all_types)))

    def generate():
        type_acc_a = {}
        type_acc_b = {}
        total = len(test_types)

        for idx, mbti_type in enumerate(test_types):
            results_a = []
            results_b = []
            description = MBTI_DESCRIPTIONS.get(mbti_type, mbti_type)
            style = MBTI_SPEAKING_STYLES.get(mbti_type, "")
            cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti_type, "")

            # 系统提示词A（有风格）
            system_a = f"""你是{mbti_type}型人格。请严格以该类型的身份回答用户问题。

【{mbti_type}人格画像】
{description}

【认知运作方式】
{cognitive}

【说话风格指南】
{style}

约束：
- 每个回答必须自然体现{mbti_type}的认知模式和说话方式
- 不要提及MBTI术语、不要评价自己的性格——只需成为{mbti_type}
- 答案控制在3-5句话，简洁自然"""

            # 系统提示词B（无风格，仅标签）
            system_b = f"你的MBTI性格类型是{mbti_type}。请以第一人称回答以下问题。"

            for scenario in MBTI_SCENARIOS:
                scenario_text = f"{scenario['scenario']}\n（{scenario.get('instruction', '')}）"
                def make_judge_prompt(scenario_txt, answer_txt):
                    return f"""评估以下回答是否体现了{mbti_type}型人格的特征。

{mbti_type}特征参考：{description}
场景：{scenario_txt}
回答：{answer_txt}

评判标准：回答的整体思维方式、语气风格和价值观是否与{mbti_type}一致。不要求每个细节都匹配，模棱两可倾向判true。
严格要求：reason字段必须写一句10字以上的具体中文评判。禁止写空字符串。
只输出JSON（一行，不要markdown）：
{{"consistent": true, "confidence": 0.85, "reason": "此处写具体评判理由"}}"""

                # 实验组A
                try:
                    resp = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.35, max_tokens=350,
                        messages=[{"role": "system", "content": system_a}, {"role": "user", "content": scenario_text}],
                    ))
                    ans_a = resp.choices[0].message.content.strip()
                    if not ans_a:
                        resp = retry_api_call(lambda: client.chat.completions.create(
                            model=MODEL_NAME, temperature=0.45, max_tokens=350,
                            messages=[{"role": "system", "content": system_a}, {"role": "user", "content": scenario_text}],
                        ))
                        ans_a = resp.choices[0].message.content.strip()
                    judge_a = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.1, max_tokens=200,
                        messages=[{"role": "user", "content": make_judge_prompt(scenario_text, ans_a)}],
                    ))
                    judge_a = parse_judge_result(judge_a.choices[0].message.content.strip())
                    results_a.append(judge_a["consistent"])
                except Exception:
                    pass

                # 对照组B
                try:
                    resp = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.35, max_tokens=350,
                        messages=[{"role": "system", "content": system_b}, {"role": "user", "content": scenario_text}],
                    ))
                    ans_b = resp.choices[0].message.content.strip()
                    if not ans_b:
                        resp = retry_api_call(lambda: client.chat.completions.create(
                            model=MODEL_NAME, temperature=0.45, max_tokens=350,
                            messages=[{"role": "system", "content": system_b}, {"role": "user", "content": scenario_text}],
                        ))
                        ans_b = resp.choices[0].message.content.strip()
                    judge_b = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.1, max_tokens=200,
                        messages=[{"role": "user", "content": make_judge_prompt(scenario_text, ans_b)}],
                    ))
                    judge_b = parse_judge_result(judge_b.choices[0].message.content.strip())
                    results_b.append(judge_b["consistent"])
                except Exception:
                    pass

            if results_a:
                acc_a = sum(results_a) / len(results_a)
                type_acc_a[mbti_type] = round(acc_a, 4)
            if results_b:
                acc_b = sum(results_b) / len(results_b)
                type_acc_b[mbti_type] = round(acc_b, 4)

            delta = type_acc_a.get(mbti_type, 0) - type_acc_b.get(mbti_type, 0)
            yield f"data: {json.dumps({'type': 'progress', 'mbti': mbti_type, 'current': idx + 1, 'total': total, 'with_style': type_acc_a.get(mbti_type, 0), 'without_style': type_acc_b.get(mbti_type, 0), 'delta': round(delta, 4), 'type_comparison': {t: {'with_style': type_acc_a.get(t, 0), 'without_style': type_acc_b.get(t, 0), 'improvement': round(type_acc_a.get(t, 0) - type_acc_b.get(t, 0), 4)} for t in type_acc_a}}, ensure_ascii=False)}\n\n"

        overall_a = sum(type_acc_a.values()) / len(type_acc_a) if type_acc_a else 0
        overall_b = sum(type_acc_b.values()) / len(type_acc_b) if type_acc_b else 0
        summary = {
            "with_style_accuracy": round(overall_a, 4),
            "without_style_accuracy": round(overall_b, 4),
            "improvement": round(overall_a - overall_b, 4),
            "tested_types": test_types,
        }
        type_comparison = {t: {
            "with_style": type_acc_a.get(t, 0),
            "without_style": type_acc_b.get(t, 0),
            "improvement": round(type_acc_a.get(t, 0) - type_acc_b.get(t, 0), 4),
        } for t in type_acc_a}

        yield f"data: {json.dumps({'type': 'complete', 'summary': summary, 'type_comparison': type_comparison}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/eval/mbti/error_analysis/stream")
def api_eval_error_analysis_stream(config: dict = None):
    """SSE流式错误分析：快速评估并流式返回错误分布"""
    import random as _random
    all_types = list(MBTI_DESCRIPTIONS.keys())
    test_types = (config or {}).get("test_types", None)
    if test_types is None:
        test_types = _random.sample(all_types, min(4, len(all_types)))

    def generate():
        all_results = []
        errors_by_type = {}
        errors_by_dim = {}
        total = len(test_types)

        for idx, mbti_type in enumerate(test_types):
            description = MBTI_DESCRIPTIONS.get(mbti_type, mbti_type)
            style = MBTI_SPEAKING_STYLES.get(mbti_type, "")
            cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti_type, "")

            system_prompt = f"""你是{mbti_type}型人格。请严格以该类型的身份回答用户问题。

【{mbti_type}人格画像】
{description}

【认知运作方式】
{cognitive}

【说话风格指南】
{style}

约束：
- 每个回答必须自然体现{mbti_type}的认知模式和说话方式
- 不要提及MBTI术语、不要评价自己的性格——只需成为{mbti_type}
- 答案控制在3-5句话，简洁自然"""

            for scenario in MBTI_SCENARIOS:
                scenario_text = f"{scenario['scenario']}\n（{scenario.get('instruction', '')}）"
                try:
                    response = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.35, max_tokens=350,
                        messages=[{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": scenario_text}],
                    ))
                    answer = response.choices[0].message.content.strip()
                    if not answer:
                        response = retry_api_call(lambda: client.chat.completions.create(
                            model=MODEL_NAME, temperature=0.45, max_tokens=350,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": scenario_text}],
                        ))
                        answer = response.choices[0].message.content.strip()
                except Exception:
                    continue

                judge_prompt = f"""评估以下回答是否体现了{mbti_type}型人格的特征。

{mbti_type}特征参考：{description}
场景：{scenario_text}
回答：{answer}

评判标准：回答的整体思维方式、语气风格和价值观是否与{mbti_type}一致。不要求每个细节都匹配，模棱两可倾向判true。
严格要求：reason字段必须写一句10字以上的具体中文评判。禁止写空字符串。
只输出JSON（一行，不要markdown）：
{{"consistent": true, "confidence": 0.85, "reason": "此处写具体评判理由"}}"""

                try:
                    judge_resp = retry_api_call(lambda: client.chat.completions.create(
                        model=MODEL_NAME, temperature=0.1, max_tokens=200,
                        messages=[{"role": "user", "content": judge_prompt}],
                    ))
                    judge = parse_judge_result(judge_resp.choices[0].message.content.strip())
                except Exception:
                    judge = {"consistent": False, "confidence": 0.5, "reason": "评判失败"}

                result = {
                    "mbti_type": mbti_type, "dimension": scenario["dimension"],
                    "answer": answer, "consistent": judge["consistent"],
                    "confidence": judge["confidence"], "reason": judge["reason"],
                }
                all_results.append(result)
                if not judge["consistent"]:
                    errors_by_type[mbti_type] = errors_by_type.get(mbti_type, 0) + 1
                    errors_by_dim[scenario["dimension"]] = errors_by_dim.get(scenario["dimension"], 0) + 1

            error_count = sum(errors_by_type.values())
            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total, 'error_count': error_count, 'total_samples': len(all_results), 'by_type': errors_by_type, 'by_dim': errors_by_dim}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'error_count': error_count, 'total_samples': len(all_results), 'error_rate': round(error_count / len(all_results), 4) if all_results else 0, 'by_type': errors_by_type, 'by_dim': errors_by_dim}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/eval/mbti/error_analysis")
def api_get_error_analysis():
    """返回最近一次 MBTI 评估的错误分析"""
    error_file = "error_analysis.json"
    if not os.path.exists(error_file):
        return {"code": 404, "message": "尚未运行错误分析，请先执行评估"}
    with open(error_file, "r", encoding="utf-8") as f:
        return {"code": 200, "data": json.load(f)}


@app.get("/api/eval/mbti/history")
def api_get_mbti_eval_history():
    """获取历史MBTI评估结果列表"""
    results = []
    if not os.path.exists(EVAL_DIR):
        return {"code": 200, "data": results}
    for filename in sorted(os.listdir(EVAL_DIR), reverse=True):
        if filename.startswith("mbti_eval_") and filename.endswith(".json"):
            filepath = os.path.join(EVAL_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "file": filename,
                    "time": data["summary"].get("evaluation_time", ""),
                    "accuracy": data["summary"].get("overall_accuracy_pct", ""),
                    "samples": data["summary"].get("total_samples", 0),
                })
            except Exception:
                continue
    return {"code": 200, "data": results}


# ===================== BLEU 评估接口 =====================
def load_bleu_eval_samples():
    from config import MAX_EVAL_SAMPLES
    samples = []
    if not os.path.exists(COSER_FOLDER):
        return samples
    for filename in os.listdir(COSER_FOLDER):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(COSER_FOLDER, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for plot in data.get("plots", []):
                for conv in plot.get("conversation", []):
                    dialogues = conv.get("dialogues", [])
                    if len(dialogues) < 2:
                        continue
                    for i in range(1, len(dialogues)):
                        context = dialogues[:i]
                        reference = dialogues[i]
                        samples.append({
                            "book": filename.replace(".json", ""),
                            "scenario": conv.get("scenario", ""),
                            "topic": conv.get("topic", ""),
                            "context": context,
                            "reference_char": reference["character"],
                            "reference_text": reference["message"],
                        })
                        if len(samples) >= MAX_EVAL_SAMPLES:
                            return samples
        except Exception:
            continue
    return samples


def compute_bleu(reference, candidate):
    try:
        from eval_bleu import tokenize
        ref_tokens = [tokenize(reference)]
        cand_tokens = tokenize(candidate)
    except ImportError:
        ref_tokens = [reference.split()]
        cand_tokens = candidate.split()
    smooth = SmoothingFunction()
    try:
        return sentence_bleu(ref_tokens, cand_tokens,
                             weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smooth.method1)
    except Exception:
        return 0.0


@app.post("/api/eval/bleu")
async def api_eval_bleu(config: dict = None):
    """
    运行BLEU对话质量评估
    可选参数: {"max_samples": 50, "mbti_type": "ENFJ"}
    """
    max_samples = (config or {}).get("max_samples", MAX_EVAL_SAMPLES)
    mbti_type = (config or {}).get("mbti_type", None)

    samples = load_bleu_eval_samples()[:max_samples]
    if not samples:
        return {"code": 500, "message": "未加载到评估样本"}

    system_prompt = None
    if mbti_type and mbti_type in MBTI_SPEAKING_STYLES:
        system_prompt = build_character_prompt({
            "name": f"评估角色({mbti_type})",
            "mbti": mbti_type,
            "personality": MBTI_DESCRIPTIONS.get(mbti_type, ""),
            "career": "未知",
            "intro": f"一个{mbti_type}性格类型的人物",
            "experience": "未知",
        })

    bleu_scores = []
    results = []

    for idx, sample in enumerate(samples):
        context_text = ""
        for d in sample["context"]:
            context_text += f"{d['character']}：{d['message']}\n"

        prompt = f"""背景：{sample['scenario']}

以下是一段对话，请以 {sample['reference_char']} 的身份回复下一句话，保持角色风格一致：

{context_text}{sample['reference_char']}："""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = retry_api_call(lambda: client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                max_tokens=300,
            ))
            generated = response.choices[0].message.content.strip()
            bleu = compute_bleu(sample["reference_text"], generated)
            bleu_scores.append(bleu)
            results.append({
                "book": sample["book"],
                "character": sample["reference_char"],
                "reference": sample["reference_text"],
                "generated": generated,
                "bleu": round(bleu, 4),
            })
        except Exception:
            continue

    if not bleu_scores:
        return {"code": 500, "message": "无有效评估结果"}

    avg_bleu = round(np.mean(bleu_scores), 4)
    std_bleu = round(np.std(bleu_scores), 4)

    summary = {
        "total_samples": len(bleu_scores),
        "avg_bleu": avg_bleu,
        "std_bleu": std_bleu,
        "max_bleu": round(max(bleu_scores), 4),
        "min_bleu": round(min(bleu_scores), 4),
        "mbti_type": mbti_type,
        "evaluation_time": datetime.datetime.now().isoformat(),
    }

    result_data = {"summary": summary, "details": results}
    result_file = os.path.join(EVAL_DIR, f"bleu_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return {"code": 200, "data": result_data}


@app.post("/api/eval/bleu/stream")
def api_eval_bleu_stream(config: dict = None):
    """SSE流式BLEU评估：逐样本返回进度，前端实时更新"""
    max_samples = (config or {}).get("max_samples", 30)
    mbti_type = (config or {}).get("mbti_type", None)

    samples = load_bleu_eval_samples()[:max_samples]
    if not samples:
        def err_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': '未加载到评估样本'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    system_prompt = None
    if mbti_type and mbti_type in MBTI_SPEAKING_STYLES:
        system_prompt = build_character_prompt({
            "name": f"评估角色({mbti_type})", "mbti": mbti_type,
            "personality": MBTI_DESCRIPTIONS.get(mbti_type, ""),
            "career": "未知", "intro": f"一个{mbti_type}性格类型的人物",
            "experience": "未知",
        })

    def generate():
        bleu_scores = []
        results = []
        total = len(samples)

        for idx, sample in enumerate(samples):
            context_text = ""
            for d in sample["context"]:
                context_text += f"{d['character']}：{d['message']}\n"
            prompt = f"""背景：{sample['scenario']}

以下是一段对话，请以 {sample['reference_char']} 的身份回复下一句话，保持角色风格一致：

{context_text}{sample['reference_char']}："""

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                response = retry_api_call(lambda: client.chat.completions.create(
                    model=MODEL_NAME, messages=messages,
                    temperature=0.5, max_tokens=300,
                ))
                generated = response.choices[0].message.content.strip()
                bleu = compute_bleu(sample["reference_text"], generated)
                bleu_scores.append(bleu)
                results.append({
                    "book": sample["book"], "character": sample["reference_char"],
                    "reference": sample["reference_text"], "generated": generated,
                    "bleu": round(bleu, 4),
                })
            except Exception:
                continue

            # 每5个样本推送一次进度
            if (idx + 1) % 5 == 0 or idx == total - 1:
                avg = round(np.mean(bleu_scores), 4) if bleu_scores else 0
                yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total, 'avg_bleu': avg}, ensure_ascii=False)}\n\n"

        avg_bleu = round(np.mean(bleu_scores), 4) if bleu_scores else 0

        # 计算字符级BLEU和ROUGE-L
        char_bleu = 0.0
        rouge_f1 = 0.0
        if results:
            try:
                from eval_bleu import tokenize, corpus_bleu, rouge_l
                ref_texts = [r["reference"] for r in results]
                gen_texts = [r["generated"] for r in results]
                char_bleu, _, _ = corpus_bleu([[list(r)] for r in ref_texts], [list(g) for g in gen_texts])
                rouge_f1, _, _ = rouge_l([[tokenize(r)] for r in ref_texts], [tokenize(g) for g in gen_texts])
            except Exception:
                pass

        summary = {
            "total_samples": len(bleu_scores),
            "avg_bleu": avg_bleu,
            "corpus_bleu_char": round(char_bleu, 4),
            "rouge_l_f1": round(rouge_f1, 4),
            "std_bleu": round(np.std(bleu_scores), 4) if bleu_scores else 0,
            "mbti_type": mbti_type,
        }
        yield f"data: {json.dumps({'type': 'complete', 'summary': summary}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/eval/bleu/history")
def api_get_bleu_eval_history():
    results = []
    if not os.path.exists(EVAL_DIR):
        return {"code": 200, "data": results}
    for filename in sorted(os.listdir(EVAL_DIR), reverse=True):
        if filename.startswith("bleu_eval_") and filename.endswith(".json"):
            filepath = os.path.join(EVAL_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "file": filename,
                    "time": data["summary"].get("evaluation_time", ""),
                    "avg_bleu": data["summary"].get("avg_bleu", 0),
                    "samples": data["summary"].get("total_samples", 0),
                    "mbti_type": data["summary"].get("mbti_type", "N/A"),
                })
            except Exception:
                continue
    return {"code": 200, "data": results}


# ===================== MBTI 类型信息 =====================
@app.get("/api/mbti/types")
def api_get_mbti_types():
    """返回所有MBTI类型的描述和说话风格"""
    types = []
    for code, desc in MBTI_DESCRIPTIONS.items():
        types.append({
            "code": code,
            "description": desc,
            "style": MBTI_SPEAKING_STYLES.get(code, ""),
        })
    return {"code": 200, "data": types}


# ===================== 对话记忆系统 =====================
MEMORY_DIR = os.path.join(DATA_DIR, "memories")


def load_memories(character_name):
    """加载角色的持久记忆"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', character_name)
    filepath = os.path.join(MEMORY_DIR, f"{safe_name}.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memories(character_name, memories):
    """保存角色的持久记忆"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', character_name)
    filepath = os.path.join(MEMORY_DIR, f"{safe_name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def extract_memories_from_chat(character_name, history):
    """使用LLM从对话历史中提取用户关键信息作为角色记忆"""
    if not history or len(history) < 3:
        return []

    # 构建对话摘要
    chat_text = ""
    for msg in history[-20:]:
        role = "用户" if msg.get("role") == "user" else character_name
        chat_text += f"{role}：{msg.get('content', '')}\n"

    prompt = f"""从以下对话中提取用户透露的关键信息，以便角色 {character_name} 在下次对话中记住。

对话内容：
{chat_text}

请输出一个JSON数组，每个元素是一个记忆点（简洁的一句话）。只提取用户相关的重要信息（如姓名、喜好、经历、需求、约定等），不要输出无关内容。
示例：["用户叫张三，是人工智能专业大三学生", "用户正在准备考研，目标是北航", "用户喜欢喝咖啡，不喜欢甜食"]
只输出JSON数组："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        # 提取JSON数组
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return []


@app.get("/api/memory/{character_name}")
def api_get_memories(character_name: str):
    memories = load_memories(character_name)
    return {"code": 200, "data": memories}


@app.delete("/api/memory/{character_name}/{index}")
def api_delete_memory(character_name: str, index: int):
    memories = load_memories(character_name)
    if 0 <= index < len(memories):
        memories.pop(index)
        save_memories(character_name, memories)
    return {"code": 200, "data": memories}


@app.post("/api/memory/extract")
def api_extract_memories(request: dict):
    """从对话中提取并保存记忆"""
    character_name = request.get("character_name", "角色")
    history = request.get("history", [])
    new_memories = extract_memories_from_chat(character_name, history)
    if new_memories:
        existing = load_memories(character_name)
        # 去重合并
        existing_set = set(existing)
        for m in new_memories:
            if m not in existing_set:
                existing.append(m)
        save_memories(character_name, existing[-20:])  # 最多保留20条
    return {"code": 200, "data": {"added": len(new_memories), "memories": load_memories(character_name)}}


def build_character_prompt_with_memory(config: dict) -> str:
    """构建含持久记忆的系统提示词"""
    base_prompt = build_character_prompt(config)
    name = config.get("name", "")
    if name:
        memories = load_memories(name)
        if memories:
            memory_text = "\n".join([f"  - {m}" for m in memories[-10:]])  # 最近10条
            base_prompt += f"\n\n【历史记忆——你记得关于用户的以下信息】\n{memory_text}\n请在对话中恰当地运用这些记忆，让交流更自然亲切。"
    return base_prompt


# ===================== 最新完整评估结果 =====================
@app.get("/api/eval/mbti/latest")
def api_get_latest_mbti_eval():
    """返回最近一次 MBTI 评估的完整结果（优先Web端保存，其次命令行输出）"""
    # 优先检查 EVAL_DIR 中时间戳文件
    if os.path.exists(EVAL_DIR):
        files = sorted(
            [f for f in os.listdir(EVAL_DIR) if f.startswith("mbti_eval_") and f.endswith(".json")],
            reverse=True,
        )
        if files:
            with open(os.path.join(EVAL_DIR, files[0]), "r", encoding="utf-8") as f:
                return {"code": 200, "data": json.load(f)}
    # 回退：检查命令行输出的固定文件名
    for cli_file in ["mbti_evaluation_results.json"]:
        if os.path.exists(cli_file):
            with open(cli_file, "r", encoding="utf-8") as f:
                return {"code": 200, "data": json.load(f)}
    return {"code": 404, "message": "暂无评估结果，请先在终端运行 python eval_mbti.py"}


@app.get("/api/eval/bleu/latest")
def api_get_latest_bleu_eval():
    """返回最近一次 BLEU 评估的完整结果（优先Web端保存，其次命令行输出）"""
    if os.path.exists(EVAL_DIR):
        files = sorted(
            [f for f in os.listdir(EVAL_DIR) if f.startswith("bleu_eval_") and f.endswith(".json")],
            reverse=True,
        )
        if files:
            with open(os.path.join(EVAL_DIR, files[0]), "r", encoding="utf-8") as f:
                return {"code": 200, "data": json.load(f)}
    for cli_file in ["bleu_evaluation_results.json"]:
        if os.path.exists(cli_file):
            with open(cli_file, "r", encoding="utf-8") as f:
                return {"code": 200, "data": json.load(f)}
    return {"code": 404, "message": "暂无评估结果，请先在终端运行 python eval_bleu.py"}


# ===================== 启动 =====================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
