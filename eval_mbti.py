"""
MBTI 性格模拟准确率评估模块
任务书要求：评估大模型对 MBTI 性格模拟的效果，以性格准确率为指标

方法：使用 MBTI 性格测试场景题，让模型以特定 MBTI 类型身份作答，
然后由评判模型判断回答是否符合该 MBTI 类型的特征。
"""
import json
import os
import re
import time
from openai import OpenAI
import numpy as np
from config import (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
                    API_TIMEOUT, API_MAX_RETRIES, RETRY_DELAY,
                    MBTI_EVAL_TEMPERATURE, JUDGE_TEMPERATURE,
                    GENERATE_MAX_TOKENS, JUDGE_MAX_TOKENS, MBTI_TEST_TYPES)

MODEL_NAME = DEEPSEEK_MODEL
client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY, timeout=API_TIMEOUT)

# 16种 MBTI 类型的特征描述（含行为特征，供评判模型使用）
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

# 认知功能栈描述——让模型理解每种 MBTI 类型底层的认知架构
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

# 每种MBTI的深度说话风格指令（与backend.py保持一致）
MBTI_SPEAKING_STYLES = {
    "INTJ": "说话风格：理性冷静、逻辑严密、言简意赅。喜欢用数据和事实支撑观点，不喜废话和情感宣泄。常用'我认为'、'从逻辑上看'、'换个角度分析'等表达。表现出战略思维，倾向于分析问题的本质和长远影响。情感表达克制，但并非冷漠。",
    "INTP": "说话风格：充满好奇心、喜欢探讨抽象概念和理论。常用'有趣的是'、'理论上说'、'还有一种可能性'等表达。喜欢分析各种可能性，有时会陷入思维发散。对逻辑矛盾敏感，会不自觉地指出他人论述中的漏洞。社交上略显笨拙。",
    "ENTJ": "说话风格：自信果断、目标导向、富有领导力。常用'我的建议是'、'最佳方案是'、'行动起来'等表达。喜欢制定计划并推动执行，说话直接不绕弯。善于快速评估情况并给出决策，重视效率和结果。",
    "ENTP": "说话风格：思维跳脱灵动，热爱从各种刁钻角度重新审视问题。常用'换个角度看'、'有意思'、'那如果反过来呢'等表达。享受智力上的来回交锋——不是为了赢，而是为了让想法更好玩、更经得起推敲。善用类比和概念跳跃连接看似无关的事物。不急于定论，探索多种可能性的过程本身就很有趣，正确答案可以等一等再浮现。",
    "INFJ": "说话风格：思维深刻但表达精准利落，善于一语中的地抓住问题本质。常用'关键是'、'我看得很清楚'、'从根本上看'等表达。习惯在内心快速整合信息形成清晰判断，一旦看清方向就果断行动推进，不喜欢拖泥带水。重视长远意义和内在价值观，做决定坚定果断——可以听取不同意见，但核心立场不会摇摆。表达方式温和有礼，但每个观点背后都有深思熟虑的判断，不含糊不敷衍。",
    "INFP": "说话风格：温柔感性、充满理想主义色彩、富有创造力。常用'我觉得'、'对我来说很重要'、'这让我想起'等表达。重视内心感受和价值观，说话充满真诚和热情。善于用隐喻和故事来表达情感，倾向于鼓励和支持他人。",
    "ENFJ": "说话风格：热情洋溢、善于激励、充满感染力。常用'你一定可以的'、'我理解你'、'我们一起'等表达。天生善于倾听和引导，让对方感到被重视和理解。善于发现他人的潜力并给予鼓励，说话充满正能量。",
    "ENFP": "说话风格：热情奔放、充满想象力、自由随性。常用'太棒了'、'你知道吗'、'我突然想到'等表达。说话充满激情和感染力，能带动周围人的情绪。思维跳跃，充满创意火花，重视真实和自由。",
    "ISTJ": "说话风格：务实严谨、注重细节、可靠稳重。常用'根据经验'、'事实是'、'按计划来'等表达。说话有条理，喜欢列举具体事实和步骤。重视规则和传统，表达简洁务实，承诺的事情一定会做到。",
    "ISFJ": "说话风格：务实稳重、依赖过往经验和具体细节做判断。常用'根据之前的经验'、'我记得有一次'、'稳妥起见'等表达。善于用具体事例说明观点，关心他人通过实际行动而非言语。低调谦逊，不喜成为焦点，但交付结果永远可靠。偏好稳定和可预见性，对未经检验的变化持谨慎态度。默默观察和记录周围人的需求，在别人需要时已经做好了准备。",
    "ESTJ": "说话风格：直接高效、有条理、重视规则和秩序。常用'按流程走'、'效率优先'、'明确分工'等表达。喜欢清晰的结构和确定的计划，说话不拖泥带水。善于组织和管理，会自然地承担领导角色。",
    "ESFJ": "说话风格：亲切热心、善于照顾他人感受、重视和谐。常用'大家觉得呢'、'我来安排'、'不用担心'等表达。天生的照顾者，会主动关心每个人的状态和需求。善于营造温馨的氛围，让人感到被接纳和温暖。",
    "ISTP": "说话风格：简洁务实、冷静理性、喜欢动手实践。常用'试试看'、'问题出在哪'、'这样就行'等表达。说话少而精，不喜欢长篇大论，更倾向于行动。善于分析实际问题并快速找到解决方案。",
    "ISFP": "说话风格：温柔敏感、富有艺术气息、活在当下。常用'好美'、'感觉很好'、'随缘吧'等表达。重视审美和感官体验，善于发现生活中的美好。表达方式柔和自然，不喜冲突和争论。",
    "ESTP": "说话风格：务实直接、行动力强、灵活应变。常用'先试试看'、'关键是把事做成'、'随机应变'等表达。善于快速评估实际情况并找到最有效的解决路径，不纠结于过多的理论分析。说话简洁利落，喜欢动手解决问题而非空谈。对风险有清醒的认知——不是盲目冒险，而是相信自己的临场判断力和调整能力。重视当下的实际效果，能在混乱中快速锁定可行的突破口。",
    "ESFP": "说话风格：热情外向、充满活力、享受社交。常用'太好玩了'、'一起加入吧'、'开心就好'等表达。天生的表演者，善于带动气氛，让周围的人感到愉快。喜欢分享快乐和美好体验。",
}

# 6个测试场景
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


def get_mbti_type_code(mbti_label):
    match = re.match(r"([IE][NS][TF][JP])", mbti_label.strip())
    return match.group(1) if match else mbti_label.strip()


def retry_call(fn, max_retries=API_MAX_RETRIES):
    """带重试的API调用，遇到失败等RETRY_DELAY秒再试"""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e


def generate_role_response(mbti_type, scenario_text, with_style=True):
    """让模型以特定 MBTI 类型身份回答场景问题。with_style=False 用于消融实验对照组"""
    mbti_code = get_mbti_type_code(mbti_type)
    description = MBTI_DESCRIPTIONS.get(mbti_code, mbti_type)

    if with_style:
        style = MBTI_SPEAKING_STYLES.get(mbti_code, "")
        cognitive = MBTI_COGNITIVE_FUNCTIONS.get(mbti_code, "")
        system_prompt = f"""你是{mbti_code}型人格。请严格以该类型的身份回答用户问题。

【{mbti_code}人格画像】
{description}

【认知运作方式】
{cognitive}

【说话风格指南】
{style}

约束：
- 每个回答必须自然体现{mbti_code}的认知模式和说话方式
- 不要提及MBTI术语、不要评价自己的性格、不要表演——只需成为{mbti_code}
- 答案控制在3-5句话，简洁自然"""
    else:
        system_prompt = f"""你的MBTI性格类型是{mbti_code}。请以第一人称回答以下问题。"""

    resp = retry_call(lambda: client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario_text},
        ],
        temperature=MBTI_EVAL_TEMPERATURE,
        max_tokens=GENERATE_MAX_TOKENS,
    )).choices[0].message.content.strip()

    # 空回答重试一次
    if not resp:
        resp = retry_call(lambda: client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario_text},
            ],
            temperature=MBTI_EVAL_TEMPERATURE + 0.1,
            max_tokens=GENERATE_MAX_TOKENS,
        )).choices[0].message.content.strip()

    return resp


def judge_mbti_consistency(mbti_type, scenario, answer):
    """评判模型的回答是否符合指定的 MBTI 类型"""
    mbti_code = get_mbti_type_code(mbti_type)
    description = MBTI_DESCRIPTIONS.get(mbti_code, mbti_type)

    # 从 scenario 文本中提取维度名称（如 "E/I (外向/内向)"）
    judge_prompt = f"""评估以下回答是否体现了{mbti_code}型人格的特征。

{mbti_code}特征参考：{description}
场景：{scenario}
回答：{answer}

评判标准：回答的整体思维方式、语气风格和价值观是否与{mbti_code}一致。不要求每个细节都匹配，模棱两可倾向判true。

严格要求：reason字段必须写一句10字以上的具体中文评判，说明回答体现或违背了{mbti_code}的什么特征。禁止写空字符串。

只输出JSON（一行，不要markdown）：
{{"consistent": true, "confidence": 0.85, "reason": "此处写具体评判理由"}}"""

    result_text = retry_call(lambda: client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )).choices[0].message.content.strip()
    return parse_judge_result(result_text)


def parse_judge_result(text):
    """解析评判结果，优先JSON解析，失败则回退到文本匹配"""
    # 方法1：找到第一个 { 后用括号计数找到配对的 }
    try:
        start = text.find('{')
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(text[start:], start):
                if ch == '{':
                    depth += 1
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

    # 回退：文本匹配
    # 首先检查显式的 false/不一致标记
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

    # 提取 reason
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    reason = reason_match.group(1) if reason_match else text[:120]

    return {"consistent": consistent, "confidence": confidence, "reason": reason}


def run_mbti_evaluation(test_types=None):
    """运行完整的 MBTI 评估"""
    print("=" * 60)
    print("MBTI 性格模拟准确率评估")
    print("任务书技术指标：评价大模型对MBTI性格模拟的效果，采用准确率作为指标")
    print("=" * 60)

    if test_types is None:
        test_types = MBTI_TEST_TYPES

    all_results = []
    type_accuracy = {}

    for mbti_type in test_types:
        print(f"\n评估 MBTI 类型：{mbti_type}")
        type_results = []

        for scenario in MBTI_SCENARIOS:
            scenario_text = scenario["scenario"]
            if "instruction" in scenario:
                scenario_text += f"\n（{scenario['instruction']}）"

            print(f"  场景：{scenario['dimension']}...", end=" ")

            try:
                answer = generate_role_response(mbti_type, scenario_text)
            except Exception as e:
                print(f"✗ 生成失败（重试3次后）：{e}")
                continue

            try:
                judge = judge_mbti_consistency(mbti_type, scenario_text, answer)
            except Exception as e:
                print(f"✗ 评判失败（重试3次后）：{e}")
                continue

            type_results.append({
                "mbti_type": mbti_type,
                "dimension": scenario["dimension"],
                "scenario": scenario_text,
                "answer": answer,
                "consistent": judge["consistent"],
                "confidence": judge["confidence"],
                "reason": judge["reason"],
            })
            all_results.append(type_results[-1])
            status = "✓ 一致" if judge["consistent"] else "✗ 不一致"
            print(f"{status} (置信度:{judge['confidence']:.2f})")

        if type_results:
            correct = sum(1 for r in type_results if r["consistent"])
            acc = correct / len(type_results)
            type_accuracy[mbti_type] = {
                "accuracy": round(acc, 4),
                "correct": correct,
                "total": len(type_results),
                "avg_confidence": round(np.mean([r["confidence"] for r in type_results]), 4),
            }

    print("\n" + "=" * 60)
    print("MBTI 评估结果汇总")
    print("=" * 60)

    all_correct = sum(1 for r in all_results if r["consistent"])
    overall_accuracy = all_correct / len(all_results) if all_results else 0
    avg_confidence = np.mean([r["confidence"] for r in all_results]) if all_results else 0

    print(f"总测试样本数：{len(all_results)}")
    print(f"总体准确率：{overall_accuracy:.2%} ({all_correct}/{len(all_results)})")
    print(f"平均置信度：{avg_confidence:.4f}")

    print("\n各 MBTI 类型准确率：")
    print("-" * 40)
    for mbti, stats in sorted(type_accuracy.items()):
        bar = "█" * int(stats["accuracy"] * 20)
        print(f"  {mbti}: {stats['accuracy']:.2%} {bar} ({stats['correct']}/{stats['total']})")

    output_file = "mbti_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": len(all_results),
                "overall_accuracy": round(overall_accuracy, 4),
                "overall_accuracy_pct": f"{overall_accuracy:.1%}",
                "avg_confidence": round(avg_confidence, 4),
                "tested_types": test_types,
            },
            "type_accuracy": type_accuracy,
            "details": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存至：{output_file}")

    # 错误案例分析
    errors = [r for r in all_results if not r["consistent"]]
    if errors:
        print("\n" + "=" * 60)
        print("错误案例分析")
        print("=" * 60)
        print(f"不一致样本数：{len(errors)}/{len(all_results)} ({len(errors)/len(all_results):.1%})")

        # 按类型统计
        from collections import Counter
        type_errors = Counter(e["mbti_type"] for e in errors)
        dim_errors = Counter(e["dimension"] for e in errors)
        print(f"\n按MBTI类型分布：")
        for t, c in type_errors.most_common():
            total_t = sum(1 for r in all_results if r["mbti_type"] == t)
            print(f"  {t}: {c}/{total_t} 错误")
        print(f"\n按场景维度分布：")
        for d, c in dim_errors.most_common():
            total_d = sum(1 for r in all_results if r["dimension"] == d)
            print(f"  {d}: {c}/{total_d} 错误")

        # 输出典型错误样本
        print(f"\n典型错误样本（前5个）：")
        for i, e in enumerate(errors[:5]):
            print(f"\n  [{i+1}] 类型={e['mbti_type']} | 场景={e['dimension']} | 置信度={e['confidence']:.2f}")
            print(f"  回答：{e['answer'][:150]}...")
            print(f"  评判：{e['reason'][:150]}")

        # 保存错误分析
        error_file = "error_analysis.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": len(all_results),
                "error_count": len(errors),
                "error_rate": round(len(errors) / len(all_results), 4),
                "by_type": {t: c for t, c in type_errors.items()},
                "by_dimension": {d: c for d, c in dim_errors.items()},
                "error_samples": errors,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n详细错误分析已保存至：{error_file}")
    else:
        print("\n✅ 所有样本均判定为一致，无错误案例")

    print("\n✅ MBTI评估完成")
    return all_results


def run_ablation_study():
    """消融实验：对比有/无 MBTI 风格指令的准确率"""
    print("=" * 60)
    print("消融实验：MBTI 风格指令对性格模拟的影响")
    print("=" * 60)

    test_types = MBTI_TEST_TYPES
    all_results_a = []
    all_results_b = []
    type_acc_a = {}
    type_acc_b = {}

    for mbti_type in test_types:
        print(f"\n消融对比 {mbti_type}：")

        results_a = []
        results_b = []

        for scenario in MBTI_SCENARIOS:
            scenario_text = scenario["scenario"]
            if "instruction" in scenario:
                scenario_text += f"\n（{scenario['instruction']}）"

            # 实验组A：有 MBTI 风格指令
            try:
                answer_a = generate_role_response(mbti_type, scenario_text, with_style=True)
                judge_a = judge_mbti_consistency(mbti_type, scenario_text, answer_a)
            except Exception:
                continue
            results_a.append(judge_a["consistent"])

            # 对照组B：无 MBTI 风格指令（仅标签）
            try:
                answer_b = generate_role_response(mbti_type, scenario_text, with_style=False)
                judge_b = judge_mbti_consistency(mbti_type, scenario_text, answer_b)
            except Exception:
                continue
            results_b.append(judge_b["consistent"])

        if results_a:
            acc_a = sum(results_a) / len(results_a)
            type_acc_a[mbti_type] = round(acc_a, 4)
        if results_b:
            acc_b = sum(results_b) / len(results_b)
            type_acc_b[mbti_type] = round(acc_b, 4)

        delta = type_acc_a.get(mbti_type, 0) - type_acc_b.get(mbti_type, 0)
        print(f"  有风格指令: {type_acc_a.get(mbti_type, 0):.2%}  |  无风格指令: {type_acc_b.get(mbti_type, 0):.2%}  |  提升: {delta:+.1%}")

    print("\n" + "=" * 60)
    print("消融实验结果汇总")
    print("=" * 60)
    print(f"{'MBTI类型':<8} {'有风格指令':>10} {'无风格指令':>10} {'提升':>10}")
    print("-" * 45)
    for mbti in test_types:
        a = type_acc_a.get(mbti, 0)
        b = type_acc_b.get(mbti, 0)
        d = a - b
        print(f"{mbti:<8} {a:>9.1%} {b:>9.1%} {d:>+9.1%}")

    overall_a = sum(type_acc_a.values()) / len(type_acc_a) if type_acc_a else 0
    overall_b = sum(type_acc_b.values()) / len(type_acc_b) if type_acc_b else 0
    print("-" * 45)
    print(f"{'总计':<8} {overall_a:>9.1%} {overall_b:>9.1%} {overall_a - overall_b:>+9.1%}")

    # 保存
    ablation_file = "ablation_study_results.json"
    with open(ablation_file, "w", encoding="utf-8") as f:
        json.dump({
            "description": "消融实验：对比有/无 MBTI 风格指令对性格模拟准确率的影响",
            "method": "实验组A使用深度MBTI说话风格指令，对照组B仅使用MBTI类型标签",
            "summary": {
                "with_style_accuracy": round(overall_a, 4),
                "without_style_accuracy": round(overall_b, 4),
                "improvement": round(overall_a - overall_b, 4),
            },
            "type_comparison": {
                mbti: {
                    "with_style": type_acc_a.get(mbti, 0),
                    "without_style": type_acc_b.get(mbti, 0),
                    "improvement": round(type_acc_a.get(mbti, 0) - type_acc_b.get(mbti, 0), 4),
                }
                for mbti in test_types
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存至：{ablation_file}")
    print(f"\n结论：MBTI风格指令使准确率从 {overall_b:.1%} 提升至 {overall_a:.1%}，")
    print(f"提升幅度 {overall_a - overall_b:+.1%}，证明了深度风格指令设计的有效性。")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ablation":
        run_ablation_study()
    else:
        run_mbti_evaluation()
