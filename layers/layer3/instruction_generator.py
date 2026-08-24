STAGE_GOAL = {
    "behavioral_activation": "引导行为激活，探索可执行的小行动",
    "cognitive_restructuring": "识别并挑战负性自动化思维",
    "sleep_hygiene": "睡眠卫生与CBT-I技术指导",
    "stimulus_control": "刺激控制技术",
    "sleep_restriction": "睡眠限制技术",
    "mindfulness": "正念练习与认知解离",
    "cognitive_defusion": "认知解离练习",
    "exposure_response_prevention": "暴露与反应阻止",
    "trauma_processing": "创伤处理（注意禁忌）",
    "interpersonal_rhythm": "人际节律稳定",
    "interpersonal_work": "人际技能工作",
    "solution_focused": "焦点解决取向",
}


class InstructionGenerator:
    def generate(self, report_diagnosis, therapy, profile):
        primary = report_diagnosis.get("primary")
        secondary = report_diagnosis.get("secondary", [])
        ruled_out = report_diagnosis.get("ruled_out", [])

        primary_text = "未确定"
        if primary:
            scale_map = {"MDD": "PHQ-9", "GAD": "GAD-7", "PTSD": "PCL-5",
                         "Insomnia": "ISI", "Bipolar": "MDQ", "OCD": "Y-BOCS"}
            scale_name = scale_map.get(primary["disorder"], "")
            score = profile.get(scale_name.replace("-", ""), "")
            primary_text = (
                f"{primary['disorder']} ({scale_name}={score}，置信度{primary['confidence']})"
            )

        secondary_text = "、".join(s["disorder"] for s in secondary) or "无"
        ruled_out_text = "、".join(
            f"{r['disorder']} ({r.get('reason', '')})" for r in ruled_out
        ) or "无"

        plugin_text = "无"
        if therapy.get("plugins"):
            plugin_text = "、".join(therapy["plugins"])

        basic_therapy = therapy.get("basic_therapy", "CBT")

        instructions = {}
        instructions["L1"] = (
            "你是一名受过CBT训练的心理治疗师。风格：温暖、共情、结构化。"
            "边界：不做医疗建议、不开药、不替代医生。"
        )

        instructions["L2"] = (
            f"主诊断：{primary_text}；共病：{secondary_text}；"
            f"排除：{ruled_out_text}；基础疗法：{basic_therapy}；"
            f"已激活插件：{plugin_text}。诊断反馈使用'初步印象'措辞，禁止'确诊'。"
        )

        instructions["L3"] = (
            "本轮目标：建立共情与治疗联盟，确认当前状态。"
            "技术：开放式提问+反映。禁止：建议、确诊。衔接上轮：无。"
        )

        instructions["L4"] = (
            "禁止：开药、诊断新疾病、预测未来、替代医生、谈论其他患者。"
            "安全触发：用户表达自杀意图→立即终止对话，触发危机协议。"
            "输出：用'你'第二人称，每段最多3句，共情语气。"
        )

        return instructions
