PLUGIN_PRIORITY = ["IPSRT", "EMDR", "ERP", "CBT-I", "MBCT", "IPT", "ACT", "SFBT"]


class TherapySelector:
    def select_basic(self, diagnosis):
        therapy_table = {
            "MDD": "CBT+BA",
            "GAD": "CBT",
            "PTSD": "CBT-TF",
            "OCD": "CBT",
            "Bipolar": "CBT+IPSRT",
            "Insomnia": "CBT-I",
        }
        return therapy_table.get(diagnosis, "CBT")

    def select_plugins(self, diagnosis, profile):
        plugins = []

        if (
            diagnosis == "Bipolar"
            and profile.get("MDQ", 0) >= 7
            and profile.get("rhythm_disturbance")
        ):
            plugins.append("IPSRT")

        if (
            diagnosis == "PTSD"
            and profile.get("PCL5", 0) >= 33
            and profile.get("exposure_avoidant")
        ):
            plugins.append("EMDR")

        if (
            diagnosis == "OCD"
            and profile.get("Y_BOCS", 0) >= 16
            and profile.get("ritual_behaviors")
        ):
            plugins.append("ERP")

        if (
            diagnosis == "Insomnia"
            and profile.get("ISI", 0) >= 15
            and profile.get("chronic_months", 0) >= 3
        ):
            plugins.append("CBT-I")

        if diagnosis == "MDD" and (
            profile.get("relapse_count", 0) >= 3 or profile.get("RRS", 0) >= 44
        ):
            plugins.append("MBCT")

        if (
            profile.get("interpersonal_conflicts", 0) >= 2
            and profile.get("interpersonal_impairment")
        ):
            plugins.append("IPT")

        if profile.get("AAQ_II", 0) >= 24 or profile.get("PHQ15", 0) >= 10:
            plugins.append("ACT")

        if profile.get("ADNM", 0) >= 47.5 and profile.get("single_focus"):
            plugins.append("SFBT")

        ordered = sorted(
            plugins,
            key=lambda x: PLUGIN_PRIORITY.index(x) if x in PLUGIN_PRIORITY else 99,
        )
        return ordered[:2]

    def contraindications(self, diagnosis, profile):
        cons = []
        if diagnosis == "Bipolar":
            cons.append("双相障碍建议同时就医")
        if diagnosis != "PTSD" and profile.get("PCL5", 0) >= 33:
            cons.append("不推荐EMDR（无PTSD诊断证据）")
        if profile.get("psychotic_symptoms"):
            cons.append("合并精神病性症状→建议精神科就医")
        if profile.get("substance_active"):
            cons.append("物质依赖活跃期→先处理成瘾")
        if profile.get("severe_dissociation"):
            cons.append("严重解离→先做稳定化，不做暴露")
        return cons

    def run(self, diagnosis, profile):
        basic = self.select_basic(diagnosis)
        plugins = self.select_plugins(diagnosis, profile)
        return {
            "basic_therapy": basic,
            "plugins": plugins,
            "contraindications": self.contraindications(diagnosis, profile),
        }
