from tools.mentalbench_quality import exclusion_reasons, screen_profile


BODY = "The patient describes changes in sleep and energy during the past weeks. Work and relationships have become harder. More history is needed to understand the sequence of events and the clinical context. "


def screen(text):
    return screen_profile({"key_narratives": [{"value": text}]})


def test_disease_name_and_historical_diagnosis_are_not_automatic_leakage():
    for ending in ["She was previously diagnosed with Major Depressive Disorder.",
                   "There is no history of Bipolar I Disorder.",
                   "She does not meet criteria for Major Depressive Disorder.",
                   "She asks whether she has Major Depressive Disorder."]:
        result = screen(BODY + ending)
        assert not exclusion_reasons(result, "dev")
        assert result["review_only"]


def test_explicit_current_answer_is_excluded_from_blind_inputs_but_can_be_kb():
    result = screen(BODY + "Assessment: Major Depressive Disorder.")
    assert exclusion_reasons(result, "dev") == ["explicit_current_diagnosis"]
    assert exclusion_reasons(result, "eval") == ["explicit_current_diagnosis"]
    assert not exclusion_reasons(result, "kb")


def test_corrupt_and_placeholder_records_are_excluded_from_all_three_parts():
    for text in ["(Patient Vignette)", BODY + "Everything feels 腐烂.", "Short fragment only."]:
        result = screen(text)
        assert all(exclusion_reasons(result, part) for part in ["kb", "dev", "eval"])


def test_clinical_language_without_answer_is_not_automatically_excluded():
    assert not exclusion_reasons(screen(BODY + "The clinical course remains unclear and differential assessment is needed."), "eval")
