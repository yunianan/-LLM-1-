"""Text-quality and explicit-answer screens; no changes to medical criteria."""
import re

from shared.disorder_catalog import DISORDERS, SUPPLEMENT_DISORDERS


RULE_VERSION = "mentalbench-quality-v2"
DISORDER_NAMES = sorted({item["name"].casefold() for item in [*DISORDERS.values(), *SUPPLEMENT_DISORDERS.values()]}, key=len, reverse=True)


def profile_text(profile):
    return "\n\n".join(item["value"] for item in profile.get("key_narratives", []) if isinstance(item, dict) and isinstance(item.get("value"), str))


def screen_profile(profile):
    text = profile_text(profile)
    exclude_all, exclude_blind, review = [], [], []
    if not text.strip() or re.fullmatch(r"[\s()\[\]:.\-]*(?:patient\s+vignette|vignette|placeholder|todo)[\s()\[\]:.\-]*", text, re.I):
        exclude_all.append("empty_or_placeholder")
    if re.search(r"[\u3400-\u9fff]", text):
        exclude_all.append("mixed_language_requires_review")
    if len(re.findall(r"[A-Za-z]+", text)) < 30:
        exclude_all.append("short_fragment_requires_review")

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        lower = sentence.casefold()
        if not any(name in lower for name in DISORDER_NAMES):
            continue
        # A prior diagnosis, negated diagnosis, question or differential name
        # is not by itself the answer to this case.
        review.append("diagnosis_name_in_narrative_review_only")
        if re.search(r"\b(previously|historical|prior diagnosis|past diagnosis|history of|was diagnosed|had been diagnosed)\b", lower):
            continue
        if re.search(r"\b(?:does|did|do|has|have|is|are|was|were)\s+not\b|\bnot\s+(?:meet|consistent|fulfill|satisfy)|\brule\s+out\b|\bdenies\b|\bwhether\b", lower):
            continue
        conclusion = re.search(
            r"\b(?:diagnosis|impression|assessment)\s*:|"
            r"\b(?:meets?|fulfills?|satisfies?)\s+(?:\w+\s+){0,6}(?:criteria|requirements)\s+(?:\w+\s+){0,3}for\b|"
            r"\bconsistent\s+with\s+(?:a\s+)?diagnosis\s+of\b|"
            r"\b(?:final|current|primary)\s+diagnosis\s+is\b", lower,
        )
        if conclusion and any(name in lower[conclusion.start():] for name in DISORDER_NAMES):
            exclude_blind.append("explicit_current_diagnosis")
    return {"rule_version": RULE_VERSION,
            "exclude_all": sorted(set(exclude_all)), "exclude_blind": sorted(set(exclude_blind)),
            "review_only": sorted(set(review))}


def exclusion_reasons(screen, part):
    return screen["exclude_all"] + (screen["exclude_blind"] if part in {"dev", "eval"} else [])
