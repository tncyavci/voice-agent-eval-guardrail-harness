from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set

_TAG_GRAMMAR_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")

# Closed-vocabulary registry of canonical `namespace.concept` tags
CLOSED_VOCABULARY: Set[str] = {
    "incident.root_cause_isolation",
    "incident.blast_radius_containment",
    "incident.rollback_verification",
    "incident.stakeholder_communication",
    "safety.lockout_tagout_verification",
    "safety.zero_energy_check",
    "safety.ppe_compliance",
    "safety.emergency_escalation",
    "clinical.airway_assessment",
    "clinical.hemodynamic_monitoring",
    "clinical.medication_contraindication",
    "clinical.informed_consent",
}

TAG_ALIASES: Dict[str, str] = {
    "rca": "incident.root_cause_isolation",
    "root_cause": "incident.root_cause_isolation",
    "blast_radius": "incident.blast_radius_containment",
    "rollback": "incident.rollback_verification",
    "loto": "safety.lockout_tagout_verification",
    "zero_energy": "safety.zero_energy_check",
    "airway": "clinical.airway_assessment",
    "contraindication": "clinical.medication_contraindication",
}


def normalize_concept_tag(raw_tag: str) -> Optional[str]:
    """
    Normalizes an untrusted tag from an LLM extractor into a canonical
    closed-vocabulary `namespace.concept` identifier. Returns None rather
    than raising if the tag is outside the closed vocabulary.
    """
    cleaned = raw_tag.strip().lower()
    if cleaned in TAG_ALIASES:
        cleaned = TAG_ALIASES[cleaned]
    if not _TAG_GRAMMAR_RE.match(cleaned):
        return None
    if cleaned not in CLOSED_VOCABULARY:
        return None
    return cleaned


def filter_valid_tags(raw_tags: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    valid: List[str] = []
    for raw in raw_tags:
        norm = normalize_concept_tag(raw)
        if norm and norm not in seen:
            seen.add(norm)
            valid.append(norm)
    return valid
