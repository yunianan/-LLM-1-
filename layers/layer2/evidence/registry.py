from __future__ import annotations

import uuid

from layers.layer2.schemas import EvidenceNode, L2SessionState


class EvidenceRegistry:
    @staticmethod
    def add_patient_evidence(
        state: L2SessionState,
        *,
        message_id: str,
        quote: str,
        target_path: str,
        source_type: str = "patient_message",
        parent_ids: list[str] | None = None,
    ) -> EvidenceNode:
        node = EvidenceNode(
            evidence_id=f"L2E-{uuid.uuid4().hex}",
            source_type=source_type,
            message_id=message_id,
            turn_index=state.turn_index,
            quote=quote,
            target_path=target_path,
            created_by=(
                "scale_rule" if source_type == "scale_item"
                else "llm_candidate_rule_verified"
            ),
            parent_ids=parent_ids or [],
        )
        state.evidence_nodes.append(node)
        return node
