from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str


def get_prompt_templates() -> List[PromptTemplate]:
    return [
        PromptTemplate(
            name="concise_findings",
            template=(
                "You are a medical imaging assistant. Describe chest X-ray findings in 2-4 sentences. "
                "Be cautious and avoid definitive diagnosis. Include: lung fields, opacities, consolidation, pleural effusion, and overall impression. "
                "If image quality is limited, mention limitations."
            ),
        ),
        PromptTemplate(
            name="structured_bullets",
            template=(
                "Write a short radiology-style note using this format:\n"
                "Findings: <1-3 bullet points>\n"
                "Impression: <1 sentence, cautious>\n"
                "Only describe what can be inferred from the image."
            ),
        ),
        PromptTemplate(
            name="label_aware_caution",
            template=(
                "Generate a cautious report for a chest X-ray. "
                "If pneumonia-like patterns are present, mention 'possible infectious/inflammatory change'. "
                "If no focal abnormality is visible, mention 'no clear focal consolidation seen'. "
                "Keep it under 70 words."
            ),
        ),
    ]
