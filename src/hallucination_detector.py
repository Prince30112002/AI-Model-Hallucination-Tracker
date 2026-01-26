"""
hallucination_detector.py

Detects potential hallucinations in model responses using:
- Keyword heuristics
- Simple factual consistency checks
- Confidence estimation

Author: Prince
Project: AI Model Hallucination Tracker
"""

import pandas as pd
import re
from typing import Tuple


class HallucinationDetector:
    def __init__(self):
        # Common hallucination indicators
        self.uncertainty_phrases = [
            "i think",
            "it seems",
            "possibly",
            "might be",
            "not sure",
            "approximately",
            "around",
            "estimated"
        ]

    # ---------------------------------
    # Core checks
    # ---------------------------------
    def contains_uncertainty(self, text: str) -> bool:
        text = text.lower()
        return any(phrase in text for phrase in self.uncertainty_phrases)

    def has_numeric_claim(self, text: str) -> bool:
        """
        Detects numbers or dates (often hallucinated)
        """
        return bool(re.search(r"\d", text))

    # ---------------------------------
    # Scoring logic
    # ---------------------------------
    def score_response(self, response_text: str) -> Tuple[int, float, str]:
        """
        Returns:
        - hallucination_flag (0 or 1)
        - confidence_score (0–1)
        - final_label
        """

        response_text = str(response_text).strip().lower()

        hallucination_score = 0.0

        if self.contains_uncertainty(response_text):
            hallucination_score += 0.4

        if self.has_numeric_claim(response_text):
            hallucination_score += 0.2

        # Length-based heuristic
        if len(response_text) < 30:
            hallucination_score += 0.3

        # Clamp score
        hallucination_score = min(hallucination_score, 1.0)

        confidence_score = round(1 - hallucination_score, 2)

        if hallucination_score >= 0.6:
            return 1, confidence_score, "hallucinated"
        elif hallucination_score >= 0.4:
            return 0, confidence_score, "uncertain"
        else:
            return 0, confidence_score, "accurate"

    # ---------------------------------
    # Batch processing
    # ---------------------------------
    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects column: response_text
        """
        if "response_text" not in df.columns:
            raise ValueError("DataFrame must contain 'response_text' column")

        results = df.copy()

        outputs = results["response_text"].apply(self.score_response)

        results["hallucination_flag"] = outputs.apply(lambda x: x[0])
        results["confidence_score"] = outputs.apply(lambda x: x[1])
        results["final_label"] = outputs.apply(lambda x: x[2])

        return results


# ---------------------------------
# Quick test
# ---------------------------------
if __name__ == "__main__":
    detector = HallucinationDetector()

    sample_text = "I think the capital of Australia might be Sydney."
    flag, confidence, label = detector.score_response(sample_text)

    print("Hallucination Flag:", flag)
    print("Confidence Score:", confidence)
    print("Final Label:", label)
