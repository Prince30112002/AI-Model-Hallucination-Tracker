"""
scoring.py

Combines hallucination detection outputs into:
- Final risk score
- Model-level summary metrics

Author: Prince
Project: AI Model Hallucination Tracker
"""

import pandas as pd


class HallucinationScorer:
    def __init__(self):
        pass

    # ---------------------------------
    # Row-level scoring
    # ---------------------------------
    def compute_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects columns:
        - hallucination_flag
        - confidence_score
        - final_label
        """

        required_cols = {
            "hallucination_flag",
            "confidence_score",
            "final_label",
        }

        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}"
            )

        scored_df = df.copy()

        # Risk score: higher = worse
        scored_df["hallucination_risk_score"] = (
            (1 - scored_df["confidence_score"])
            + scored_df["hallucination_flag"] * 0.5
        ).round(2)

        return scored_df

    # ---------------------------------
    # Model-level summary
    # ---------------------------------
    def generate_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates aggregated metrics per model
        """

        if "model_name" not in df.columns:
            raise ValueError("Column 'model_name' is required")

        summary = (
            df.groupby("model_name")
            .agg(
                total_responses=("final_label", "count"),
                hallucinated_count=(
                    "final_label",
                    lambda x: (x == "hallucinated").sum(),
                ),
                uncertain_count=(
                    "final_label",
                    lambda x: (x == "uncertain").sum(),
                ),
                avg_confidence_score=("confidence_score", "mean"),
                avg_risk_score=("hallucination_risk_score", "mean"),
            )
            .reset_index()
        )

        summary["hallucination_rate"] = (
            summary["hallucinated_count"] / summary["total_responses"]
        ).round(2)

        return summary


# ---------------------------------
# Quick test
# ---------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "model_name": ["Model-A", "Model-A", "Model-B"],
            "hallucination_flag": [0, 1, 0],
            "confidence_score": [0.9, 0.4, 0.8],
            "final_label": ["accurate", "hallucinated", "uncertain"],
        }
    )

    scorer = HallucinationScorer()

    scored = scorer.compute_final_score(sample_data)
    print("🔹 Scored Data")
    print(scored)

    summary = scorer.generate_model_summary(scored)
    print("\n🔹 Model Summary")
    print(summary)

