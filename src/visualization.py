"""
visualization.py

Creates visual insights for:
- Hallucination rate by model
- Confidence score distribution
- Risk score comparison

Author: Prince
Project: AI Model Hallucination Tracker
"""

# -------------------------------
# PATH FIX (IMPORTANT)
# -------------------------------
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# -------------------------------
# Imports
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.hallucination_detector import HallucinationDetector
from src.scoring import HallucinationScorer


class HallucinationVisualizer:
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------
    # Hallucination rate per model
    # ---------------------------------
    def plot_hallucination_rate(self, summary_df: pd.DataFrame):
        plt.figure()
        plt.bar(
            summary_df["model_name"],
            summary_df["hallucination_rate"],
        )
        plt.xlabel("Model")
        plt.ylabel("Hallucination Rate")
        plt.title("Hallucination Rate by Model")
        plt.tight_layout()

        output_path = self.output_dir / "hallucination_rate.png"
        plt.savefig(output_path)
        plt.close()

        print(f"📊 Saved: {output_path}")

    # ---------------------------------
    # Confidence score distribution
    # ---------------------------------
    def plot_confidence_distribution(self, scored_df: pd.DataFrame):
        plt.figure()
        plt.hist(scored_df["confidence_score"], bins=10)
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Confidence Score Distribution")
        plt.tight_layout()

        output_path = self.output_dir / "confidence_distribution.png"
        plt.savefig(output_path)
        plt.close()

        print(f"📊 Saved: {output_path}")

    # ---------------------------------
    # Risk score comparison
    # ---------------------------------
    def plot_risk_score_by_label(self, scored_df: pd.DataFrame):
        plt.figure()

        labels = scored_df["final_label"].unique()
        data = [
            scored_df[scored_df["final_label"] == label][
                "hallucination_risk_score"
            ]
            for label in labels
        ]

        plt.boxplot(data, labels=labels)
        plt.xlabel("Final Label")
        plt.ylabel("Hallucination Risk Score")
        plt.title("Risk Score by Prediction Label")
        plt.tight_layout()

        output_path = self.output_dir / "risk_score_by_label.png"
        plt.savefig(output_path)
        plt.close()

        print(f"📊 Saved: {output_path}")


# ---------------------------------
# MAIN EXECUTION
# ---------------------------------
if __name__ == "__main__":
    loader = DataLoader()
    detector = HallucinationDetector()
    scorer = HallucinationScorer()

    # Load & process data
    df = loader.load_model_responses()
    df = loader.clean_responses(df)
    df = detector.analyze_dataframe(df)
    df = scorer.compute_final_score(df)

    summary_df = scorer.generate_model_summary(df)

    # Visualize
    viz = HallucinationVisualizer()
    viz.plot_hallucination_rate(summary_df)
    viz.plot_confidence_distribution(df)
    viz.plot_risk_score_by_label(df)
