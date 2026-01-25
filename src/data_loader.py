"""
data_loader.py

Responsible for:
- Loading raw question data
- Loading model responses
- Basic validation & cleaning
- Saving processed datasets

Author: Prince
Project: AI Model Hallucination Tracker
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"

        # Ensure folders exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Loaders
    # -----------------------------
    def load_questions(self, filename: str = "questions.csv") -> pd.DataFrame:
        file_path = self.raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found in data/raw/")
        return pd.read_csv(file_path)

    def load_model_responses(
        self, filename: str = "model_responses_raw.csv"
    ) -> pd.DataFrame:
        file_path = self.raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} not found in data/raw/")
        return pd.read_csv(file_path)

    # -----------------------------
    # Cleaning
    # -----------------------------
    def clean_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning:
        - Remove empty responses
        - Strip whitespace
        - Normalize text
        """
        df = df.copy()

        if "response_text" not in df.columns:
            raise ValueError("Column 'response_text' is required")

        df["response_text"] = (
            df["response_text"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        df = df[df["response_text"].str.len() > 10]
        df.reset_index(drop=True, inplace=True)

        return df

    # -----------------------------
    # Save processed data
    # -----------------------------
    def save_processed(
        self, df: pd.DataFrame, filename: str = "cleaned_responses.csv"
    ) -> None:
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"✅ Saved processed data to {output_path}")


# -----------------------------
# Quick test (optional)
# -----------------------------
if __name__ == "__main__":
    loader = DataLoader()
    print("📂 DataLoader initialized successfully")
