import json
import os

OUTPUT_DIR = os.path.join("F:/", "LegalResults", "JUDGE_PREDICTIONS")

OUTCOME_MATRIX = [
    {"outcome": "Denial of Injunction, No ruling on fraud", "likelihood": 42},
    {"outcome": "Partial Injunction", "likelihood": 33},
    {"outcome": "Full Injunction + Discovery", "likelihood": 17},
    {"outcome": "Rejection on Technical Grounds", "likelihood": 8},
]


def build_prediction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "aug12_outcome_matrix.json")
    with open(path, "w") as f:
        json.dump(OUTCOME_MATRIX, f, indent=2)
    print(f"Outcome matrix saved to {path}")


if __name__ == "__main__":
    build_prediction()
