import json
import os

OUTPUT_PATH = os.path.join("data", "contradiction_matrix.json")


def detect_contradictions(index_path="data/scan_index.json", output_path=OUTPUT_PATH):
    matrix = []
    if not os.path.exists(index_path):
        return matrix
    with open(index_path) as f:
        docs = json.load(f)
    paths = list(docs.keys())
    for i, a in enumerate(paths):
        for b in paths[i + 1 :]:
            if a.split("/")[-1] == b.split("/")[-1]:
                matrix.append({"file_a": a, "file_b": b, "contradiction": "duplicate filename"})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Contradiction matrix written to {output_path}")
    return matrix


if __name__ == "__main__":
    detect_contradictions()
