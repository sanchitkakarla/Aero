"""Runs the pipeline against the labeled calls and reports per-field
accuracy plus a confusion matrix for emotional_tone.

Only 3 labeled examples exist, so this is a smoke test, not a real
validation study -- treat the numbers as directional. Real validation
happens against AutoAce's hidden test set.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import pipeline


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

FIELDS = [
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
]


def load_labels(csv_path):
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = json.loads(row["result_json"])
    return rows


def run(data_dir, csv_path):
    labels = load_labels(csv_path)
    per_field_correct = {f: 0 for f in FIELDS}
    total = 0
    confusion = {}
    rows_out = []

    for name, expected in labels.items():
        audio_path = Path(data_dir) / name
        if not audio_path.exists():
            print(f"skipping {name}: file not found at {audio_path}")
            continue

        predicted, elapsed = pipeline.process(str(audio_path))
        total += 1

        exp_tone = expected["emotional_tone"]
        pred_tone = predicted["emotional_tone"]
        confusion.setdefault(exp_tone, {}).setdefault(pred_tone, 0)
        confusion[exp_tone][pred_tone] += 1

        row = {"name": name, "latency_sec": round(elapsed, 2)}
        for f in FIELDS:
            match = predicted[f] == expected[f]
            per_field_correct[f] += int(match)
            row[f] = {"expected": expected[f], "predicted": predicted[f], "match": match}
        rows_out.append(row)

    print(f"\n{'field':<28} accuracy")
    for f in FIELDS:
        acc = per_field_correct[f] / total if total else 0
        print(f"{f:<28} {acc:.0%} ({per_field_correct[f]}/{total})")

    print("\nconfusion matrix (emotional_tone) -- rows=actual, cols=predicted")
    for actual, preds in confusion.items():
        print(f"  {actual}: {preds}")

    return rows_out


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    csv_path = sys.argv[2] if len(sys.argv) > 2 else "data/labels.csv"
    results = run(data_dir, csv_path)
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print("\nfull results written to validation_results.json")
