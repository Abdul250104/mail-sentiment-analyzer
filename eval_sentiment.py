
import json
import csv
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from main import analyze_sentiment

DATASET_PATH = Path("datasets/email_golden_dataset.json")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

fields = ["sentiment", "priority", "category"]

expected = {field: [] for field in fields}
predicted = {field: [] for field in fields}
results = []

for index, email in enumerate(dataset, start=1):
    print(f"Evaluating email {index}/{len(dataset)}")

    try:
        prediction = analyze_sentiment(email)

        row = {"id": email["id"], "status": "success"}

        for field in fields:
            actual = prediction[field]
            target = email[f"expected_{field}"]

            expected[field].append(target)
            predicted[field].append(actual)

            row[f"expected_{field}"] = target
            row[f"predicted_{field}"] = actual

        results.append(row)

    except Exception as error:
        print(f"Failed: {error}")
        results.append({
            "id": email["id"],
            "status": "failed",
            "error": str(error)
        })

print("\n===== LLM EVALUATION REPORT =====")

for field in fields:
    if not expected[field]:
        print(f"\n{field}: No successful predictions")
        continue

    print(f"\n--- {field.upper()} ---")

    accuracy = accuracy_score(
        expected[field],
        predicted[field]
    )

    print(f"Accuracy: {accuracy:.2%}")

    print(classification_report(
        expected[field],
        predicted[field],
        zero_division=0
    ))

with open("eval_results.csv", "w",
          newline="", encoding="utf-8") as f:

    columns = ["id", "status", "error"]
    for field in fields:
        columns.extend([
            f"expected_{field}",
            f"predicted_{field}"
        ])

    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to eval_results.csv")
