# validate_query_coverage.py
import pandas as pd

pred_path = "predictions_top10.csv"
sample_path = "data/sample_submission.csv"

pred = pd.read_csv(pred_path, dtype=str).fillna("")
sample = pd.read_csv(sample_path, dtype=str).fillna("")

pred_qs = [q.strip() for q in pred['query'].astype(str)]
sample_qs = [q.strip() for q in sample['query'].astype(str)]

pred_set = set(pred_qs)

missing = [q for q in sample_qs if q not in pred_set]
extra = [q for q in pred_qs if q not in set(sample_qs)]

print("Sample rows:", len(sample_qs))
print("Pred rows:  ", len(pred_qs))
print("Missing queries (count):", len(missing))
if missing:
    print("First missing queries:")
    for m in missing[:50]:
        print(" -", m)
print("Extra queries in predictions (count):", len(extra))
