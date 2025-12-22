# save as inspect_missing.py
import pandas as pd

pred_path = "predictions_top10.csv"
sample_path = "data/sample_submission.csv"

pred = pd.read_csv(pred_path, dtype=str).fillna("")
sample = pd.read_csv(sample_path, dtype=str).fillna("")

pred_qs = list(pred['query'].astype(str).str.strip())
sample_qs = list(sample['query'].astype(str).str.strip())

pred_set = set(pred_qs)
missing = [q for q in sample_qs if q not in pred_set]

print(f"Sample rows total: {len(sample_qs)}")
print(f"Pred rows total:   {len(pred_qs)}")
print(f"Missing queries:   {len(missing)}\n")

for q in missing:
    # index in sample
    si = sample_qs.index(q)
    print("="*60)
    print(f"SAMPLE INDEX {si}: {repr(q)}")
    # show a few sample neighbors
    n0 = max(0, si-3)
    n1 = min(len(sample_qs), si+4)
    print("Sample neighbors (index: query):")
    for i in range(n0, n1):
        print(f"  {i}: {repr(sample_qs[i])}")
    # show where the predictions end around the same index
    start_pred = max(0, si-5)
    end_pred = min(len(pred_qs), si+5)
    print("Prediction neighbors around same index (may be unrelated if order differs):")
    for i in range(start_pred, end_pred):
        print(f"  {i}: {repr(pred_qs[i])}")
    print()
