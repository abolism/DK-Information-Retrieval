# save as fill_with_popular.py
import pandas as pd
from collections import Counter
import csv

pred_path = "predictions_top10.csv"
sample_path = "data/sample_submission.csv"
out_path = "predictions_aligned_fallback.csv"

pred = pd.read_csv(pred_path, dtype=str).fillna("")
sample = pd.read_csv(sample_path, dtype=str).fillna("")

# compute top popular pids across your predictions (flatten pid1..pid10)
pid_cols = [f"pid{i}" for i in range(1,11)]
all_pids = []
for _, row in pred.iterrows():
    for c in pid_cols:
        v = str(row.get(c,"")).strip()
        if v:
            all_pids.append(v)
popular = [pid for pid, cnt in Counter(all_pids).most_common(10)]
if len(popular) < 10:
    popular = popular + [""]*(10-len(popular))

print("Top-10 popular pids used for fallback:", popular)

# build quick lookup map of pred by normalized query
pred_map = {str(r['query']).strip(): r for _, r in pred.iterrows()}

rows_out = []
for _, srow in sample.iterrows():
    q = str(srow['query']).strip()
    if q in pred_map:
        rows_out.append([q] + [pred_map[q].get(c,"") for c in pid_cols])
    else:
        rows_out.append([q] + popular)

# write CSV
cols = ["query"] + pid_cols
import csv
with open(out_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    for r in rows_out:
        writer.writerow(r)

print("Wrote:", out_path)
print("Upload predictions_aligned_fallback.csv to the platform to test whether the low score was due to missing rows.")
