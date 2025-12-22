# save as fuzzy_lookup.py
import pandas as pd
from difflib import get_close_matches
import unicodedata

def normalize(s):
    if s is None: return ""
    s = unicodedata.normalize("NFC", s).strip()
    return s

pred_path = "predictions_top10.csv"
sample_path = "data/sample_submission.csv"

pred = pd.read_csv(pred_path, dtype=str).fillna("")
sample = pd.read_csv(sample_path, dtype=str).fillna("")

pred_qs = [normalize(q) for q in pred['query'].astype(str)]
uniq_pred_qs = list(dict.fromkeys(pred_qs))  # preserve order, remove duplicates

sample_qs = [normalize(q) for q in sample['query'].astype(str)]

missing = [q for q in sample_qs if q not in set(uniq_pred_qs)]

print("Missing count:", len(missing))
print()

for q in missing:
    close = get_close_matches(q, uniq_pred_qs, n=5, cutoff=0.6)  # cutoff adjustable
    print("-"*60)
    print("MISSING:", repr(q))
    if close:
        print("Possible close matches:")
        for c in close:
            print("   ", repr(c))
    else:
        print("No close match found (cutoff=0.6).")
