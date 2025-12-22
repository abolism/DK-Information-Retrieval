import pandas as pd
def normalize_numeric_series_simple(s):
    s = s.astype(str).fillna('').str.strip()
    s = s.str.replace('\u06F0','0').str.replace('\u06F1','1')  # etc, or use full map from code above
    s = s.str.replace(',', '.').str.replace(r'[^0-9\.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce')

feats = pd.read_csv('data/features_train.csv', dtype=str)
cols = [c for c in feats.columns if c not in ['query_id','query','p_id','label']]
sample = feats[cols].head(10).apply(lambda col: normalize_numeric_series_simple(col).isna().sum())
print("NaNs after parse per column (first 30):")
print(sample.head(30))