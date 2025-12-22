# diagnostics.py
import pandas as pd
from collections import Counter

# load same objects your trainer uses
feats = pd.read_csv('data/features_train.csv', dtype=str)    # features DataFrame produced earlier
train_pairs = pd.read_csv('data/train_query_product_pairs.csv', dtype=str)

# canonicalize
feats['p_id'] = feats['p_id'].astype(str).str.strip().str.replace(r'\.0$','',regex=True)
train_pairs['p_id'] = train_pairs['p_id'].astype(str).str.strip().str.replace(r'\.0$','',regex=True)

print("Features shape:", feats.shape)
print("Train pairs shape:", train_pairs.shape)

# how many train pids appear anywhere in features?
train_pids = set(train_pairs['p_id'].astype(str).tolist())
feat_pids = set(feats['p_id'].astype(str).tolist())
common_pids = train_pids.intersection(feat_pids)
print("unique train pids:", len(train_pids))
print("unique product pids in features:", len(feat_pids))
print("train pids present in features:", len(common_pids))
print("sample missing train pids (up to 20):", list(train_pids - feat_pids)[:20])

# Similarly, inspect matching by query text if available
if 'query' in feats.columns and 'query' in train_pairs.columns:
    feat_queries = set(feats['query'].astype(str).tolist())
    train_queries = set(train_pairs['query'].astype(str).tolist())
    common_q = feat_queries.intersection(train_queries)
    print("unique train queries:", len(train_queries))
    print("unique feat queries:", len(feat_queries))
    print("train queries present in features:", len(common_q))
    print("sample unmatched train queries (up to 10):", list(train_queries - feat_queries)[:10])

# show label distribution prepared by prepare_training_data if available
try:
    from trainer import prepare_training_data
    labeled = prepare_training_data(feats, train_pairs)
    print("Prepared labeled rows:", labeled.shape)
    print("Label counts:\n", labeled['label'].value_counts(dropna=False))
    print("Sample labeled positives (head):\n", labeled[labeled['label']==1].head(10).to_string(index=False))
except Exception as e:
    print("Could not run prepare_training_data (exception):", e)


import pandas as pd
# feats = pd.read_csv('data/features_train.csv', dtype=str)
from trainer import prepare_training_data
# train_pairs = pd.read_csv('data/train_query_product_pairs.csv', dtype=str)
# labeled = prepare_training_data(feats, train_pairs)

# 1) label counts
print("Label counts:\n", labeled['label'].value_counts())

# 2) positives per query distribution
ppq = labeled[labeled['label']==1].groupby('query_id').size().sort_values(ascending=False)
print("Positives per query (summary):")
print(ppq.describe())
print("Top 20 queries by #positives:")
print(ppq.head(20))

# 3) How many queries have at least one positive?
print("Queries with >=1 positive:", (ppq>0).sum(), " / ", labeled['query_id'].nunique())
