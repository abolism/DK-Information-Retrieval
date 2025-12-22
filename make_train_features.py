# make_train_features.py
import pandas as pd
from normalizer import PersianNormalizer
from data_preprocessing import Preprocessor
from data_loader import load_train_pairs
from feature_engineering import FeatureEngineer

# 1) init
normalizer = PersianNormalizer()
pre = Preprocessor(normalizer=normalizer, map_digits=True, max_text_chars=1024)

# 2) load products cleaned (fast) or raw + clean
# prefer using cleaned pickle if you have it:
try:
    products = pd.read_pickle('data/products_clean_full.pkl')
except Exception:
    products_raw = pd.read_csv('data/products.csv', dtype=str)
    products = pre.clean_products_df(products_raw)

# 3) load train pairs and build unique train queries DataFrame
train_pairs = load_train_pairs('data/train_query_product_pairs.csv')
# keep only unique textual queries and assign query_id
train_queries_uniq = pd.DataFrame({
    'query': train_pairs['query'].astype(str)
}).drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'query_id'})
# normalize queries using same preprocessor
train_queries_uniq['query_norm'] = train_queries_uniq['query'].map(lambda s: pre.normalize_text(s) if pd.notna(s) else '')

print("Unique train queries:", len(train_queries_uniq))

# 4) fit FeatureEngineer on product corpus
fe = FeatureEngineer(top_k=50, tfidf_max_features=75000, ngram_range=(1,2))
fe.fit(products, prod_id_col='p_id', text_col='text_norm',
       title_col='title' if 'title' in products.columns else None,
       brand_col='brand' if 'brand' in products.columns else None,
       attr_col='attributes_list' if 'attributes_list' in products.columns else None)

# optional: feed product popularity from train pairs
fe.fit_train_counts(train_pairs)

# 5) get candidates for the train queries
cands_train = fe.get_candidates_for_queries(train_queries_uniq, query_col='query_norm', id_col='query_id')
print("Candidates rows:", len(cands_train))

# 6) compute features for train candidates
features_train = fe.transform(cands_train)
# attach original textual query to features for mapping ease
# # map query_id -> query text
# qid2q = dict(zip(train_queries_uniq['query_id'].astype(str), train_queries_uniq['query']))
# features_train['query'] = features_train['query_id'].map(lambda qid: qid2q.get(int(qid) if isinstance(qid, (int,float)) else int(str(qid)), ''))
# ensure both sides use string keys
qid2q = {str(qid): q for qid, q in zip(train_queries_uniq['query_id'].astype(str).tolist(),
                                       train_queries_uniq['query'].tolist())}

# Map robustly: convert features' query_id to string and map; keep original query_norm if needed
features_train['query_id'] = features_train['query_id'].astype(str)
features_train['query'] = features_train['query_id'].map(qid2q)
# Some rows might still be missing (rare); fill them with empty string or their normalized form
missing_q = features_train['query'].isna().sum()
if missing_q > 0:
    # fallback: try to fill from query column present in candidates_df (sometimes fe.transform sets it)
    if 'query' in cands_train.columns:
        # cands_train had textual (normalized) query values
        qmap_from_cands = dict(zip(cands_train['query_id'].astype(str), cands_train['query']))
        features_train['query'] = features_train['query'].fillna(features_train['query_id'].map(qmap_from_cands))
    features_train['query'] = features_train['query'].fillna('')  # final safety fill
print("Mapped query_id -> query texts. Missing after fill:", features_train['query'].isna().sum())


# 7) save
features_train.to_csv('data/features_train.csv', index=False)
print("Saved features_train.csv:", features_train.shape)
