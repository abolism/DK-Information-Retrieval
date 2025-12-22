# export_results.py
"""
Export top-10 predicted product ids per test query using your existing project modules
(data_loader, normalizer, data_preprocessing.Preprocessor).

Run:
    python export_results.py

Outputs:
 - data/features_test_generated.csv   (optional inspection)
 - predictions_top10.csv              (final requested format)
"""

import os, sys, math, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# project modules (you implemented these)
from normalizer import PersianNormalizer
from data_loader import load_products, load_train_pairs, load_test_queries
from data_preprocessing import Preprocessor

# sklearn helpers
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import lightgbm for model; fallback to sklearn pickle
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
from sklearn.linear_model import LogisticRegression

# -------- CONFIG --------
TEST_QUERIES_CSV = "data/test_queries.csv"
PRODUCTS_CSV = "data/products.csv"
PRODUCTS_CLEAN_PKL = "data/products_cleaned.pkl"
FEATURES_OUT = "data/features_test_generated.csv"
OUTPUT_CSV = "predictions_top10.csv"
MODEL_LGB_PATH = "models/lgb_ranker.txt"
MODEL_SKL_PATH = "models/lr_classifier.pkl"
CANDIDATES_PER_QUERY = 50
TOPK = 10
# ------------------------

def canonicalize_pid_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r'\.0$','', regex=True)

def simple_tokenize(text: str):
    if text is None: return []
    return [t for t in str(text).lower().split() if t.strip()]

def difflib_ratio(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

# Minimal BM25 (same helper as before)
class BM25:
    def __init__(self, docs, tokenizer=lambda x: x.split(), k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.k1 = k1; self.b = b
        self.docs = [tokenizer(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1,self.N)
        self.df = {}
        self.f = []
        for d in self.docs:
            freq = {}
            for w in d:
                freq[w] = freq.get(w,0) + 1
            self.f.append(freq)
            for w in freq.keys():
                self.df[w] = self.df.get(w,0) + 1
        self.idf = {}
        for w,freq in self.df.items():
            self.idf[w] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
    def get_score(self, query, index):
        q = self.tokenizer(query)
        score = 0.0
        doc_freq = self.f[index]
        dl = len(self.docs[index])
        for term in q:
            if term not in doc_freq:
                continue
            tf = doc_freq[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * tf * (self.k1 + 1) / denom
        return score
    def get_scores(self, query):
        return [self.get_score(query, i) for i in range(self.N)]

def load_model():
    if HAVE_LGB and os.path.exists(MODEL_LGB_PATH):
        try:
            return lgb.Booster(model_file=MODEL_LGB_PATH), 'lgb'
        except Exception as e:
            print("Warning: failed to load LGBM model:", e)
    if os.path.exists(MODEL_SKL_PATH):
        with open(MODEL_SKL_PATH,'rb') as f:
            return pickle.load(f), 'sklearn'
    raise FileNotFoundError("Model not found: put models/lgb_ranker.txt or models/lr_classifier.pkl")

def normalize_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna('').str.strip()
    persian_map = {
        '\u06F0':'0','\u06F1':'1','\u06F2':'2','\u06F3':'3','\u06F4':'4',
        '\u06F5':'5','\u06F6':'6','\u06F7':'7','\u06F8':'8','\u06F9':'9',
        '\u0660':'0','\u0661':'1','\u0662':'2','\u0663':'3','\u0664':'4',
        '\u0665':'5','\u0666':'6','\u0667':'7','\u0668':'8','\u0669':'9'
    }
    for k,v in persian_map.items():
        s = s.str.replace(k, v)
    s = s.str.replace(r'[,]\s*(?=\d{1,3}(\D|$))', '', regex=True)
    s = s.str.replace(',', '.')
    s = s.str.replace(r'[^0-9\.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def main():
    # init normalizer+preprocessor from your project code
    normalizer = PersianNormalizer()
    pre = Preprocessor(normalizer=normalizer, map_digits=True, max_text_chars=1024)

    # load test queries
    if not os.path.exists(TEST_QUERIES_CSV):
        print("Missing test queries:", TEST_QUERIES_CSV); sys.exit(1)
    test_q = pd.read_csv(TEST_QUERIES_CSV, dtype=str)
    if 'query_id' not in test_q.columns or 'query' not in test_q.columns:
        print("test_queries.csv must contain query_id,query"); sys.exit(1)
    test_q['query'] = test_q['query'].astype(str).str.strip()

    # load (or build) products cleaned using your Preprocessor
    products = None
    if os.path.exists(PRODUCTS_CLEAN_PKL):
        print("Loading cleaned products from", PRODUCTS_CLEAN_PKL)
        products = pd.read_pickle(PRODUCTS_CLEAN_PKL)
    else:
        # try using your data_loader if available
        try:
            print("Attempting to load products via data_loader.load_products ...")
            products = load_products(PRODUCTS_CSV, normalizer)  # your loader may normalize already
        except Exception:
            print("data_loader.load_products not available or failed; falling back to raw CSV:", PRODUCTS_CSV)
            products = pd.read_csv(PRODUCTS_CSV, dtype=str)
        # run your Preprocessor clean to produce text_norm + attributes, etc.
        print("Cleaning products with Preprocessor.clean_products_df ...")
        products = pre.clean_products_df(products)

    # canonicalize p_id and ensure text_norm exists
    if 'p_id' not in products.columns:
        products['p_id'] = products.index.astype(str)
    products['p_id'] = canonicalize_pid_col(products['p_id'])
    if 'text_norm' not in products.columns:
        products['text_norm'] = products.get('title','').fillna('').astype(str) + ' ' + products.get('attributes','').fillna('').astype(str)
        products['text_norm'] = products['text_norm'].str.strip()
    prod_texts = products['text_norm'].fillna('').astype(str).tolist()
    prod_ids = products['p_id'].astype(str).tolist()

    # TF-IDF vectorizer fit on product texts
    print("Fitting TF-IDF on product texts (this may take a while)...")
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    prod_tfidf = tfidf.fit_transform(prod_texts)

    # BM25 index
    bm25 = BM25(prod_texts, tokenizer=lambda s: simple_tokenize(s))

    # candidate generation per query (TF-IDF cosine + BM25 combination)
    rows = []
    q_to_candidates = {}
    for _, qr in test_q.iterrows():
        qid = str(qr['query_id']); qtext = str(qr['query'])
        qvec = tfidf.transform([qtext])
        sims = (qvec.dot(prod_tfidf.T)).toarray().ravel()
        bm_scores = bm25.get_scores(qtext)
        bm_arr = np.array(bm_scores)
        if bm_arr.max() - bm_arr.min() > 0:
            bm_norm = (bm_arr - bm_arr.min()) / (bm_arr.max() - bm_arr.min())
        else:
            bm_norm = bm_arr
        combined = 0.6 * sims + 0.4 * bm_norm
        top_idx = list(np.argsort(combined)[::-1][:CANDIDATES_PER_QUERY])
        q_to_candidates[qid] = top_idx
        for idx in top_idx:
            rows.append({
                'query_id': qid,
                'query': qtext,
                'p_id': prod_ids[idx],
                'product_text': prod_texts[idx]
            })

    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        print("No candidates generated; aborting."); sys.exit(1)

    # compute features consistent with your training features
    feat_rows = []
    # precompute token sets & attributes list
    prod_tokens_cache = {str(r['p_id']): simple_tokenize(r['text_norm']) for _, r in products[['p_id','text_norm']].iterrows()}
    # train product freq if you have train pairs file
    train_freq = {}
    if os.path.exists("data/train_query_product_pairs.csv"):
        try:
            tp = pd.read_csv("data/train_query_product_pairs.csv", dtype=str)
            tp['p_id'] = canonicalize_pid_col(tp['p_id'])
            train_freq = tp['p_id'].value_counts().to_dict()
        except Exception:
            train_freq = {}

    for _, r in cand_df.iterrows():
        qid = r['query_id']; qtext = r['query']; pid = r['p_id']; ptext = r['product_text']
        # tfidf cosine
        qv = tfidf.transform([qtext])
        try:
            pidx = prod_ids.index(pid)
            sim = (qv.dot(prod_tfidf[pidx].T)).toarray().ravel()[0]
            bm = bm25.get_score(qtext, pidx)
        except ValueError:
            sim = float((qv.dot(tfidf.transform([ptext]).T)).toarray().ravel()[0])
            bm = bm25.get_score(qtext, 0)

        q_tokens = simple_tokenize(qtext); p_tokens = prod_tokens_cache.get(pid, simple_tokenize(ptext))
        set_q = set(q_tokens); set_p = set(p_tokens)
        jacc = (len(set_q & set_p) / len(set_q | set_p)) if (len(set_q | set_p) > 0) else 0.0
        overlap_count = len(set_q & set_p)
        query_len = len(q_tokens); prod_len = len(p_tokens)
        title_fuzzy = difflib_ratio(qtext, ptext)

        # try to compute brand/category exact if products df has cleaned fields
        brand_exact = 0
        if 'brand_norm' in products.columns:
            # find product row quickly
            try:
                prow = products.loc[products['p_id']==pid].iloc[0]
                b = str(prow.get('brand_norm','')).strip()
                if b and any(tok == b for tok in q_tokens): brand_exact = 1
            except Exception:
                brand_exact = 0
        category_exact = 0
        if 'category' in products.columns:
            try:
                prow = products.loc[products['p_id']==pid].iloc[0]
                c = str(prow.get('category','')).strip()
                if c and any(tok == c for tok in q_tokens): category_exact = 1
            except Exception:
                category_exact = 0

        # attributes overlap (if attributes_list present)
        attr_overlap_count = 0
        attr_overlap_ratio = 0.0
        if 'attributes_list' in products.columns:
            try:
                prow = products.loc[products['p_id']==pid].iloc[0]
                alist = prow.get('attributes_list', []) or []
                # tokenized attributes simple approach
                a_tokens = set()
                for a in alist:
                    a_tokens.update(simple_tokenize(a))
                if len(a_tokens) > 0:
                    inter = len(set_q & a_tokens)
                    attr_overlap_count = inter
                    attr_overlap_ratio = inter / len(a_tokens) if len(a_tokens) > 0 else 0.0
            except Exception:
                pass

        # digits overlap
        q_digits = set([c for c in qtext if c.isdigit()])
        p_digits = set([c for c in ptext if c.isdigit()])
        digit_overlap_ratio = (len(q_digits & p_digits) / len(q_digits)) if len(q_digits)>0 else 0.0
        num_query_digits = len(q_digits)

        # placeholders for model_token_match / num_q_model_tokens (if you used them in training, replace with your logic)
        model_token_match = 0
        num_q_model_tokens = 0

        train_product_freq = int(train_freq.get(pid, 0))

        feat_rows.append({
            'query_id': qid, 'query': qtext, 'p_id': pid,
            'tfidf_cosine': float(sim), 'bm25': float(bm),
            'jaccard_tokens': float(jacc), 'overlap_count': int(overlap_count),
            'query_len': int(query_len), 'prod_len': int(prod_len),
            'title_fuzzy': float(title_fuzzy),
            'brand_exact': int(brand_exact), 'category_exact': int(category_exact),
            'attr_overlap_count': int(attr_overlap_count), 'attr_overlap_ratio': float(attr_overlap_ratio),
            'digit_overlap_ratio': float(digit_overlap_ratio), 'num_query_digits': int(num_query_digits),
            'model_token_match': int(model_token_match), 'num_q_model_tokens': int(num_q_model_tokens),
            'train_product_freq': int(train_product_freq)
        })

    feats_df = pd.DataFrame(feat_rows)
    # save features for inspection
    os.makedirs(os.path.dirname(FEATURES_OUT) or ".", exist_ok=True)
    feats_df.to_csv(FEATURES_OUT, index=False, encoding='utf-8')
    print("Saved features to", FEATURES_OUT, "rows:", len(feats_df))

    # normalize numeric columns
    feat_cols = [c for c in feats_df.columns if c not in ['query_id','query','p_id']]
    for c in feat_cols:
        feats_df[c] = normalize_numeric_series(feats_df[c])

    # load model and predict
    model, mtype = load_model()
    print("Loaded model type:", mtype)
    X = feats_df[feat_cols].values
    if mtype == 'lgb' and HAVE_LGB and isinstance(model, lgb.Booster):
        try:
            scores = model.predict(X, num_iteration=getattr(model, 'best_iteration', None))
        except Exception:
            scores = model.predict(X)
    else:
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X)[:,1]
        else:
            scores = model.predict(X).astype(float)

    feats_df['__score'] = scores

    # gather top-K per query using the full candidate set used in val
    preds = {}
    for qid, g in feats_df.groupby('query_id'):
        ordered = g.sort_values('__score', ascending=False)
        top_pids = ordered['p_id'].astype(str).tolist()[:TOPK]
        top_pids += [''] * max(0, TOPK - len(top_pids))
        preds[str(qid)] = top_pids

    # write final CSV with query text + pid1..pid10
    out_rows = []
    for _, qr in test_q.iterrows():
        qid = str(qr['query_id']); qtext = str(qr['query'])
        pids = preds.get(qid, ['']*TOPK)
        out_rows.append([qtext] + pids)

    out_cols = ['query'] + [f'pid{i+1}' for i in range(TOPK)]
    out_df = pd.DataFrame(out_rows, columns=out_cols)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print("Wrote top-{} predictions to: {}".format(TOPK, OUTPUT_CSV))
    print(out_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
