# feature_engineering.py
"""
Feature engineering utilities for e-commerce IR.

Provides:
 - FeatureEngineer: fit on a product corpus and produce features for query-candidate pairs.
 - BM25 implementation (okapi) for scoring queries->documents.
 - Candidate generation via TF-IDF top-k.
 - Many explainable features:
     * TF-IDF cosine similarity
     * BM25 score
     * token Jaccard / overlap
     * exact brand / category match
     * attributes overlap count / ratio
     * digit/model matching features
     * query/product length features
     * fuzzy title similarity (difflib.SequenceMatcher)
 - All functions expect preprocessed text (use text_norm / query_norm from your Preprocessor)

Dependencies:
  numpy, pandas, scikit-learn

Usage:
  from feature_engineering import FeatureEngineer
  fe = FeatureEngineer(top_k=100)
  fe.fit(products_df, prod_id_col='p_id', text_col='text_norm')
  # build candidates (uses TF-IDF top_k retrieval on products)
  candidates_df = fe.get_candidates_for_queries(tests_df, query_col='query_norm')
  # compute features for candidate pairs
  features_df = fe.transform(candidates_df, queries_df=tests_df)
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# -------------------------
# Tokenizer (conservative)
# -------------------------
_WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)

def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return _WORD_RE.findall(str(text))


def extract_digits(text: str) -> List[str]:
    """Return contiguous digit tokens found in text (ASCII or persian digits)."""
    if text is None:
        return []
    return re.findall(r"\d+", str(text))


def jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / len(union) if union else 0.0


# -------------------------
# Small BM25 implementation (vectorized)
# -------------------------
class BM25:
    """
    Simple BM25 implemented with CountVectorizer internals for speed.
    After fit(corpus) you can call score_queries(queries) to compute BM25 scores against all docs.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.tf = None             # document-term matrix (csr)
        self.df = None             # document frequency per term
        self.idf = None            # idf per term
        self.doc_len = None        # length per document
        self.avgdl = None
        self.N = 0

    def fit(self, corpus: List[str], min_df: int = 1, token_pattern: str = r"(?u)\b\w+\b"):
        """
        Fit BM25 to a list of documents (strings).
        """
        self.vectorizer = CountVectorizer(token_pattern=token_pattern, min_df=min_df)
        self.tf = self.vectorizer.fit_transform(corpus)  # shape (N_docs, V)
        self.N = self.tf.shape[0]
        # document frequencies
        self.df = np.bincount(self.tf.indices, minlength=self.tf.shape[1])
        # safe idf: standard BM25 idf
        # idf = log((N - df + 0.5) / (df + 0.5))
        self.idf = np.log((self.N - self.df + 0.5) / (self.df + 0.5) + 1e-9)
        self.doc_len = np.array(self.tf.sum(axis=1)).ravel()
        self.avgdl = float(self.doc_len.mean()) if self.N > 0 else 0.0

    def _tf_query(self, query: str):
        # transform query using same vocabulary
        if self.vectorizer is None:
            raise ValueError("BM25 not fitted")
        q_tf = self.vectorizer.transform([query])  # shape (1, V)
        return q_tf

    def score(self, query: str, topk: Optional[int] = None) -> np.ndarray:
        """
        Return BM25 score array for all documents for a single query.
        If topk specified, returns only topk scores (and corresponding doc indices).
        """
        q_tf = self._tf_query(query)  # 1 x V
        q_indices = q_tf.indices
        if q_indices.size == 0:
            # empty query => zeros
            if topk:
                return np.array([]), np.array([])
            return np.zeros(self.N, dtype=float)

        # only compute on terms present in query for speed
        # tf_doc_col = self.tf[:, q_index].toarray().ravel()
        scores = np.zeros(self.N, dtype=float)
        k1 = self.k1
        b = self.b
        for term_idx in q_indices:
            idf_t = self.idf[term_idx]
            tf_col = self.tf[:, term_idx].toarray().ravel()  # tf for this term in all docs
            denom = tf_col + k1 * (1 - b + b * (self.doc_len / self.avgdl))
            numer = tf_col * (k1 + 1)
            scores += idf_t * (numer / (denom + 1e-12))
        if topk:
            # return topk indices and scores sorted
            topk = min(topk, len(scores))
            idx = np.argpartition(-scores, topk - 1)[:topk]
            idx_sorted = idx[np.argsort(-scores[idx])]
            return idx_sorted, scores[idx_sorted]
        return scores


# -------------------------
# FeatureEngineer
# -------------------------
class FeatureEngineer:
    """
    Fitable feature engineer for IR.

    Primary usage:
      fe = FeatureEngineer(top_k=100)
      fe.fit(products_df, prod_id_col='p_id', text_col='text_norm', title_col='title', brand_col='brand', attr_col='attributes_list')
      candidates_df = fe.get_candidates_for_queries(tests_df, query_col='query_norm')  # returns DataFrame with columns ['query_id','query','pid']
      features_df = fe.transform(candidates_df, queries_df=tests_df)

    Output features_df columns include:
      query_id, query, p_id,
      tfidf_cosine, bm25, jaccard_tokens, title_fuzzy, brand_exact, category_exact,
      attr_overlap_count, attr_overlap_ratio, digit_overlap_ratio, query_len, prod_len, ...
    """

    def __init__(self, top_k: int = 100, tfidf_max_features: int = 75000, ngram_range: Tuple[int,int]=(1,2)):
        self.top_k = top_k
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range

        # Will be initialized in fit
        self.prod_ids = None
        self.prod_texts = None
        self.tfidf_vectorizer = None
        self.X_prod_tfidf = None
        self.bm25 = BM25()
        self.count_vectorizer = None  # for overlap numeric features if needed
        self.prod_meta = None  # DataFrame copy with meta columns

    def fit(self,
            products_df: pd.DataFrame,
            prod_id_col: str = 'p_id',
            text_col: str = 'text_norm',
            title_col: Optional[str] = 'title',
            brand_col: Optional[str] = 'brand',
            category_col: Optional[str] = 'category',
            attr_col: Optional[str] = 'attributes_list'):
        """
        Fit TF-IDF, BM25 on product corpus and store meta fields.

        products_df: DataFrame with product rows. Must contain prod_id_col and text_col.
        """
        df = products_df.copy().reset_index(drop=True)
        if prod_id_col not in df.columns or text_col not in df.columns:
            raise ValueError("products_df must contain prod_id_col and text_col")

        self.prod_meta = df[[prod_id_col]].copy()
        self.prod_meta = self.prod_meta.rename(columns={prod_id_col: 'p_id'})
        # keep other meta if present
        for c in (title_col, brand_col, category_col, attr_col):
            if c and c in df.columns:
                self.prod_meta[c] = df[c]

        self.prod_ids = self.prod_meta['p_id'].astype(str).tolist()
        self.prod_texts = df[text_col].fillna('').astype(str).tolist()

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.tfidf_max_features,
                                                token_pattern=r"[\w\u0600-\u06FF]+")
        self.X_prod_tfidf = self.tfidf_vectorizer.fit_transform(self.prod_texts)
        # normalize for cosine
        self.X_prod_tfidf = normalize(self.X_prod_tfidf, axis=1)

        # BM25 (uses CountVectorizer internally)
        self.bm25.fit(self.prod_texts, token_pattern=r"[\w\u0600-\u06FF]+")

        # CountVectorizer for attribute/overlap counts (shared vocabulary)
        self.count_vectorizer = CountVectorizer(token_pattern=r"[\w\u0600-\u06FF]+")
        self.count_vectorizer.fit(self.prod_texts)

    def get_candidates_for_queries(self,
                                   queries_df: pd.DataFrame,
                                   query_col: str = 'query_norm',
                                   id_col: str = 'query_id') -> pd.DataFrame:
        """
        Return a DataFrame of candidate pairs (query_id, query, p_id) using TF-IDF top_k retrieval.

        queries_df must include query_col and id_col.
        """
        if query_col not in queries_df.columns:
            raise ValueError(f"queries_df must contain {query_col}")

        qs = queries_df[query_col].fillna('').astype(str).tolist()
        qids = queries_df[id_col].astype(str).tolist()

        # vectorize queries
        Xq = self.tfidf_vectorizer.transform(qs)
        Xq = normalize(Xq, axis=1)
        # compute cosine to products in batches (memory-savvy)
        # for simplicity here we compute full matrix if small; otherwise batch
        sims = cosine_similarity(Xq, self.X_prod_tfidf)  # Q x P
        candidates = []
        for i, qid in enumerate(qids):
            row = sims[i]
            if self.top_k >= len(row):
                idx = np.argsort(-row)
            else:
                idx = np.argpartition(-row, self.top_k - 1)[:self.top_k]
                idx = idx[np.argsort(-row[idx])]
            for j in idx:
                candidates.append({'query_id': str(qid), 'query': qs[i], 'p_id': self.prod_ids[j], 'tfidf_score': float(row[j])})
        candidates_df = pd.DataFrame(candidates)
        return candidates_df

    def transform(self,
                  candidates_df: pd.DataFrame,
                  queries_df: Optional[pd.DataFrame] = None,
                  query_col: str = 'query',
                  query_id_col: str = 'query_id') -> pd.DataFrame:
        """
        Compute features for the candidate pairs. Returns a DataFrame with features.

        candidates_df must contain: ['query_id','query','p_id'] (query text should be normalized string)
        If queries_df provided, it may include additional columns (original query, etc.)
        """
        required = ['query_id', 'query', 'p_id']
        for c in required:
            if c not in candidates_df.columns:
                raise ValueError(f"candidates_df must contain {required}")

        # prepare quick lookups
        pid_to_idx = {p: i for i, p in enumerate(self.prod_ids)}
        prod_texts = self.prod_texts
        prod_meta = self.prod_meta if self.prod_meta is not None else pd.DataFrame({'p_id': self.prod_ids})

        # build mapping dicts for faster, safe meta lookup (avoid .loc inside loop)
        title_map = {}
        brand_map = {}
        category_map = {}
        attr_map = {}
        if 'p_id' in prod_meta.columns:
            if 'title' in prod_meta.columns:
                title_map = pd.Series(prod_meta['title'].values, index=prod_meta['p_id'].astype(str)).to_dict()
            if 'brand' in prod_meta.columns:
                brand_map = pd.Series(prod_meta['brand'].values, index=prod_meta['p_id'].astype(str)).to_dict()
            if 'category' in prod_meta.columns:
                category_map = pd.Series(prod_meta['category'].values, index=prod_meta['p_id'].astype(str)).to_dict()
            if 'attributes_list' in prod_meta.columns:
                attr_map = pd.Series(prod_meta['attributes_list'].values, index=prod_meta['p_id'].astype(str)).to_dict()

        rows = []

        # small BM25 cache per query to avoid repeated full-score computation
        last_qtext = None
        last_full_scores = None

        for _, row in candidates_df.iterrows():
            qid = str(row['query_id'])
            qtext = str(row['query'])
            pid = str(row['p_id'])

            entry = {'query_id': qid, 'query': qtext, 'p_id': pid}

            # TF-IDF cosine (prefer candidate-provided score)
            if 'tfidf_score' in row.index and not pd.isna(row['tfidf_score']):
                entry['tfidf_cosine'] = float(row['tfidf_score'])
            else:
                qv = self.tfidf_vectorizer.transform([qtext])
                qv = normalize(qv, axis=1)
                if pid in pid_to_idx:
                    pv = self.X_prod_tfidf[pid_to_idx[pid]]
                    sim = float(qv.dot(pv.T).data[0]) if qv.dot(pv.T).data.size else 0.0
                else:
                    sim = 0.0
                entry['tfidf_cosine'] = sim

            # BM25 score (use cached full_scores for this query)
            try:
                if last_qtext != qtext:
                    last_full_scores = self.bm25.score(qtext)
                    last_qtext = qtext
                entry['bm25'] = float(last_full_scores[pid_to_idx[pid]]) if (last_full_scores is not None and pid in pid_to_idx) else 0.0
            except Exception:
                # safe fallback
                try:
                    full_scores = self.bm25.score(qtext)
                    entry['bm25'] = float(full_scores[pid_to_idx[pid]]) if pid in pid_to_idx else 0.0
                except Exception:
                    entry['bm25'] = 0.0

            # lexical overlap / jaccard
            q_tokens = tokenize(qtext)
            prod_text = prod_texts[pid_to_idx.get(pid, 0)] if pid in pid_to_idx else ""
            p_tokens = tokenize(prod_text)
            entry['jaccard_tokens'] = jaccard(q_tokens, p_tokens)
            entry['overlap_count'] = len(set(q_tokens).intersection(p_tokens))
            entry['query_len'] = len(q_tokens)
            entry['prod_len'] = len(p_tokens)

            # title fuzzy similarity (if title available)
            title = ''
            if pid in title_map:
                title_val = title_map.get(pid)
                title = '' if pd.isna(title_val) else str(title_val)
            entry['title_fuzzy'] = SequenceMatcher(None, qtext, title).ratio() if title else 0.0

            # brand/category exact match features (safe NaN handling)
            entry['brand_exact'] = 0
            if pid in brand_map:
                brand_val = brand_map.get(pid)
                if not pd.isna(brand_val):
                    brand_str = str(brand_val).strip()
                    entry['brand_exact'] = 1 if brand_str and (brand_str in qtext) else 0

            entry['category_exact'] = 0
            if pid in category_map:
                cat_val = category_map.get(pid)
                if not pd.isna(cat_val):
                    cat_str = str(cat_val).strip()
                    entry['category_exact'] = 1 if cat_str and (cat_str in qtext) else 0

            # attributes overlap (safe robust handling for lists, arrays, strings, NaN)
            entry['attr_overlap_count'] = 0
            entry['attr_overlap_ratio'] = 0.0
            if pid in attr_map:
                attrs = attr_map.get(pid)

                # normalize attrs to a Python list of strings
                if attrs is None:
                    attrs_list = []
                elif isinstance(attrs, (list, tuple)):
                    attrs_list = list(attrs)
                elif isinstance(attrs, (np.ndarray,)):
                    attrs_list = attrs.tolist()
                else:
                    # attrs is scalar (maybe string) or pandas object
                    try:
                        # handle pandas NA-like values
                        if pd.isna(attrs):
                            attrs_list = []
                        else:
                            # if it's a string, split on commas / arabic comma / newlines; otherwise coerce to str and split
                            if isinstance(attrs, str):
                                attrs_list = [a.strip() for a in re.split(r'[,\u060C\；\n\r]+', attrs) if a.strip()]
                            else:
                                # fallback: coerce to str and split
                                s = str(attrs)
                                attrs_list = [a.strip() for a in re.split(r'[,\u060C\；\n\r]+', s) if a.strip()]
                    except Exception:
                        attrs_list = []

                # final cleanup and overlap counting
                if isinstance(attrs_list, (list, tuple)):
                    attrs_norm = [str(a).strip() for a in attrs_list if a is not None and str(a).strip()]
                    overlap_count = 0
                    if attrs_norm:
                        q_tokens_set = set(q_tokens)
                        for a in attrs_norm:
                            if (a in qtext) or (len(set(tokenize(a)).intersection(q_tokens_set)) > 0):
                                overlap_count += 1
                    entry['attr_overlap_count'] = int(overlap_count)
                    entry['attr_overlap_ratio'] = float(overlap_count) / len(attrs_norm) if len(attrs_norm) > 0 else 0.0
                else:
                    entry['attr_overlap_count'] = 0
                    entry['attr_overlap_ratio'] = 0.0

            # digits / model tokens overlap
            q_digits = extract_digits(qtext)
            p_digits = extract_digits(prod_text)
            entry['digit_overlap_ratio'] = 0.0
            if q_digits:
                entry['digit_overlap_ratio'] = len([d for d in q_digits if d in p_digits]) / len(q_digits)
            entry['num_query_digits'] = len(q_digits)

            # model token heuristics
            def is_model_token(tok):
                return any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok)
            q_model_tokens = [t for t in q_tokens if is_model_token(t)]
            p_model_tokens = [t for t in p_tokens if is_model_token(t)]
            entry['model_token_match'] = int(len(set(q_model_tokens).intersection(p_model_tokens)))
            entry['num_q_model_tokens'] = int(len(q_model_tokens))

            # product popularity proxy
            entry['train_product_freq'] = int(self._train_pid_counts.get(pid, 0)) if hasattr(self, '_train_pid_counts') else 0

            rows.append(entry)

        feats = pd.DataFrame(rows)
        cols_first = ['query_id', 'query', 'p_id']
        rest = [c for c in feats.columns if c not in cols_first]
        feats = feats[cols_first + rest]
        return feats

    def fit_train_counts(self, train_pairs_df: pd.DataFrame, pid_col: str = 'p_id'):
        """
        Optional: store train pid frequencies to use as a popularity signal.
        """
        if pid_col in train_pairs_df.columns:
            cnt = Counter(train_pairs_df[pid_col].astype(str).tolist())
            self._train_pid_counts = dict(cnt)
        else:
            self._train_pid_counts = {}



from normalizer import PersianNormalizer
from data_preprocessing import Preprocessor
from data_loader import load_train_pairs  # optional: you can also use load_test_queries
from feature_engineering import FeatureEngineer
import pandas as pd

# init
normalizer = PersianNormalizer()
pre = Preprocessor(normalizer=normalizer, map_digits=True, max_text_chars=1024)

# load products (raw) and preprocess (creates text_norm, attributes_list, etc.)
products_raw = pd.read_csv('data/products.csv', dtype=str)
products_clean = pre.clean_products_df(products_raw)

# load train pairs (for popularity signal)
train_pairs = load_train_pairs('data/train_query_product_pairs.csv')

# load tests and ensure normalized query column exists
tests = pd.read_csv('data/test_queries.csv', dtype=str)    # your file
if 'query_norm' not in tests.columns:
    tests['query_norm'] = tests['query'].map(lambda s: pre.normalize_query(s) if pd.notna(s) else '')

# build features
fe = FeatureEngineer(top_k=50)   # adjust top_k as you like
fe.fit(products_clean, prod_id_col='p_id', text_col='text_norm', title_col='title', brand_col='brand', attr_col='attributes_list')
fe.fit_train_counts(train_pairs)

cands = fe.get_candidates_for_queries(tests, query_col='query_norm', id_col='query_id')
features = fe.transform(cands)
features.to_csv('data/features_baseline.csv', index=False)
print("Saved features to data/features_baseline.csv")