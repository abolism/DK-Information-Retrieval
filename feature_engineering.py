# # feature_engineering.py
# """
# Feature engineering utilities for e-commerce IR.

# Provides:
#  - FeatureEngineer: fit on a product corpus and produce features for query-candidate pairs.
#  - BM25 implementation (okapi) for scoring queries->documents.
#  - Candidate generation via TF-IDF top-k.
#  - Many explainable features:
#      * TF-IDF cosine similarity
#      * BM25 score
#      * token Jaccard / overlap
#      * exact brand / category match
#      * attributes overlap count / ratio
#      * digit/model matching features
#      * query/product length features
#      * fuzzy title similarity (difflib.SequenceMatcher)
#  - All functions expect preprocessed text (use text_norm / query_norm from your Preprocessor)

# Dependencies:
#   numpy, pandas, scikit-learn

# Usage:
#   from feature_engineering import FeatureEngineer
#   fe = FeatureEngineer(top_k=100)
#   fe.fit(products_df, prod_id_col='p_id', text_col='text_norm')
#   # build candidates (uses TF-IDF top_k retrieval on products)
#   candidates_df = fe.get_candidates_for_queries(tests_df, query_col='query_norm')
#   # compute features for candidate pairs
#   features_df = fe.transform(candidates_df, queries_df=tests_df)
# """

# from typing import List, Optional, Dict, Any, Tuple
# from collections import Counter
# import re
# import math
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# from difflib import SequenceMatcher

# # -------------------------
# # Tokenizer (conservative)
# # -------------------------
# _WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)

# def tokenize(text: str) -> List[str]:
#     if text is None:
#         return []
#     return _WORD_RE.findall(str(text))


# def extract_digits(text: str) -> List[str]:
#     """Return contiguous digit tokens found in text (ASCII or persian digits)."""
#     if text is None:
#         return []
#     return re.findall(r"\d+", str(text))


# def jaccard(a: List[str], b: List[str]) -> float:
#     if not a and not b:
#         return 0.0
#     sa, sb = set(a), set(b)
#     inter = sa.intersection(sb)
#     union = sa.union(sb)
#     return len(inter) / len(union) if union else 0.0


# # -------------------------
# # Small BM25 implementation (vectorized)
# # -------------------------
# class BM25:
#     """
#     Simple BM25 implemented with CountVectorizer internals for speed.
#     After fit(corpus) you can call score_queries(queries) to compute BM25 scores against all docs.
#     """

#     def __init__(self, k1: float = 1.2, b: float = 0.75):
#         self.k1 = k1
#         self.b = b
#         self.vectorizer = None
#         self.tf = None             # document-term matrix (csr)
#         self.df = None             # document frequency per term
#         self.idf = None            # idf per term
#         self.doc_len = None        # length per document
#         self.avgdl = None
#         self.N = 0

#     def fit(self, corpus: List[str], min_df: int = 1, token_pattern: str = r"(?u)\b\w+\b"):
#         """
#         Fit BM25 to a list of documents (strings).
#         """
#         self.vectorizer = CountVectorizer(token_pattern=token_pattern, min_df=min_df)
#         self.tf = self.vectorizer.fit_transform(corpus)  # shape (N_docs, V)
#         self.N = self.tf.shape[0]
#         # document frequencies
#         self.df = np.bincount(self.tf.indices, minlength=self.tf.shape[1])
#         # safe idf: standard BM25 idf
#         # idf = log((N - df + 0.5) / (df + 0.5))
#         self.idf = np.log((self.N - self.df + 0.5) / (self.df + 0.5) + 1e-9)
#         self.doc_len = np.array(self.tf.sum(axis=1)).ravel()
#         self.avgdl = float(self.doc_len.mean()) if self.N > 0 else 0.0

#     def _tf_query(self, query: str):
#         # transform query using same vocabulary
#         if self.vectorizer is None:
#             raise ValueError("BM25 not fitted")
#         q_tf = self.vectorizer.transform([query])  # shape (1, V)
#         return q_tf

#     def score(self, query: str, topk: Optional[int] = None) -> np.ndarray:
#         """
#         Return BM25 score array for all documents for a single query.
#         If topk specified, returns only topk scores (and corresponding doc indices).
#         """
#         q_tf = self._tf_query(query)  # 1 x V
#         q_indices = q_tf.indices
#         if q_indices.size == 0:
#             # empty query => zeros
#             if topk:
#                 return np.array([]), np.array([])
#             return np.zeros(self.N, dtype=float)

#         # only compute on terms present in query for speed
#         # tf_doc_col = self.tf[:, q_index].toarray().ravel()
#         scores = np.zeros(self.N, dtype=float)
#         k1 = self.k1
#         b = self.b
#         for term_idx in q_indices:
#             idf_t = self.idf[term_idx]
#             tf_col = self.tf[:, term_idx].toarray().ravel()  # tf for this term in all docs
#             denom = tf_col + k1 * (1 - b + b * (self.doc_len / self.avgdl))
#             numer = tf_col * (k1 + 1)
#             scores += idf_t * (numer / (denom + 1e-12))
#         if topk:
#             # return topk indices and scores sorted
#             topk = min(topk, len(scores))
#             idx = np.argpartition(-scores, topk - 1)[:topk]
#             idx_sorted = idx[np.argsort(-scores[idx])]
#             return idx_sorted, scores[idx_sorted]
#         return scores


# # -------------------------
# # FeatureEngineer
# # -------------------------
# class FeatureEngineer:
#     """
#     Fitable feature engineer for IR.

#     Primary usage:
#       fe = FeatureEngineer(top_k=100)
#       fe.fit(products_df, prod_id_col='p_id', text_col='text_norm', title_col='title', brand_col='brand', attr_col='attributes_list')
#       candidates_df = fe.get_candidates_for_queries(tests_df, query_col='query_norm')  # returns DataFrame with columns ['query_id','query','pid']
#       features_df = fe.transform(candidates_df, queries_df=tests_df)

#     Output features_df columns include:
#       query_id, query, p_id,
#       tfidf_cosine, bm25, jaccard_tokens, title_fuzzy, brand_exact, category_exact,
#       attr_overlap_count, attr_overlap_ratio, digit_overlap_ratio, query_len, prod_len, ...
#     """

#     def __init__(self, top_k: int = 100, tfidf_max_features: int = 75000, ngram_range: Tuple[int,int]=(1,2)):
#         self.top_k = top_k
#         self.tfidf_max_features = tfidf_max_features
#         self.ngram_range = ngram_range

#         # Will be initialized in fit
#         self.prod_ids = None
#         self.prod_texts = None
#         self.tfidf_vectorizer = None
#         self.X_prod_tfidf = None
#         self.bm25 = BM25()
#         self.count_vectorizer = None  # for overlap numeric features if needed
#         self.prod_meta = None  # DataFrame copy with meta columns

#     def fit(self,
#             products_df: pd.DataFrame,
#             prod_id_col: str = 'p_id',
#             text_col: str = 'text_norm',
#             title_col: Optional[str] = 'title',
#             brand_col: Optional[str] = 'brand',
#             category_col: Optional[str] = 'category',
#             attr_col: Optional[str] = 'attributes_list'):
#         """
#         Fit TF-IDF, BM25 on product corpus and store meta fields.

#         products_df: DataFrame with product rows. Must contain prod_id_col and text_col.
#         """
#         df = products_df.copy().reset_index(drop=True)
#         if prod_id_col not in df.columns or text_col not in df.columns:
#             raise ValueError("products_df must contain prod_id_col and text_col")

#         self.prod_meta = df[[prod_id_col]].copy()
#         self.prod_meta = self.prod_meta.rename(columns={prod_id_col: 'p_id'})
#         # keep other meta if present
#         for c in (title_col, brand_col, category_col, attr_col):
#             if c and c in df.columns:
#                 self.prod_meta[c] = df[c]

#         self.prod_ids = self.prod_meta['p_id'].astype(str).tolist()
#         self.prod_texts = df[text_col].fillna('').astype(str).tolist()

#         # TF-IDF
#         self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.tfidf_max_features,
#                                                 token_pattern=r"[\w\u0600-\u06FF]+")
#         self.X_prod_tfidf = self.tfidf_vectorizer.fit_transform(self.prod_texts)
#         # normalize for cosine
#         self.X_prod_tfidf = normalize(self.X_prod_tfidf, axis=1)

#         # BM25 (uses CountVectorizer internally)
#         self.bm25.fit(self.prod_texts, token_pattern=r"[\w\u0600-\u06FF]+")

#         # CountVectorizer for attribute/overlap counts (shared vocabulary)
#         self.count_vectorizer = CountVectorizer(token_pattern=r"[\w\u0600-\u06FF]+")
#         self.count_vectorizer.fit(self.prod_texts)

#     def get_candidates_for_queries(self,
#                                    queries_df: pd.DataFrame,
#                                    query_col: str = 'query_norm',
#                                    id_col: str = 'query_id') -> pd.DataFrame:
#         """
#         Return a DataFrame of candidate pairs (query_id, query, p_id) using TF-IDF top_k retrieval.

#         queries_df must include query_col and id_col.
#         """
#         if query_col not in queries_df.columns:
#             raise ValueError(f"queries_df must contain {query_col}")

#         qs = queries_df[query_col].fillna('').astype(str).tolist()
#         qids = queries_df[id_col].astype(str).tolist()

#         # vectorize queries
#         Xq = self.tfidf_vectorizer.transform(qs)
#         Xq = normalize(Xq, axis=1)
#         # compute cosine to products in batches (memory-savvy)
#         # for simplicity here we compute full matrix if small; otherwise batch
#         sims = cosine_similarity(Xq, self.X_prod_tfidf)  # Q x P
#         candidates = []
#         for i, qid in enumerate(qids):
#             row = sims[i]
#             if self.top_k >= len(row):
#                 idx = np.argsort(-row)
#             else:
#                 idx = np.argpartition(-row, self.top_k - 1)[:self.top_k]
#                 idx = idx[np.argsort(-row[idx])]
#             for j in idx:
#                 candidates.append({'query_id': str(qid), 'query': qs[i], 'p_id': self.prod_ids[j], 'tfidf_score': float(row[j])})
#         candidates_df = pd.DataFrame(candidates)
#         return candidates_df

#     def transform(self,
#                   candidates_df: pd.DataFrame,
#                   queries_df: Optional[pd.DataFrame] = None,
#                   query_col: str = 'query',
#                   query_id_col: str = 'query_id') -> pd.DataFrame:
#         """
#         Compute features for the candidate pairs. Returns a DataFrame with features.

#         candidates_df must contain: ['query_id','query','p_id'] (query text should be normalized string)
#         If queries_df provided, it may include additional columns (original query, etc.)
#         """
#         required = ['query_id', 'query', 'p_id']
#         for c in required:
#             if c not in candidates_df.columns:
#                 raise ValueError(f"candidates_df must contain {required}")

#         # prepare quick lookups
#         pid_to_idx = {p: i for i, p in enumerate(self.prod_ids)}
#         prod_texts = self.prod_texts
#         prod_meta = self.prod_meta if self.prod_meta is not None else pd.DataFrame({'p_id': self.prod_ids})

#         # build mapping dicts for faster, safe meta lookup (avoid .loc inside loop)
#         title_map = {}
#         brand_map = {}
#         category_map = {}
#         attr_map = {}
#         if 'p_id' in prod_meta.columns:
#             if 'title' in prod_meta.columns:
#                 title_map = pd.Series(prod_meta['title'].values, index=prod_meta['p_id'].astype(str)).to_dict()
#             if 'brand' in prod_meta.columns:
#                 brand_map = pd.Series(prod_meta['brand'].values, index=prod_meta['p_id'].astype(str)).to_dict()
#             if 'category' in prod_meta.columns:
#                 category_map = pd.Series(prod_meta['category'].values, index=prod_meta['p_id'].astype(str)).to_dict()
#             if 'attributes_list' in prod_meta.columns:
#                 attr_map = pd.Series(prod_meta['attributes_list'].values, index=prod_meta['p_id'].astype(str)).to_dict()

#         rows = []

#         # small BM25 cache per query to avoid repeated full-score computation
#         last_qtext = None
#         last_full_scores = None

#         for _, row in candidates_df.iterrows():
#             qid = str(row['query_id'])
#             qtext = str(row['query'])
#             pid = str(row['p_id'])

#             entry = {'query_id': qid, 'query': qtext, 'p_id': pid}

#             # TF-IDF cosine (prefer candidate-provided score)
#             if 'tfidf_score' in row.index and not pd.isna(row['tfidf_score']):
#                 entry['tfidf_cosine'] = float(row['tfidf_score'])
#             else:
#                 qv = self.tfidf_vectorizer.transform([qtext])
#                 qv = normalize(qv, axis=1)
#                 if pid in pid_to_idx:
#                     pv = self.X_prod_tfidf[pid_to_idx[pid]]
#                     sim = float(qv.dot(pv.T).data[0]) if qv.dot(pv.T).data.size else 0.0
#                 else:
#                     sim = 0.0
#                 entry['tfidf_cosine'] = sim

#             # BM25 score (use cached full_scores for this query)
#             try:
#                 if last_qtext != qtext:
#                     last_full_scores = self.bm25.score(qtext)
#                     last_qtext = qtext
#                 entry['bm25'] = float(last_full_scores[pid_to_idx[pid]]) if (last_full_scores is not None and pid in pid_to_idx) else 0.0
#             except Exception:
#                 # safe fallback
#                 try:
#                     full_scores = self.bm25.score(qtext)
#                     entry['bm25'] = float(full_scores[pid_to_idx[pid]]) if pid in pid_to_idx else 0.0
#                 except Exception:
#                     entry['bm25'] = 0.0

#             # lexical overlap / jaccard
#             q_tokens = tokenize(qtext)
#             prod_text = prod_texts[pid_to_idx.get(pid, 0)] if pid in pid_to_idx else ""
#             p_tokens = tokenize(prod_text)
#             entry['jaccard_tokens'] = jaccard(q_tokens, p_tokens)
#             entry['overlap_count'] = len(set(q_tokens).intersection(p_tokens))
#             entry['query_len'] = len(q_tokens)
#             entry['prod_len'] = len(p_tokens)

#             # title fuzzy similarity (if title available)
#             title = ''
#             if pid in title_map:
#                 title_val = title_map.get(pid)
#                 title = '' if pd.isna(title_val) else str(title_val)
#             entry['title_fuzzy'] = SequenceMatcher(None, qtext, title).ratio() if title else 0.0

#             # brand/category exact match features (safe NaN handling)
#             entry['brand_exact'] = 0
#             if pid in brand_map:
#                 brand_val = brand_map.get(pid)
#                 if not pd.isna(brand_val):
#                     brand_str = str(brand_val).strip()
#                     entry['brand_exact'] = 1 if brand_str and (brand_str in qtext) else 0

#             entry['category_exact'] = 0
#             if pid in category_map:
#                 cat_val = category_map.get(pid)
#                 if not pd.isna(cat_val):
#                     cat_str = str(cat_val).strip()
#                     entry['category_exact'] = 1 if cat_str and (cat_str in qtext) else 0

#             # attributes overlap (safe robust handling for lists, arrays, strings, NaN)
#             entry['attr_overlap_count'] = 0
#             entry['attr_overlap_ratio'] = 0.0
#             if pid in attr_map:
#                 attrs = attr_map.get(pid)

#                 # normalize attrs to a Python list of strings
#                 if attrs is None:
#                     attrs_list = []
#                 elif isinstance(attrs, (list, tuple)):
#                     attrs_list = list(attrs)
#                 elif isinstance(attrs, (np.ndarray,)):
#                     attrs_list = attrs.tolist()
#                 else:
#                     # attrs is scalar (maybe string) or pandas object
#                     try:
#                         # handle pandas NA-like values
#                         if pd.isna(attrs):
#                             attrs_list = []
#                         else:
#                             # if it's a string, split on commas / arabic comma / newlines; otherwise coerce to str and split
#                             if isinstance(attrs, str):
#                                 attrs_list = [a.strip() for a in re.split(r'[,\u060C\；\n\r]+', attrs) if a.strip()]
#                             else:
#                                 # fallback: coerce to str and split
#                                 s = str(attrs)
#                                 attrs_list = [a.strip() for a in re.split(r'[,\u060C\；\n\r]+', s) if a.strip()]
#                     except Exception:
#                         attrs_list = []

#                 # final cleanup and overlap counting
#                 if isinstance(attrs_list, (list, tuple)):
#                     attrs_norm = [str(a).strip() for a in attrs_list if a is not None and str(a).strip()]
#                     overlap_count = 0
#                     if attrs_norm:
#                         q_tokens_set = set(q_tokens)
#                         for a in attrs_norm:
#                             if (a in qtext) or (len(set(tokenize(a)).intersection(q_tokens_set)) > 0):
#                                 overlap_count += 1
#                     entry['attr_overlap_count'] = int(overlap_count)
#                     entry['attr_overlap_ratio'] = float(overlap_count) / len(attrs_norm) if len(attrs_norm) > 0 else 0.0
#                 else:
#                     entry['attr_overlap_count'] = 0
#                     entry['attr_overlap_ratio'] = 0.0

#             # digits / model tokens overlap
#             q_digits = extract_digits(qtext)
#             p_digits = extract_digits(prod_text)
#             entry['digit_overlap_ratio'] = 0.0
#             if q_digits:
#                 entry['digit_overlap_ratio'] = len([d for d in q_digits if d in p_digits]) / len(q_digits)
#             entry['num_query_digits'] = len(q_digits)

#             # model token heuristics
#             def is_model_token(tok):
#                 return any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok)
#             q_model_tokens = [t for t in q_tokens if is_model_token(t)]
#             p_model_tokens = [t for t in p_tokens if is_model_token(t)]
#             entry['model_token_match'] = int(len(set(q_model_tokens).intersection(p_model_tokens)))
#             entry['num_q_model_tokens'] = int(len(q_model_tokens))

#             # product popularity proxy
#             entry['train_product_freq'] = int(self._train_pid_counts.get(pid, 0)) if hasattr(self, '_train_pid_counts') else 0

#             rows.append(entry)

#         feats = pd.DataFrame(rows)
#         cols_first = ['query_id', 'query', 'p_id']
#         rest = [c for c in feats.columns if c not in cols_first]
#         feats = feats[cols_first + rest]
#         return feats

#     def fit_train_counts(self, train_pairs_df: pd.DataFrame, pid_col: str = 'p_id'):
#         """
#         Optional: store train pid frequencies to use as a popularity signal.
#         """
#         if pid_col in train_pairs_df.columns:
#             cnt = Counter(train_pairs_df[pid_col].astype(str).tolist())
#             self._train_pid_counts = dict(cnt)
#         else:
#             self._train_pid_counts = {}



# from normalizer import PersianNormalizer
# from data_preprocessing import Preprocessor
# from data_loader import load_train_pairs  # optional: you can also use load_test_queries
# from feature_engineering import FeatureEngineer
# import pandas as pd

# # init
# normalizer = PersianNormalizer()
# pre = Preprocessor(normalizer=normalizer, map_digits=True, max_text_chars=1024)

# # load products (raw) and preprocess (creates text_norm, attributes_list, etc.)
# products_raw = pd.read_csv('data/products.csv', dtype=str)
# products_clean = pre.clean_products_df(products_raw)

# # load train pairs (for popularity signal)
# train_pairs = load_train_pairs('data/train_query_product_pairs.csv')

# # load tests and ensure normalized query column exists
# tests = pd.read_csv('data/test_queries.csv', dtype=str)    # your file
# if 'query_norm' not in tests.columns:
#     tests['query_norm'] = tests['query'].map(lambda s: pre.normalize_query(s) if pd.notna(s) else '')

# # build features
# fe = FeatureEngineer(top_k=50)   # adjust top_k as you like
# fe.fit(products_clean, prod_id_col='p_id', text_col='text_norm', title_col='title', brand_col='brand', attr_col='attributes_list')
# fe.fit_train_counts(train_pairs)

# cands = fe.get_candidates_for_queries(tests, query_col='query_norm', id_col='query_id')
# features = fe.transform(cands)
# features.to_csv('data/features_baseline.csv', index=False)
# print("Saved features to data/features_baseline.csv")


# feature_engineering.py
"""
Feature engineering & retrieval helpers for the IR pipeline.

Usage sketch:
    from feature_engineering import ProductIndex, build_pair_features

    idx = ProductIndex(normalizer, min_df=3)
    idx.fit(products_df)                # products_df must contain p_id,title,category,brand,attributes (text/text_norm optional)
    candidates = idx.get_candidates_bm25("ضد آفتاب لاروش", k=200)
    ranked = idx.retrieve_ranked_candidates("ضد آفتاب لاروش", k=50)
    feats_df = build_pair_features(query="ضد آفتاب لاروش", query_norm="...", products_df=products_df.loc[ranked], idx=idx)
"""

from __future__ import annotations
import re
import math
import pickle
from typing import List, Dict, Iterable, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local normalizer expected
# from normalizer import PersianNormalizer
# We'll accept a normalizer instance passed to ProductIndex to avoid tight coupling.

# ----------------------------
# Utilities: tokenization / model number extraction
# ----------------------------
_TOKEN_RE = re.compile(r'[\w\u0600-\u06FF]+', flags=re.UNICODE)  # keeps digits, latin, persian/arabic letters
_MODEL_RE = re.compile(r'([a-zA-Z]{1,4}\s?-?\s?\d{2,6}|\d{2,6}[a-zA-Z]{0,4}|[A-Za-z0-9]+[-_/]\d+)', flags=re.I)

def tokenize_text_for_ir(text: str) -> List[str]:
    """Simple whitespace/token regex tokenizer for Persian + latin + numbers. Expects normalized text."""
    if text is None:
        return []
    return [t for t in _TOKEN_RE.findall(str(text)) if t.strip()]

def extract_model_tokens(text: str) -> List[str]:
    """Extract likely model tokens (e.g., 'A35', 'note9', 'iPhone12', 'M33')."""
    if text is None:
        return []
    res = []
    for m in _MODEL_RE.findall(text):
        s = re.sub(r'\s+', '', m)
        if len(s) >= 2:
            res.append(s.lower())
    return list(dict.fromkeys(res))

def numeric_tokens(text: str) -> List[str]:
    """Return numeric tokens found in text."""
    toks = tokenize_text_for_ir(text)
    return [t for t in toks if re.fullmatch(r'\d+', t)]

# ----------------------------
# Minimal BM25 implementation
# ----------------------------
class BM25:
    """
    Lightweight BM25 (Okapi) implementation for document scoring.
    Stores an inverted index mapping token -> list of (doc_id, freq).
    Document ids are internal 0..N-1; mapping to external p_id is handled by ProductIndex.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.doc_len = []
        self.inv_index = defaultdict(dict)  # token -> {docid: tf}
        self.df = {}  # document frequency per token

    def fit(self, docs_tokens: Iterable[List[str]]):
        docs_tokens = list(docs_tokens)
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = float(sum(self.doc_len) / self.N) if self.N>0 else 0.0
        for docid, tokens in enumerate(docs_tokens):
            tf = Counter(tokens)
            for t, c in tf.items():
                self.inv_index[t][docid] = c
        self.df = {t: len(d) for t, d in self.inv_index.items()}

    def score(self, query_tokens: List[str]) -> Dict[int, float]:
        """Return docid -> BM25 score for docs containing query tokens."""
        scores = defaultdict(float)
        qtf = Counter(query_tokens)
        for t in qtf:
            if t not in self.inv_index:
                continue
            df = self.df.get(t, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1e-9)
            postings = self.inv_index[t]
            for docid, f in postings.items():
                dl = self.doc_len[docid]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * (f * (self.k1 + 1)) / (denom + 1e-9)
                scores[docid] += score
        return scores

# ----------------------------
# ProductIndex: encapsulates products, indices, vectorizers
# ----------------------------
class ProductIndex:
    """
    Holds product data and retrieval indices (BM25 + TF-IDF).
    Fit with a products DataFrame and a normalizer instance (PersianNormalizer or similar).
    """
    def __init__(self,
                 normalizer,
                 min_df: int = 2,
                 ngram_range: Tuple[int,int] = (1,2),
                 tfidf_max_features: Optional[int] = 200000):
        """
        Args:
            normalizer: instance exposing .normalize(text) method
            min_df: min document freq for TF-IDF vectorizer
            ngram_range: n-gram range for TF-IDF
            tfidf_max_features: cap on TF-IDF features
        """
        self.normalizer = normalizer
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.tfidf_max_features = tfidf_max_features

        # product storage
        self.products: pd.DataFrame = pd.DataFrame()
        self.docid_to_pid: List[str] = []
        self.pid_to_docid: Dict[str,int] = {}

        # indices
        self.bm25 = BM25()
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[sparse.csr_matrix] = None

        # tokenized form cached
        self.docs_tokens: List[List[str]] = []
        self.model_tokens: List[List[str]] = []

        # popularity (optional) - counts from training pairs if provided
        self.popularity: Dict[str,int] = {}

    # ---------------------
    # Fit / persistence
    # ---------------------
    def fit(self, products_df: pd.DataFrame, text_field: str = "text_norm", build_tfidf: bool = True,
            train_pairs_df: Optional[pd.DataFrame] = None):
        """
        Fit index structures from products DataFrame.

        products_df must contain at least columns: p_id, title, category, brand, attributes.
        Preferred included columns: text_norm (if not present, normalized from available text columns).
        """
        df = products_df.copy()
        df.columns = [c.strip() for c in df.columns]
        required = ['p_id']
        for r in required:
            if r not in df.columns:
                raise ValueError(f"products_df must contain '{r}'")
        # Ensure p_id string
        df['p_id'] = df['p_id'].astype(str)

        # Make sure text_norm exists
        if text_field not in df.columns or df[text_field].isna().all():
            # try to build from 'text' or title/category fields
            if 'text' in df.columns and not df['text'].isna().all():
                df[text_field] = df['text'].astype(str).map(lambda s: self.normalizer.normalize(s))
            else:
                # compose from title/category/brand/attributes
                cmp_fields = []
                for c in ('title','category','brand','attributes'):
                    if c in df.columns:
                        cmp_fields.append(df[c].fillna("").astype(str))
                if cmp_fields:
                    df['text'] = cmp_fields[0].astype(str)
                    for s in cmp_fields[1:]:
                        df['text'] = df['text'].astype(str) + ' ' + s.astype(str)
                    df[text_field] = df['text'].astype(str).map(lambda s: self.normalizer.normalize(s))
                else:
                    df[text_field] = ''

        # create canonical mapping
        self.products = df.reset_index(drop=True)
        self.docid_to_pid = list(self.products['p_id'].astype(str))
        self.pid_to_docid = {pid: i for i, pid in enumerate(self.docid_to_pid)}

        # construct tokens
        self.docs_tokens = [tokenize_text_for_ir(t) for t in list(self.products[text_field].astype(str))]
        self.model_tokens = [extract_model_tokens(t) for t in list(self.products[text_field].astype(str))]

        # fit BM25
        self.bm25 = BM25()
        self.bm25.fit(self.docs_tokens)

        # TF-IDF vectorizer
        if build_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=lambda s: tokenize_text_for_ir(s),
                lowercase=False,  # already normalized/lowercased by normalizer
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.tfidf_max_features
            )
            # fit_transform on original text_norm strings
            texts = list(self.products[text_field].astype(str))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

        # build popularity if train_pairs_df is given
        if train_pairs_df is not None:
            cnt = Counter(str(x) for x in train_pairs_df['p_id'].astype(str).tolist())
            self.popularity = dict(cnt)
        else:
            self.popularity = {}

    def save(self, path: str):
        """Persist index to disk (pickle)."""
        with open(path, 'wb') as f:
            pickle.dump({
                'products': self.products,
                'docid_to_pid': self.docid_to_pid,
                'pid_to_docid': self.pid_to_docid,
                'bm25': self.bm25,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'popularity': self.popularity,
                'docs_tokens': self.docs_tokens,
                'model_tokens': self.model_tokens,
            }, f)

    @staticmethod
    def load(path: str, normalizer) -> 'ProductIndex':
        """Load index state from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        idx = ProductIndex(normalizer)
        idx.products = data['products']
        idx.docid_to_pid = data['docid_to_pid']
        idx.pid_to_docid = data['pid_to_docid']
        idx.bm25 = data['bm25']
        idx.tfidf_vectorizer = data['tfidf_vectorizer']
        idx.tfidf_matrix = data['tfidf_matrix']
        idx.popularity = data.get('popularity', {})
        idx.docs_tokens = data.get('docs_tokens', [])
        idx.model_tokens = data.get('model_tokens', [])
        return idx

    # ---------------------
    # Retrieval helpers
    # ---------------------
    def get_candidates_bm25(self, query: str, topk: int = 200) -> List[Tuple[str, float]]:
        """
        Return topk candidates (p_id, bm25_score) using BM25 over tokenized query.
        Query should be normalized text.
        """
        tokens = tokenize_text_for_ir(str(query))
        scores = self.bm25.score(tokens)
        if not scores:
            return []
        # sort by score desc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [(self.docid_to_pid[docid], float(score)) for docid, score in ranked]

    def rank_candidates_tfidf(self, query: str, candidate_pids: List[str], topk: int = 50) -> List[Tuple[str, float]]:
        """
        Given a list of candidate p_ids, compute TF-IDF cosine similarity and return ranked p_id list.
        If TF-IDF not built, returns candidate order.
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            # fallback: return with no additional ranking
            return [(pid, 0.0) for pid in candidate_pids[:topk]]
        # vectorize query
        qvec = self.tfidf_vectorizer.transform([str(query)])
        # map candidate pids to docids
        docids = [self.pid_to_docid[pid] for pid in candidate_pids if pid in self.pid_to_docid]
        if not docids:
            return []
        cand_mat = self.tfidf_matrix[docids]
        sims = cosine_similarity(qvec, cand_mat).reshape(-1)
        ranked = sorted(zip(docids, sims), key=lambda x: x[1], reverse=True)[:topk]
        return [(self.docid_to_pid[d], float(s)) for d, s in ranked]

    def retrieve_ranked_candidates(self, query: str, bm25_k: int = 500, final_k: int = 50,
                                   bm25_weight: float = 0.6, tfidf_weight: float = 0.4) -> List[Tuple[str, float, float]]:
        """
        Combined retrieval pipeline:
            1) BM25 top bm25_k
            2) TF-IDF re-rank top bm25_k
            3) Combine BM25 and TF-IDF scores by weighted sum and return top final_k
        Returns list of (p_id, bm25_score, tfidf_score)
        """
        bm25_hits = self.get_candidates_bm25(query, topk=bm25_k)
        if not bm25_hits:
            return []
        pids, bm25_scores = zip(*bm25_hits)
        tfidf_ranked = self.rank_candidates_tfidf(query, list(pids), topk=len(pids))
        # build maps
        bm25_map = dict(bm25_hits)
        tf_map = {pid: score for pid, score in tfidf_ranked}
        combined = []
        # normalize both scores (min-max) for combination
        bm_vals = np.array(list(bm25_map.values()), dtype=float)
        tf_vals = np.array([tf_map.get(pid, 0.0) for pid in pids], dtype=float)
        # avoid zero-range
        def minmax_norm(x):
            if len(x)==0: return x
            lo, hi = x.min(), x.max()
            if hi - lo < 1e-9:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)
        bm_norm = minmax_norm(bm_vals)
        tf_norm = minmax_norm(tf_vals)
        for i, pid in enumerate(pids):
            b = float(bm_norm[i]) if i < len(bm_norm) else 0.0
            t = float(tf_norm[i]) if i < len(tf_norm) else 0.0
            score = bm25_weight * b + tfidf_weight * t
            combined.append((pid, float(bm25_map.get(pid, 0.0)), float(tf_map.get(pid, 0.0)), float(score)))
        combined_sorted = sorted(combined, key=lambda x: x[3], reverse=True)[:final_k]
        return combined_sorted

# ----------------------------
# Pairwise feature builder
# ----------------------------
def _safe_get_product_field(products_df: pd.DataFrame, pid: str, field: str) -> str:
    try:
        row = products_df[products_df['p_id'].astype(str) == str(pid)]
        if len(row) == 0:
            return ""
        return str(row.iloc[0].get(field, "") or "")
    except Exception:
        return ""

def build_pair_features(query: str,
                        query_norm: str,
                        products_df: pd.DataFrame,
                        idx: Optional[ProductIndex] = None) -> pd.DataFrame:
    """
    Build features for (query, product) pairs. Returns DataFrame with:
        ['p_id', 'bm25_score', 'tfidf_score', 'brand_match', 'category_match',
         'model_match', 'token_overlap', 'token_overlap_ratio',
         'attr_overlap', 'num_query_tokens', 'num_title_tokens', 'title_contains_query',
         'title_prefix_match', 'popularity', 'numeric_token_match', 'len_query', 'len_title']
    - `query` is original query text (for some string-based checks)
    - `query_norm` must be the same normalization used in index (normalized)
    - `products_df` should be the candidate products (subset of index.products)
    - `idx` optional: if provided, bm25/tfidf scores and popularity are computed if available.
    """
    rows = []
    q_tokens = tokenize_text_for_ir(query_norm)
    q_token_set = set(q_tokens)
    q_model_tokens = extract_model_tokens(query_norm)
    q_nums = set(numeric_tokens(query_norm))
    q_len = len(query_norm)

    # For score retrieval if idx provided
    bm25_map = {}
    tf_map = {}
    if idx is not None:
        try:
            # get bm25 scores in batch via idx.bm25.score
            bm25_map = idx.bm25.score(q_tokens)
        except Exception:
            bm25_map = {}
        if idx.tfidf_vectorizer is not None and idx.tfidf_matrix is not None:
            try:
                # compute tfidf similarities for all candidate docids
                docids = [idx.pid_to_docid.get(str(pid)) for pid in products_df['p_id'].astype(str)]
                valid_pairs = [(i, d) for i, d in enumerate(docids) if d is not None]
                if valid_pairs:
                    sel_docids = [d for _, d in valid_pairs]
                    qvec = idx.tfidf_vectorizer.transform([str(query_norm)])
                    cand_mat = idx.tfidf_matrix[sel_docids]
                    sims = cosine_similarity(qvec, cand_mat).reshape(-1)
                    # map back to pid
                    for (i, d), sim in zip(valid_pairs, sims):
                        tf_map[products_df.iloc[i]['p_id']] = float(sim)
            except Exception:
                tf_map = {}

    for _, prow in products_df.reset_index(drop=True).iterrows():
        pid = str(prow.get('p_id', ''))
        title = str(prow.get('title', '') or '')
        title_norm = str(prow.get('title_norm', prow.get('text_norm', '')) or '')
        brand = str(prow.get('brand', '') or '').strip().lower()
        category = str(prow.get('category', '') or '').strip().lower()
        attrs = str(prow.get('attributes', '') or '')
        attrs_norm = str(prow.get('attributes_norm', '') or '')

        t_tokens = tokenize_text_for_ir(title_norm)
        t_token_set = set(t_tokens)
        t_model_tokens = extract_model_tokens(title_norm)
        t_nums = set(numeric_tokens(title_norm))

        # exact matches / flags
        brand_match = 1 if (brand and (brand in query_norm or brand in query.lower())) else 0
        category_match = 1 if (category and (category in query_norm or category in query.lower())) else 0

        # model token overlap
        model_match = 0
        if q_model_tokens:
            for mt in q_model_tokens:
                if mt in t_model_tokens or mt in idx.model_tokens[idx.pid_to_docid[pid]] if idx and pid in idx.pid_to_docid else False:
                    model_match = 1
                    break

        # numeric match
        numeric_match = 1 if (len(q_nums & t_nums) > 0) else 0

        # token overlaps
        overlap_tokens = q_token_set & t_token_set
        token_overlap = len(overlap_tokens)
        token_overlap_ratio = float(token_overlap) / (len(q_token_set) + 1e-9)

        # attribute overlap
        attr_tokens = set(tokenize_text_for_ir(attrs_norm))
        attr_overlap = len(q_token_set & attr_tokens)

        # title contains query phrase?
        title_contains_query = 1 if (query_norm and query_norm in title_norm) else 0
        # prefix match (first token matches)
        title_prefix_match = 0
        if q_tokens and t_tokens:
            title_prefix_match = 1 if q_tokens[0] == t_tokens[0] else 0

        # popularity
        popularity = int(idx.popularity.get(pid, 0)) if idx is not None else 0

        bm25_score = float(bm25_map.get(idx.pid_to_docid.get(pid, -1), 0.0)) if idx is not None and isinstance(bm25_map, dict) else 0.0
        # if bm25_map keyed by docid -> we convert; else if keyed by pid, attempt that too
        if isinstance(bm25_map, dict) and pid in bm25_map:
            try:
                bm25_score = float(bm25_map[pid])
            except Exception:
                pass
        tfidf_score = float(tf_map.get(pid, 0.0))

        # lengths
        len_title = len(title_norm)
        len_query = len(query_norm)

        rows.append({
            'p_id': pid,
            'bm25_score': bm25_score,
            'tfidf_score': tfidf_score,
            'brand_match': brand_match,
            'category_match': category_match,
            'model_match': model_match,
            'numeric_match': numeric_match,
            'token_overlap': token_overlap,
            'token_overlap_ratio': token_overlap_ratio,
            'attr_overlap': attr_overlap,
            'title_contains_query': title_contains_query,
            'title_prefix_match': title_prefix_match,
            'popularity': popularity,
            'num_query_tokens': len(q_token_set),
            'num_title_tokens': len(t_token_set),
            'len_query': len_query,
            'len_title': len_title
        })

    feats = pd.DataFrame(rows)
    # fill NAs with zeros and ensure numeric dtypes
    for c in feats.columns:
        if c != 'p_id':
            feats[c] = pd.to_numeric(feats[c].fillna(0), errors='coerce').fillna(0)
    return feats

# ----------------------------
# Batch helper: create features for many queries & candidate pools
# ----------------------------
def build_features_for_queries(queries_df: pd.DataFrame,
                               idx: ProductIndex,
                               topk_candidates: int = 100,
                               final_k: int = 50) -> pd.DataFrame:
    """
    For each query in queries_df (expected columns 'query_id','query','query_norm'),
    retrieve candidates and build features. Returns concatenated DataFrame with an extra 'query_id' & 'rank' columns.
    """
    out_rows = []
    for _, qrow in queries_df.iterrows():
        qid = str(qrow.get('query_id'))
        q = str(qrow.get('query', ''))
        qn = str(qrow.get('query_norm', q))
        # retrieve combined candidates
        ranked = idx.retrieve_ranked_candidates(qn, bm25_k=topk_candidates, final_k=final_k)
        # ranked is list of tuples (pid, bm25_score, tfidf_score, combined_score) OR (pid, bm25_score, tfidf_score, score)
        pids = [r[0] for r in ranked]
        # if ranked is empty, fallback to bm25 raw small set
        if not pids:
            pids = [pid for pid, _ in idx.get_candidates_bm25(qn, topk=final_k)]
        # build features for these
        cand_df = idx.products[idx.products['p_id'].isin(pids)].copy()
        # preserve order of pids
        cand_df['__order'] = cand_df['p_id'].apply(lambda x: pids.index(x) if x in pids else 9999)
        cand_df = cand_df.sort_values('__order').reset_index(drop=True)
        feats_df = build_pair_features(q, qn, cand_df, idx=idx)
        feats_df['query_id'] = qid
        # attach rank
        feats_df['rank'] = feats_df.index + 1
        out_rows.append(feats_df)
    if out_rows:
        return pd.concat(out_rows, ignore_index=True)
    else:
        return pd.DataFrame()

# ----------------------------
# Small sanity / utility functions
# ----------------------------
def topk_pids_from_ranked(ranked_list: List[Tuple[str, float, float, float]], k: int = 10) -> List[str]:
    """Pick top-k pids from ranked combined triples/list."""
    if not ranked_list:
        return []
    return [r[0] for r in ranked_list[:k]]

# ----------------------------
# End of module
# ----------------------------
