from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
import re
import json
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# tokenization helper
# -----------------------------
WORD_RE = re.compile(r"[\w؀-ۿ]+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize a string into words (keeps Persian/Arabic script and latin/digits).

    Very small, conservative tokenizer appropriate for E-Commerce text.
    """
    if text is None:
        return []
    txt = str(text)
    return WORD_RE.findall(txt)


# -----------------------------
# DataAnalyzer class
# -----------------------------
class DataAnalyzer:
    """Analyzer for product catalog and train query-product pairs.

    Parameters
    ----------
    products_df : pd.DataFrame
        DataFrame containing product catalog. Required columns: 'p_id', 'text' or 'text_norm'.
    train_pairs_df : Optional[pd.DataFrame]
        DataFrame of training pairs. Required columns if provided: 'query', 'p_id'.
    text_col : str
        Column name to use for product text for tokenization / overlap (default 'text_norm').
    """

    def __init__(self, products_df: pd.DataFrame, train_pairs_df: Optional[pd.DataFrame] = None, text_col: str = 'text_norm'):
        self.products = products_df.copy()
        self.train = train_pairs_df.copy() if train_pairs_df is not None else None
        self.text_col = text_col

        # canonical safety
        if 'p_id' in self.products.columns:
            self.products['p_id'] = self.products['p_id'].astype(str)
        if self.train is not None and 'p_id' in self.train.columns:
            self.train['p_id'] = self.train['p_id'].astype(str)

    # ---------- product corpus summaries ----------
    def summarize_products(self) -> Dict[str, Any]:
        """Return basic corpus stats.

        Returns a dict with:
          - n_products
          - avg_title_tokens, median_title_tokens
          - avg_text_chars
          - pct_with_attributes
          - sample_token_vocab_size (approx token count)
        """
        df = self.products
        n = len(df)
        # tokens from text_col
        if self.text_col not in df.columns:
            raise ValueError(f"text_col '{self.text_col}' not in products_df")

        token_counts = df[self.text_col].fillna('').map(lambda s: len(tokenize(s))).values
        chars = df['text'].fillna('').map(len).values if 'text' in df.columns else df[self.text_col].fillna('').map(len).values

        # attributes coverage
        has_attrs = 'attributes_list' in df.columns
        pct_with_attrs = 0.0
        if has_attrs:
            pct_with_attrs = (df['attributes_list'].map(lambda x: bool(x) if x is not None else False).sum() / max(1, n))

        # rough vocabulary size (sampled if very large)
        sample_texts = df[self.text_col].fillna('').astype(str)
        sample = sample_texts if n <= 5000 else sample_texts.sample(n=5000, random_state=42)
        vocab = Counter()
        for t in sample:
            vocab.update(tokenize(t))

        return {
            'n_products': int(n),
            'avg_text_tokens': float(np.mean(token_counts)) if n>0 else 0.0,
            'median_text_tokens': float(np.median(token_counts)) if n>0 else 0.0,
            'avg_text_chars': float(np.mean(chars)) if len(chars)>0 else 0.0,
            'pct_with_attributes': float(pct_with_attrs),
            'sample_vocab_size': int(len(vocab))
        }

    def top_product_tokens(self, top_k: int = 50) -> List[Tuple[str, int]]:
        """Return top-k tokens across product corpus (text_col)."""
        cnt = Counter()
        for s in self.products[self.text_col].fillna('').astype(str):
            cnt.update(tokenize(s))
        return cnt.most_common(top_k)

    def brand_distribution(self, top_k: int = 30) -> List[Tuple[str, int]]:
        if 'brand' not in self.products.columns:
            return []
        c = Counter(self.products['brand'].fillna('').astype(str).map(lambda s: s.strip()).tolist())
        return c.most_common(top_k)

    def category_distribution(self, top_k: int = 30) -> List[Tuple[str, int]]:
        if 'category' not in self.products.columns:
            return []
        c = Counter(self.products['category'].fillna('').astype(str).map(lambda s: s.strip()).tolist())
        return c.most_common(top_k)

    def attribute_frequency(self, top_k: int = 50) -> List[Tuple[str, int]]:
        if 'attributes_list' not in self.products.columns:
            return []
        cnt = Counter()
        for lst in self.products['attributes_list']:
            if not lst:
                continue
            cnt.update([str(x).strip() for x in lst if x and str(x).strip()])
        return cnt.most_common(top_k)

    # ---------- train pairs analysis ----------
    def analyze_train_pairs(self) -> Dict[str, Any]:
        """Return training-pairs summary: counts, top queries, top pids."""
        if self.train is None:
            return {}
        df = self.train
        n_pairs = len(df)
        q_counts = Counter(df['query'].fillna('').astype(str).tolist())
        p_counts = Counter(df['p_id'].fillna('').astype(str).tolist())

        # queries -> number of positive products per query
        q2p = df.groupby('query')['p_id'].nunique().sort_values(ascending=False)

        return {
            'n_pairs': int(n_pairs),
            'unique_queries': int(len(q_counts)),
            'unique_pids': int(len(p_counts)),
            'top_queries': q_counts.most_common(20),
            'top_pids': p_counts.most_common(20),
            'queries_with_multiple_pids_top20': q2p.head(20).to_dict()
        }

    def query_length_distribution(self, use_normalized: bool = True) -> Dict[str, float]:
        """Compute token-length stats for queries in train or test.

        If train DataFrame available uses train['query'] else empty.
        """
        series = None
        if self.train is not None:
            series = self.train['query']
        else:
            return {}
        tokens_len = series.fillna('').astype(str).map(lambda s: len(tokenize(s))).values
        if tokens_len.size == 0:
            return {}
        return {
            'min': int(tokens_len.min()),
            'max': int(tokens_len.max()),
            'mean': float(tokens_len.mean()),
            'median': float(np.median(tokens_len)),
            'p90': float(np.percentile(tokens_len, 90)),
            'p95': float(np.percentile(tokens_len, 95))
        }

    def query_product_lexical_overlap(self, sample_n: Optional[int] = 5000) -> Dict[str, float]:
        """Compute lexical overlap ratio between each train query and its positive product text(s).

        For each (query, p_id) pair we compute: overlap = |tokens(query) ∩ tokens(product)| / |tokens(query)|
        Returns mean/median/std of overlaps.
        If multiple pairs are present we treat each pair independently.
        """
        if self.train is None:
            return {}
        df = self.train.copy()
        # ensure p_id -> product text mapping exists
        if 'p_id' not in df.columns:
            return {}
        pid_to_text = self.products.set_index('p_id')[self.text_col].to_dict()

        overlaps = []
        # sample if large
        it = df.itertuples(index=False)
        if sample_n is not None and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42)
            it = df.itertuples(index=False)

        for row in it:
            q = getattr(row, 'query') if hasattr(row, 'query') else row[0]
            pid = getattr(row, 'p_id') if hasattr(row, 'p_id') else row[1]
            q_tokens = set(tokenize(str(q)))
            prod_text = pid_to_text.get(str(pid), '')
            prod_tokens = set(tokenize(str(prod_text)))
            if len(q_tokens) == 0:
                overlaps.append(0.0)
            else:
                inter = q_tokens.intersection(prod_tokens)
                overlaps.append(len(inter) / float(len(q_tokens)))

        arr = np.array(overlaps)
        return {
            'count_pairs': int(len(arr)),
            'mean_overlap': float(arr.mean()) if len(arr)>0 else 0.0,
            'median_overlap': float(np.median(arr)) if len(arr)>0 else 0.0,
            'std_overlap': float(arr.std()) if len(arr)>0 else 0.0,
            'p90_overlap': float(np.percentile(arr, 90)) if len(arr)>0 else 0.0
        }

    def coverage_of_train_products(self) -> Dict[str, Any]:
        """Return how many train pids exist in the product catalog.

        Useful to detect missing product entries referenced by train.
        """
        if self.train is None:
            return {}
        train_pids = set(self.train['p_id'].astype(str).tolist())
        prod_pids = set(self.products['p_id'].astype(str).tolist())
        missing = sorted(list(train_pids - prod_pids))
        return {
            'n_train_pids': int(len(train_pids)),
            'n_prod_pids': int(len(prod_pids)),
            'n_missing': int(len(missing)),
            'missing_sample': missing[:20]
        }

    # ---------- plotting helpers ----------
    def plot_query_length_distribution(self, bins: int = 20, show: bool = True):
        if self.train is None:
            raise ValueError("train data not provided")
        lens = self.train['query'].fillna('').astype(str).map(lambda s: len(tokenize(s))).values
        plt.figure(figsize=(6,3))
        plt.hist(lens, bins=bins)
        plt.xlabel('query token length')
        plt.ylabel('count')
        plt.title('Query token length distribution')
        if show:
            plt.show()

    def plot_top_brands(self, top_k: int = 20, show: bool = True):
        if 'brand' not in self.products.columns:
            raise ValueError("brand column not present in products")
        c = Counter(self.products['brand'].fillna('').astype(str).map(lambda s: s.strip()).tolist())
        items = c.most_common(top_k)
        labels, vals = zip(*items) if items else ([],[])
        plt.figure(figsize=(8,4))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=90)
        plt.ylabel('count')
        plt.title('Top brands')
        plt.tight_layout()
        if show:
            plt.show()

    # ---------- export a JSON report ----------
    def export_report(self, path: str):
        report = {}
        report['products_summary'] = self.summarize_products()
        report['top_product_tokens_50'] = self.top_product_tokens(50)
        report['brands_top_30'] = self.brand_distribution(30)
        if self.train is not None:
            report['train_summary'] = self.analyze_train_pairs()
            report['query_len_dist'] = self.query_length_distribution()
            report['lexical_overlap_sample'] = self.query_product_lexical_overlap(sample_n=5000)
            report['coverage'] = self.coverage_of_train_products()

        # write JSON (use ensure_ascii=False to keep Persian readable)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path


# -----------------------------
# Standalone helper functions
# -----------------------------

def top_k_tokens_from_texts(texts: List[str], k: int = 100) -> List[Tuple[str,int]]:
    c = Counter()
    for s in texts:
        c.update(tokenize(s))
    return c.most_common(k)


# -----------------------------
# Example usage (for developer):
# -----------------------------
from data_loader import load_train_pairs
from normalizer import PersianNormalizer
from data_preprocessing import Preprocessor
from data_analysis import DataAnalyzer
import pandas as pd

normalizer = PersianNormalizer()
pre = Preprocessor(normalizer=normalizer, map_digits=True)

# read raw CSV (keeps attributes/title)
products_raw = pd.read_csv('data/products.csv', dtype=str)
# create the normalized/augmented DataFrame (adds text_norm, attributes_list, etc.)
products_clean = pre.clean_products_df(products_raw)

train = load_train_pairs('data/train_query_product_pairs.csv')

analyzer = DataAnalyzer(products_clean, train, text_col='text_norm')
print(analyzer.summarize_products())
print(analyzer.analyze_train_pairs())
analyzer.plot_query_length_distribution()
print("\nQuery length statistics:")
print(analyzer.query_length_distribution())
analyzer.plot_top_brands()
print("\nQuery–Product lexical overlap (sampled):")
print(analyzer.query_product_lexical_overlap(sample_n=5000))
print("\nProduct corpus summary:")
print(analyzer.summarize_products())
print("\nTop 30 product tokens:")
for tok, cnt in analyzer.top_product_tokens(30):
    print(f"{tok:>15} : {cnt}")

print("\nTop brands:")
for brand, cnt in analyzer.brand_distribution(20):
    print(f"{brand:>20} : {cnt}")

train_stats = analyzer.analyze_train_pairs()

print("\nTrain pair statistics:")
print("Total pairs:", train_stats['n_pairs'])
print("Unique queries:", train_stats['unique_queries'])
print("Unique products:", train_stats['unique_pids'])

print("\nQueries with multiple positive products (top 10):")
for q, n in list(train_stats['queries_with_multiple_pids_top20'].items())[:10]:
    print(f"{q} → {n}")

print("\nTrain → Product catalog coverage:")
print(analyzer.coverage_of_train_products())

analyzer.export_report("data/analysis_report.json")
print("Saved numeric analysis report to data/analysis_report.json")


# products = pd.read_csv('data/products.csv', dtype=str)
# train = load_train_pairs('data/train_query_product_pairs.csv')
# analyzer = DataAnalyzer(products, train, text_col='text_norm')
# print(analyzer.summarize_products())
# print(analyzer.analyze_train_pairs())
# analyzer.plot_query_length_distribution()
# analyzer.plot_top_brands()

