from typing import List, Optional
import pandas as pd
from normalizer import PersianNormalizer

def _combine_fields(row: pd.Series, fields: List[str]) -> str:
    """Concatenate multiple fields into a single string (skip NaNs)."""
    parts = []
    for f in fields:
        # use .get to avoid KeyError, then handle NaN
        v = row.get(f, '')
        if pd.isna(v):
            v = ''
        parts.append(str(v))
    # keep only non-empty parts and join with a space
    return ' '.join([p for p in parts if p])

def load_products(csv_path: str,
                  normalizer: PersianNormalizer,
                  text_fields: Optional[List[str]] = None,
                  encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load products.csv and produce DataFrame with:
      - p_id: string
      - text: concatenated raw fields (title + category + brand + attributes)
      - text_norm: normalized text (via normalizer.normalize)

    Parameters
    ----------
    csv_path : str
        path to products.csv
    normalizer : PersianNormalizer
        normalizer instance with `.normalize(text)` method
    text_fields : List[str], optional
        which columns to concatenate; defaults to ['title','category','brand','attributes']
    encoding : str
        file encoding (default 'utf-8')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['p_id','text','text_norm']
    """
    if text_fields is None:
        text_fields = ['title', 'category', 'brand', 'attributes']

    df = pd.read_csv(csv_path, encoding=encoding)
    # strip possible whitespace in column names (some CSVs have trailing spaces)
    df.columns = [c.strip() for c in df.columns]

    # ensure we have the expected fields; if not, fall back to available ones
    available = set(df.columns)
    missing = [f for f in text_fields if f not in available]
    if missing:
        # reduce text_fields to intersection, but keep user notified
        present = [f for f in text_fields if f in available]
        if not present:
            raise ValueError(f"No text fields from {text_fields} found in CSV columns {list(df.columns)}")
        text_fields = present

    # Convert numeric IDs like '12345.0' -> '12345'
    if 'p_id' not in df.columns:
        raise ValueError("products.csv must contain 'p_id' column")
    df['p_id'] = df['p_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    # Compose combined text
    df['text'] = df.apply(lambda r: _combine_fields(r, text_fields), axis=1)
    # Normalized text for downstream sparse/dense processing
    df['text_norm'] = df['text'].map(lambda s: normalizer.normalize(s) if pd.notna(s) else '')
    return df[['p_id', 'text', 'text_norm']]

def load_train_pairs(csv_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load training query-product pairs.

    Returns DataFrame with columns ['query', 'p_id'] where p_id is string.
    Drops rows where query is null.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = [c.strip() for c in df.columns]
    if 'query' not in df.columns or 'p_id' not in df.columns:
        raise ValueError("train_query_product_pairs.csv must contain 'query' and 'p_id' columns")
    df = df[['query', 'p_id']].copy()
    df['p_id'] = df['p_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df = df.dropna(subset=['query']).reset_index(drop=True)
    return df

def load_test_queries(csv_path: str, normalizer: PersianNormalizer, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load test queries and provide normalized query text.

    Returns DataFrame with columns ['query_id', 'query', 'query_norm'].
    If 'query_id' missing, it will be created as a string index.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = [c.strip() for c in df.columns]
    if 'query' not in df.columns:
        raise ValueError("test_queries.csv must contain a 'query' column")

    # preserve query_id column as-is if present; if not present, create one (as string)
    if 'query_id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'query_id'})
    # ensure query_id is string (helps to avoid dtype issues when saving)
    df['query_id'] = df['query_id'].astype(str)
    df['query_norm'] = df['query'].map(lambda s: normalizer.normalize(s) if pd.notna(s) else '')
    return df[['query_id', 'query', 'query_norm']]
