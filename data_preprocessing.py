# from typing import List, Tuple, Optional, Dict, Any
# import re
# import pandas as pd
# import pickle
# from normalizer import PersianNormalizer

# # digit maps
# _PERSIAN_DIGITS = {
#     '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
#     '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
# }
# _ARABIC_DIGITS = {
#     '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
#     '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
# }
# _DIGIT_TRANSLATION_TABLE = str.maketrans({**_PERSIAN_DIGITS, **_ARABIC_DIGITS})

# # punctuation split pattern for attributes
# _ATTR_SPLIT_RE = re.compile(r'[,\u060C\؛\n\r]+')

# class Preprocessor:
#     """
#     Central preprocessing class.

#     Parameters
#     ----------
#     normalizer : PersianNormalizer | None
#         instance of your PersianNormalizer. If None, a default instance is created.
#     map_digits : bool
#         If True, map Persian/Arabic digits to ASCII before normalization.
#     max_text_chars : int
#         Maximum characters for text fields (truncation).
#     preserve_attribute_commas : bool
#         If True, attributes' normalized text will preserve punctuation by using
#         a temporary normalizer configured to not remove punctuation.
#     """

#     def __init__(self,
#                  normalizer: Optional[PersianNormalizer] = None,
#                  map_digits: bool = True,
#                  max_text_chars: int = 1024,
#                  preserve_attribute_commas: bool = False):
#         self.normalizer = normalizer if normalizer is not None else PersianNormalizer()
#         self.map_digits_flag = map_digits
#         self.max_text_chars = max_text_chars
#         self.preserve_attribute_commas = preserve_attribute_commas

#     # ---------------------
#     # low-level helpers
#     # ---------------------
#     def map_digits_text(self, text: str) -> str:
#         """Map Persian/Arabic digits in `text` to ASCII digits."""
#         if text is None:
#             return ''
#         return str(text).translate(_DIGIT_TRANSLATION_TABLE)

#     def split_attributes(self, attr_text: str) -> List[str]:
#         """
#         Split attributes string into list of cleaned tokens.
#         Splits on comma, Arabic comma, semicolon, newline.
#         """
#         if attr_text is None:
#             return []
#         parts = _ATTR_SPLIT_RE.split(str(attr_text))
#         return [p.strip() for p in parts if p and p.strip()]

#     def normalize_text(self, text: str, remove_punct: Optional[bool] = None) -> str:
#         """
#         Normalize `text` using PersianNormalizer with optional override for punctuation removal.
#         If `remove_punct` is None, uses the normalizer's default setting.
#         """
#         if text is None:
#             return ''
#         s = str(text)
#         if self.map_digits_flag:
#             s = self.map_digits_text(s)
#         if remove_punct is None:
#             return self.normalizer.normalize(s)
#         # override punctuation removal: create a temporary normalizer
#         tmp = PersianNormalizer(remove_punct=remove_punct)
#         return tmp.normalize(s)

#     # ---------------------
#     # product-level pipeline
#     # ---------------------
#     def clean_products_df(self,
#                       products_df: pd.DataFrame,
#                       fields: Optional[List[str]] = None) -> pd.DataFrame:
#         """
#         Clean & augment product DataFrame.
    
#         Ensures columns:
#           - p_id (str)
#           - text (raw combined if needed)
#           - text_norm
#           - title_norm, category_norm, brand_norm
#           - attributes_list, attributes_norm
#         """
#         df = products_df.copy()
#         if 'p_id' not in df.columns:
#             raise ValueError("products_df must contain 'p_id' column")
#         # ensure p_id string and remove trailing .0
#         df['p_id'] = df['p_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
#         if fields is None:
#             fields = ['title', 'category', 'brand', 'attributes']
    
#         # strip whitespace from column names (robustness)
#         df.columns = [c.strip() for c in df.columns]
    
#         # Compose text if missing
#         if 'text' not in df.columns:
#             available = [f for f in fields if f in df.columns]
#             if not available:
#                 raise ValueError("No text fields available to construct 'text' column. Need one of: " + ", ".join(fields))
#             df['text'] = df.apply(lambda r: ' '.join([str(r[f]) for f in available if pd.notna(r.get(f, ''))]), axis=1)
    
#         # ensure string and truncate
#         df['text'] = df['text'].fillna('').astype(str)
#         df['text'] = df['text'].apply(lambda s: s if len(s) <= self.max_text_chars else s[:self.max_text_chars])
    
#         # --- SAFELY obtain Series for each field (so .fillna works even if column absent) ---
#         n = len(df)
#         # For each field, if present use the series, else create an empty series of same length
#         if 'title' in df.columns:
#             title_series = df['title'].astype(str)
#         else:
#             title_series = pd.Series([''] * n)
    
#         if 'category' in df.columns:
#             category_series = df['category'].astype(str)
#         else:
#             category_series = pd.Series([''] * n)
    
#         if 'brand' in df.columns:
#             brand_series = df['brand'].astype(str)
#         else:
#             brand_series = pd.Series([''] * n)
    
#         if 'attributes' in df.columns:
#             attributes_series = df['attributes'].astype(str)
#         else:
#             attributes_series = pd.Series([''] * n)
    
#         # per-field norms (use map_digits + normalizer consistently)
#         df['title_norm'] = title_series.fillna('').map(lambda s: self.normalize_text(s))
#         df['category_norm'] = category_series.fillna('').map(lambda s: self.normalize_text(s))
#         df['brand_norm'] = brand_series.fillna('').map(lambda s: self.normalize_text(s))
    
#         # attributes: keep a list and a normalized joined string
#         df['attributes_list'] = attributes_series.map(lambda s: self.split_attributes(s) if s and s.strip() else [])
#         if self.preserve_attribute_commas:
#             df['attributes_norm'] = attributes_series.fillna('').map(lambda s: self.normalize_text(s, remove_punct=False))
#         else:
#             df['attributes_norm'] = attributes_series.fillna('').map(lambda s: self.normalize_text(s))
    
#         # overall normalized text: prefer existing 'text_norm' if present, else normalize constructed text
#         if 'text_norm' in df.columns:
#             df['text_norm'] = df['text_norm'].fillna('').astype(str).map(lambda s: self.normalize_text(s))
#         else:
#             df['text_norm'] = df['text'].map(lambda s: self.normalize_text(s))
    
#         # final check: ensure no nulls and required columns exist
#         for col in ['title_norm', 'category_norm', 'brand_norm', 'attributes_norm', 'text_norm']:
#             if col not in df.columns:
#                 df[col] = ''
#         df = df.reset_index(drop=True)
#         return df

#     # ---------------------
#     # query-level pipeline
#     # ---------------------
#     def normalize_query(self, query: str) -> str:
#         """Normalize a single query using preprocessor config."""
#         return self.normalize_text(query)

#     def preprocess_queries_df(self, queries_df: pd.DataFrame, query_col: str = 'query', id_col: str = 'query_id') -> pd.DataFrame:
#         """
#         Adds 'query_id' (string) if missing and 'query_norm' normalized column.

#         Returns new DataFrame with at least ['query_id','query','query_norm'].
#         """
#         df = queries_df.copy()
#         df.columns = [c.strip() for c in df.columns]
#         if query_col not in df.columns:
#             raise ValueError(f"queries_df must contain '{query_col}' column")
#         if id_col not in df.columns:
#             df = df.reset_index().rename(columns={'index': id_col})
#         df[id_col] = df[id_col].astype(str)
#         df['query_norm'] = df[query_col].map(lambda s: self.normalize_query(s) if pd.notna(s) else '')
#         return df[[id_col, query_col, 'query_norm']]

#     # ---------------------
#     # deduplication & helpers
#     # ---------------------
#     def remove_exact_duplicates(self, products_df: pd.DataFrame, subset: str = 'text_norm') -> Tuple[pd.DataFrame, Dict[str,int]]:
#         """
#         Remove exact duplicates based on `subset` column (default 'text_norm').

#         Returns tuple (deduped_df, stats).
#         """
#         df = products_df.copy()
#         n_orig = len(df)
#         if subset not in df.columns:
#             raise ValueError(f"subset '{subset}' not in products_df columns")
#         df = df.drop_duplicates(subset=[subset], keep='first').reset_index(drop=True)
#         n_after = len(df)
#         stats = {'n_original': n_orig, 'n_after': n_after, 'n_removed': n_orig - n_after}
#         return df, stats

#     # ---------------------
#     # optional hazm stemming
#     # ---------------------
#     def hazm_stem(self, text: str) -> str:
#         """
#         If hazm is installed, apply hazm normalizer + stemmer.
#         Raises ImportError if hazm not installed.
#         """
#         try:
#             from hazm import Normalizer, Stemmer
#         except Exception as e:
#             raise ImportError("hazm not installed. Install with `pip install hazm` to use hazm_stem") from e
#         n = Normalizer()
#         s = Stemmer()
#         txt = n.normalize(str(text))
#         tokens = txt.split()
#         return ' '.join([s.stem(tok) for tok in tokens])

#     # ---------------------
#     # persistence
#     # ---------------------
#     def save(self, df: pd.DataFrame, path: str):
#         """Pickle a DataFrame to disk."""
#         with open(path, 'wb') as f:
#             pickle.dump(df, f)

#     def load(self, path: str) -> pd.DataFrame:
#         """Load a pickled DataFrame."""
#         with open(path, 'rb') as f:
#             return pickle.load(f)


# data_preprocessing.py (patched)
from typing import List, Tuple, Optional, Dict, Any
import re
import pandas as pd
import pickle
from normalizer import PersianNormalizer

# digit maps
_PERSIAN_DIGITS = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}
_ARABIC_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}
_DIGIT_TRANSLATION_TABLE = str.maketrans({**_PERSIAN_DIGITS, **_ARABIC_DIGITS})

_ATTR_SPLIT_RE = re.compile(r'[,\u060C\؛\n\r]+')

class Preprocessor:
    def __init__(self,
                 normalizer: Optional[PersianNormalizer] = None,
                 map_digits: bool = True,
                 max_text_chars: int = 1024,
                 preserve_attribute_commas: bool = False,
                 preserve_zwnj: bool = True):
        self.normalizer = normalizer if normalizer is not None else PersianNormalizer()
        self.map_digits_flag = map_digits
        self.max_text_chars = max_text_chars
        self.preserve_attribute_commas = preserve_attribute_commas
        # keep ZWNJ (U+200C) by default; set to False if you want normalization to strip them
        self.preserve_zwnj = preserve_zwnj

    def map_digits_text(self, text: str) -> str:
        if text is None:
            return ''
        return str(text).translate(_DIGIT_TRANSLATION_TABLE)

    def split_attributes(self, attr_text: str) -> List[str]:
        if attr_text is None:
            return []
        parts = _ATTR_SPLIT_RE.split(str(attr_text))
        return [p.strip() for p in parts if p and p.strip()]

    def normalize_text(self, text: str, remove_punct: Optional[bool] = None) -> str:
        if text is None:
            return ''
        s = str(text)
        if self.map_digits_flag:
            s = self.map_digits_text(s)
        # optionally strip ZWNJ only when requested
        if not self.preserve_zwnj:
            s = s.replace('\u200c', '')
        if remove_punct is None:
            return self.normalizer.normalize(s)
        tmp = PersianNormalizer(remove_punct=remove_punct)
        return tmp.normalize(s)

    def clean_products_df(self,
                          products_df: pd.DataFrame,
                          fields: Optional[List[str]] = None) -> pd.DataFrame:
        df = products_df.copy()
        if 'p_id' not in df.columns:
            raise ValueError("products_df must contain 'p_id' column")
        df['p_id'] = df['p_id'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)

        if fields is None:
            fields = ['title', 'category', 'brand', 'attributes']

        df.columns = [c.strip() for c in df.columns]

        if 'text' not in df.columns:
            available = [f for f in fields if f in df.columns]
            if not available:
                raise ValueError("No text fields available to construct 'text' column. Need one of: " + ", ".join(fields))
            df['text'] = df.apply(lambda r: ' '.join([str(r[f]) for f in available if pd.notna(r.get(f, ''))]), axis=1)

        df['text'] = df['text'].fillna('').astype(str)
        df['text'] = df['text'].apply(lambda s: s if len(s) <= self.max_text_chars else s[:self.max_text_chars])

        n = len(df)
        title_series = df['title'].astype(str) if 'title' in df.columns else pd.Series([''] * n)
        category_series = df['category'].astype(str) if 'category' in df.columns else pd.Series([''] * n)
        brand_series = df['brand'].astype(str) if 'brand' in df.columns else pd.Series([''] * n)
        attributes_series = df['attributes'].astype(str) if 'attributes' in df.columns else pd.Series([''] * n)

        df['title_norm'] = title_series.fillna('').map(lambda s: self.normalize_text(s))
        df['category_norm'] = category_series.fillna('').map(lambda s: self.normalize_text(s))
        df['brand_norm'] = brand_series.fillna('').map(lambda s: self.normalize_text(s))

        df['attributes_list'] = attributes_series.map(lambda s: self.split_attributes(s) if s and s.strip() else [])
        if self.preserve_attribute_commas:
            df['attributes_norm'] = attributes_series.fillna('').map(lambda s: self.normalize_text(s, remove_punct=False))
        else:
            df['attributes_norm'] = attributes_series.fillna('').map(lambda s: self.normalize_text(s))

        if 'text_norm' in df.columns:
            df['text_norm'] = df['text_norm'].fillna('').astype(str).map(lambda s: self.normalize_text(s))
        else:
            df['text_norm'] = df['text'].map(lambda s: self.normalize_text(s))

        for col in ['title_norm', 'category_norm', 'brand_norm', 'attributes_norm', 'text_norm']:
            if col not in df.columns:
                df[col] = ''
        df = df.reset_index(drop=True)
        # attach metadata
        df._meta = {'n_rows': len(df), 'n_unique_text_norm': df['text_norm'].nunique()}
        return df

    def normalize_query(self, query: str, strip_zwnj: Optional[bool] = None) -> str:
        """Normalize a single query using preprocessor config."""
        orig = query if query is not None else ''
        if strip_zwnj is None:
            strip_zwnj = not self.preserve_zwnj
        if strip_zwnj:
            orig = orig.replace('\u200c', '')
        return self.normalize_text(orig)

    def preprocess_queries_df(self, queries_df: pd.DataFrame, query_col: str = 'query', id_col: str = 'query_id',
                              keep_duplicates: bool = True) -> pd.DataFrame:
        df = queries_df.copy()
        df.columns = [c.strip() for c in df.columns]
        if query_col not in df.columns:
            raise ValueError(f"queries_df must contain '{query_col}' column")
        if id_col not in df.columns:
            df = df.reset_index().rename(columns={'index': id_col})
        df[id_col] = df[id_col].astype(str)
        # keep raw query and normalized query
        df['query_raw'] = df[query_col].astype(str)
        df['query_norm'] = df['query_raw'].map(lambda s: self.normalize_query(s) if pd.notna(s) else '')
        if not keep_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=['query_raw'], keep='first').reset_index(drop=True)
            # you may want to log how many removed
            df._meta = {'n_before': before, 'n_after': len(df)}
        else:
            df._meta = {'n_rows': len(df), 'n_unique_queries': df['query_raw'].nunique()}
        return df[[id_col, query_col, 'query_raw', 'query_norm']]

    def remove_exact_duplicates(self, products_df: pd.DataFrame, subset: str = 'text_norm') -> Tuple[pd.DataFrame, Dict[str,int]]:
        df = products_df.copy()
        n_orig = len(df)
        if subset not in df.columns:
            raise ValueError(f"subset '{subset}' not in products_df columns")
        df = df.drop_duplicates(subset=[subset], keep='first').reset_index(drop=True)
        n_after = len(df)
        stats = {'n_original': n_orig, 'n_after': n_after, 'n_removed': n_orig - n_after}
        return df, stats

    def hazm_stem(self, text: str) -> str:
        try:
            from hazm import Normalizer, Stemmer
        except Exception as e:
            raise ImportError("hazm not installed. Install with `pip install hazm` to use hazm_stem") from e
        n = Normalizer()
        s = Stemmer()
        txt = n.normalize(str(text))
        tokens = txt.split()
        return ' '.join([s.stem(tok) for tok in tokens])

    def save(self, df: pd.DataFrame, path: str):
        with open(path, 'wb') as f:
            pickle.dump(df, f)

    def load(self, path: str) -> pd.DataFrame:
        with open(path, 'rb') as f:
            return pickle.load(f)
