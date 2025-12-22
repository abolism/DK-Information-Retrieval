# model_training.py
"""
Model training and evaluation utilities for IR ranking.

Implements:
 - prepare_training_data: label features using train pairs and sample negatives
 - split_by_query: query-level train/validation split
 - LightGBM ranker (lambdarank) training & LightGBM classifier training
 - prediction helper to produce top-k ranked pids per query
 - evaluation: Precision@k (P@1 and P@10) and NDCG@k
 - save/load helpers

Requirements: numpy, pandas, scikit-learn, lightgbm (optional). If lightgbm not installed,
the module will try to use sklearn's LogisticRegression for classification only.

Usage example at bottom.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import json
import math
import random
import sys
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Try to import lightgbm, otherwise we'll fallback for classifier only
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -------------------------
# Helper metrics
# -------------------------
def precision_at_k_for_one(pred_list: List[str], gt_set: set, k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = pred_list[:k]
    return float(sum(1 for p in pred_k if p in gt_set) / k)

def precision_at_k(preds: Dict[str, List[str]], gt: Dict[str, set], k: int = 10) -> float:
    # preds: query_id -> ordered list of pids
    total = 0.0
    cnt = 0
    for q, p_list in preds.items():
        cnt += 1
        total += precision_at_k_for_one(p_list, gt.get(q, set()), k)
    return total / max(1, cnt)

def ndcg_at_k(pred_list: List[str], gt_set: set, k: int) -> float:
    # simple binary relevance nDCG
    dcg = 0.0
    for i, p in enumerate(pred_list[:k]):
        rel = 1.0 if p in gt_set else 0.0
        dcg += (2**rel - 1.0) / math.log2(i + 2)
    # ideal DCG: put all relevant items at top, but for binary and unknown number of relevant items, we compute IDCG up to min(len(gt), k)
    ideal_rels = [1.0] * min(len(gt_set), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2**rel - 1.0) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def ndcg_mean(preds: Dict[str, List[str]], gt: Dict[str, set], k: int = 10) -> float:
    vals = []
    for q, p_list in preds.items():
        vals.append(ndcg_at_k(p_list, gt.get(q, set()), k))
    return float(np.mean(vals)) if vals else 0.0

def normalize_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).fillna('').str.strip()
    if s.empty:
        return pd.Series([], dtype=float)
    persian_map = {
        '\u06F0':'0','\u06F1':'1','\u06F2':'2','\u06F3':'3','\u06F4':'4',
        '\u06F5':'5','\u06F6':'6','\u06F7':'7','\u06F8':'8','\u06F9':'9',
        '\u0660':'0','\u0661':'1','\u0662':'2','\u0663':'3','\u0664':'4',
        '\u0665':'5','\u0666':'6','\u0667':'7','\u0668':'8','\u0669':'9'
    }
    for k, v in persian_map.items():
        s = s.str.replace(k, v)
    # remove thousand separators heuristically, convert comma decimals -> dot
    s = s.str.replace(r'[,]\s*(?=\d{1,3}(\D|$))', '', regex=True)
    s = s.str.replace(',', '.')
    s = s.str.replace(r'[^0-9\.\-eE]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

# -------------------------
# Prepare training data
# -------------------------
def prepare_training_data(features_df: pd.DataFrame,
                          train_pairs_df: pd.DataFrame,
                          query_id_col: str = 'query_id',
                          p_id_col: str = 'p_id',
                          neg_sample_ratio: int = 4,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Label the features DataFrame using train pairs as positives.
    - features_df must contain columns: query_id, p_id and all feature columns.
    - train_pairs_df: columns ['query','p_id'] or ['query_id','p_id'].
    Returns a DataFrame identical to features_df with a new column 'label' (1/0).
    For each query we keep all positives that appear in features_df and sample up to neg_sample_ratio * positives negatives
    from the candidate pool (per query).
    """
    rng = np.random.RandomState(random_state)
    df_feats = features_df.copy()
    # unify train pairs: map query text -> query_id if necessary
    # prefer train_pairs_df having 'query_id'; else try to match on 'query' text
    if 'query_id' in train_pairs_df.columns:
        train_qid = train_pairs_df[['query_id','p_id']].copy()
    else:
        # attempt join by query text: features_df might have 'query' strings and query_id keys. We require train to have 'query' textual form.
        if 'query' in train_pairs_df.columns and 'query' in df_feats.columns and 'query_id' in df_feats.columns:
            # build mapping query -> list of pids from train
            train_qid = train_pairs_df[['query','p_id']].copy()
            # map query text to qid using first match in features_df
            q_to_qid = dict(df_feats[['query','query_id']].drop_duplicates().values.tolist())
            train_qid['query_id'] = train_qid['query'].map(lambda s: q_to_qid.get(s, None))
            train_qid = train_qid.dropna(subset=['query_id'])[['query_id','p_id']]
        else:
            # fallback: if train pairs only have p_id and features only have p_id, we will label all pid occurrences as positive (risky)
            if 'p_id' in train_pairs_df.columns:
                # mark by p_id only (all queries referencing those pids will be positive)
                train_qid = train_pairs_df[['p_id']].copy()
                train_qid['query_id'] = None
                # will treat in matching below
            else:
                raise ValueError("train_pairs_df must contain 'query_id' or 'query' (and features must have query/query_id)")

    # Build mapping: query_id -> set(positive pids)
    q_to_pos = defaultdict(set)
    if 'query_id' in train_qid.columns:
        for _, r in train_qid.iterrows():
            q = str(r['query_id'])
            p = str(r['p_id'])
            q_to_pos[q].add(p)
    else:
        # fallback: use p_id-only mapping (rare)
        pos_pids = set(train_pairs_df['p_id'].astype(str).tolist())
        # mark any candidate whose pid is in pos_pids as positive for its query
        for _, r in df_feats.iterrows():
            if str(r[p_id_col]) in pos_pids:
                q_to_pos[str(r[query_id_col])].add(str(r[p_id_col]))

    # Now label features: for each query, positives where pid in q_to_pos[q], negatives sampled from candidate pool
    df_feats['label'] = 0
    grouped = df_feats.groupby(query_id_col)
    out_frames = []
    for q, group in grouped:
        q = str(q)
        qpos = q_to_pos.get(q, set())
        group = group.copy()
        # mark positives
        if qpos:
            mask_pos = group[p_id_col].astype(str).isin(qpos)
            group.loc[mask_pos, 'label'] = 1
            n_pos = int(mask_pos.sum())
            # sample negatives
            neg_pool = group.loc[~mask_pos]
            n_neg = min(len(neg_pool), max(0, neg_sample_ratio * max(1, n_pos)))
            if n_neg > 0 and len(neg_pool) > 0:
                neg_idx = rng.choice(neg_pool.index.values, size=n_neg, replace=False)
                # set label 0 for sampled negatives; we already have zeros for others, but we'll keep only sampled negs for speed
                sampled_neg = group.loc[neg_idx]
                # union positives + sampled_neg
                keep_idx = list(group.loc[mask_pos].index) + list(sampled_neg.index)
                group = group.loc[keep_idx]
            else:
                # keep only positives (if no negatives)
                group = group.loc[mask_pos]
        else:
            # no labeled positives for this query (rare). Option: skip or keep some negatives; we'll skip such queries to avoid noise
            group = group.iloc[0:0]  # empty
        if not group.empty:
            out_frames.append(group)
    if out_frames:
        result = pd.concat(out_frames, axis=0).reset_index(drop=True)
    else:
        result = df_feats.iloc[0:0].copy()
    return result

def canonicalize_pid_col(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r'\.0$','', regex=True)

# -------------------------
# Query-level split
# -------------------------
def split_by_query(df: pd.DataFrame, query_id_col: str = 'query_id', val_frac: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Split rows by unique query ids: returns (train_df, val_df).
    Ensures queries are disjoint between splits.
    """
    qids = df[query_id_col].astype(str).unique().tolist()
    rng = np.random.RandomState(random_state)
    rng.shuffle(qids)
    n_val = max(1, int(len(qids) * val_frac))
    val_qs = set(qids[:n_val])
    mask_val = df[query_id_col].astype(str).isin(val_qs)
    val_df = df[mask_val].reset_index(drop=True)
    train_df = df[~mask_val].reset_index(drop=True)
    return train_df, val_df

# -------------------------
# Training wrappers
# -------------------------
def train_lightgbm_ranker(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          feature_cols: List[str],
                          group_col: str = 'query_id',
                          params: Optional[Dict[str, Any]] = None,
                          num_boost_round: int = 1000,
                          early_stopping_rounds: int = 50,
                          verbose_eval: int = 50):
    """
    Train LightGBM ranker (lambdarank).
    Expects train_df and val_df to contain feature_cols + group_col + 'label'.
    """
    if not HAVE_LGB:
        raise RuntimeError("LightGBM is not installed. Install it (pip install lightgbm) to train a ranker.")

    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 10],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'force_row_wise': True
        }

    # build dataset and groups
    # def build_lgb_dataset(df):
    #     X = df[feature_cols].values
    #     y = df['label'].astype(float).values
    #     # group lengths by query order
    #     group = df.groupby(group_col).size().astype(int).tolist()
    #     return lgb.Dataset(X, label=y, group=group)
    def build_lgb_dataset(df):
        df = df.sort_values(group_col).reset_index(drop=True)
        X = df[feature_cols].values
        y = df['label'].astype(float).values
        group = df.groupby(group_col, sort=False).size().astype(int).tolist()
        return lgb.Dataset(X, label=y, group=group)

    
    # sanity: ensure consistent feature order and types between train_df and full feat table
    # ensure features present
    for c in feature_cols:
        if c not in train_df.columns:
            raise RuntimeError(f"Missing feature {c} in train_df")

    # use same normalize_numeric_series as above to guarantee same preprocessing
    def _norm_df_cols(df, cols):
        df = df.copy()
        for c in cols:
            df[c] = normalize_numeric_series(df[c])
        return df

    train_df = _norm_df_cols(train_df, feature_cols)
    val_df   = _norm_df_cols(val_df, feature_cols)

    dtrain = build_lgb_dataset(train_df)
    dval = build_lgb_dataset(val_df)

    evals_result = {}
    # bst = lgb.train(params, dtrain, num_boost_round=num_boost_round,
    #                 valid_sets=[dtrain, dval], valid_names=['train','valid'],
    #                 early_stopping_rounds=early_stopping_rounds, evals_result=evals_result,
    #                 verbose_eval=verbose_eval)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=verbose_eval)
    ]

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train','valid'],
        callbacks=callbacks
        # evals_result=evals_result
    )

    # ======= POST-TRAIN QUICK CHECK (inside main after model returned) =======
    # compute train scores and simple stats to see if model learned anything
    try:
        # predict on training fragment (train_df) with same feature order
        X_train = train_df[feature_cols].values
        train_scores = model.predict(X_train, num_iteration=getattr(model, 'best_iteration', None))
        print("Train scores: min, mean, max ->", float(train_scores.min()), float(train_scores.mean()), float(train_scores.max()))
        print("Unique train score count:", len(set(train_scores.tolist())))
    except Exception as e:
        print("Post-train predict on train_df failed:", e)

    # predict on full_val_feats (we do later anyway) - quick stats
    try:
        X_val_full = full_val_feats_num[feature_cols].values
        val_scores = model.predict(X_val_full, num_iteration=getattr(model, 'best_iteration', None))
        print("Full-val scores: min, mean, max ->", float(val_scores.min()), float(val_scores.mean()), float(val_scores.max()))
        print("Unique val score count:", len(set(val_scores.tolist())))
    except Exception as e:
        print("Post-train predict on full_val_feats failed:", e)
    # ======= END POST-TRAIN CHECK =======

    return bst, evals_result

def train_lightgbm_classifier(train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              feature_cols: List[str],
                              params: Optional[Dict[str, Any]] = None,
                              num_boost_round: int = 1000,
                              early_stopping_rounds: int = 50,
                              verbose_eval: int = 50):
    """
    Train LightGBM binary classifier on labeled (q,d) pairs.
    Falls back to sklearn LogisticRegression if LightGBM not available.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1
        }

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].astype(int).values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].astype(int).values

    if HAVE_LGB:
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        evals_result = {}
        # bst = lgb.train(params, dtrain, num_boost_round=num_boost_round,
        #                 valid_sets=[dtrain, dval], valid_names=['train','valid'],
        #                 early_stopping_rounds=early_stopping_rounds, evals_result=evals_result,
        #                 verbose_eval=verbose_eval)
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=verbose_eval)
        ]

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train','valid'],
            callbacks=callbacks
            # evals_result=evals_result
        )
        return bst, evals_result
    else:
        # sklearn logistic regression as fallback
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')
        clf.fit(X_train, y_train)
        # create a simple result container
        evals_result = {'sklearn': {'trained': True}}
        return clf, evals_result

# -------------------------
# Prediction & evaluation utilities
# -------------------------
def predict_and_rank(model, feats_df: pd.DataFrame, feature_cols: List[str], model_type: str = 'ranker') -> Dict[str, List[str]]:
    """
    Given a trained model and a features DataFrame (with query_id & p_id),
    predict scores and return ordered list of pids per query_id (descending).
    model_type: 'ranker' or 'classifier' (or 'sklearn').
    Returns dict: query_id -> [p_id1, p_id2, ...] ordered by score desc.
    """
    grouped = feats_df.groupby('query_id')
    result = {}
    if HAVE_LGB and model_type == 'ranker' and isinstance(model, lgb.Booster):
        # we can compute raw scores (predict) per group
        X = feats_df[feature_cols].values
        scores = model.predict(X, num_iteration=model.best_iteration)
    elif HAVE_LGB and isinstance(model, lgb.Booster) and model_type in ('classifier','ranker'):
        X = feats_df[feature_cols].values
        scores = model.predict(X, num_iteration=model.best_iteration)
    else:
        # sklearn or object with predict_proba
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(feats_df[feature_cols].values)[:,1]
        else:
            scores = model.predict(feats_df[feature_cols].values)

    feats_df = feats_df.copy()
    feats_df['__score'] = scores
    for q, group in feats_df.groupby('query_id'):
        ordered = group.sort_values('__score', ascending=False)
        result[str(q)] = ordered['p_id'].astype(str).tolist()
    return result

# -------------------------
# Save / load helpers
# -------------------------
# def save_model(obj, path: str):
#     # Try to save lightgbm models with booster.save_model for portability
#     if HAVE_LGB and isinstance(obj, lgb.Booster):
#         obj.save_model(path)
#     else:
#         with open(path, 'wb') as f:
#             pickle.dump(obj, f)
def save_model(obj, path: str):
    # ensure directory exists
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Try to save lightgbm models with booster.save_model for portability
    if HAVE_LGB and isinstance(obj, lgb.Booster):
        obj.save_model(path)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

def load_model(path: str):
    if HAVE_LGB:
        try:
            return lgb.Booster(model_file=path)
        except Exception:
            with open(path, 'rb') as f:
                return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)

# -------------------------
# Example usage (copy/paste)
# -------------------------
if __name__ == "__main__":
    feats = pd.read_csv('data/features_train.csv', dtype=str)
    
    # robust numeric normalizer (handles Persian digits, comma decimals, thousands separators, stray chars)
    def normalize_numeric_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).fillna('').str.strip()
        if s.empty:
            return pd.Series([], dtype=float)
    
        # map Arabic/Persian digits to ASCII digits
        persian_map = {
            '\u06F0':'0','\u06F1':'1','\u06F2':'2','\u06F3':'3','\u06F4':'4',
            '\u06F5':'5','\u06F6':'6','\u06F7':'7','\u06F8':'8','\u06F9':'9',
            '\u0660':'0','\u0661':'1','\u0662':'2','\u0663':'3','\u0664':'4',
            '\u0665':'5','\u0666':'6','\u0667':'7','\u0668':'8','\u0669':'9'
        }
        for k, v in persian_map.items():
            s = s.str.replace(k, v)
    
        # Replace common thousand separators and use dot as decimal
        # e.g. "1,234.56" -> keep, "1.234,56" -> convert comma to dot, "1,234" might be thousands -> remove commas
        # Strategy: first replace non-digit decimal commas with dots, then remove stray commas
        s = s.str.replace(r'[,]\s*(?=\d{1,3}(\D|$))', '', regex=True)     # remove commas that are likely thousands separators
        s = s.str.replace(',', '.')                                       # convert remaining commas to dots
        # remove any characters except digits, dot, minus, exponent notation
        s = s.str.replace(r'[^0-9\.\-eE]', '', regex=True)
    
        return pd.to_numeric(s, errors='coerce').fillna(0.0)
    
    # find numeric candidate columns
    id_cols = ['query_id','query','p_id']
    feat_cols = [c for c in feats.columns if c not in id_cols + ['label']]
    
    # Apply robust normalization to all feature columns
    for c in feat_cols:
        feats[c] = normalize_numeric_series(feats[c])

    train_pairs = pd.read_csv('data/train_query_product_pairs.csv', dtype=str)
    # quick sanity for features
    print("Feature columns:", feat_cols[:30], " (showing up to 30)")
    print("Sample feature row (head):")
    print(feats[feat_cols].head(3).to_string(index=False))
    print("Non-zero counts per feature (quick):")
    print((feats[feat_cols] != 0).sum().sort_values(ascending=False).head(20))

    # ======= CANONICALIZE IDS BEFORE MAPPING =======
    # make p_id & query_id canonical and consistent
    def canonicalize_pid_col(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.replace(r'\.0$','', regex=True)
    
    feats['p_id'] = canonicalize_pid_col(feats['p_id'])
    train_pairs['p_id'] = canonicalize_pid_col(train_pairs['p_id'])
    # If train_pairs has query_id column, ensure its type matches feats
    if 'query_id' in train_pairs.columns:
        train_pairs['query_id'] = train_pairs['query_id'].astype(str).str.strip()
    if 'query_id' in feats.columns:
        feats['query_id'] = feats['query_id'].astype(str).str.strip()
    # also strip query text to avoid invisible whitespace issues
    if 'query' in feats.columns:
        feats['query'] = feats['query'].astype(str).str.strip()
    if 'query' in train_pairs.columns:
        train_pairs['query'] = train_pairs['query'].astype(str).str.strip()
    # ======= END CANONICALIZE =======

    labeled = prepare_training_data(feats, train_pairs, neg_sample_ratio=4)
    if labeled.empty:
        print("No labeled training rows produced. Check that features overlap training pairs.")
        sys.exit(1)

    # split by query
    # labeled = labeled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    labeled = labeled.sort_values('query_id').reset_index(drop=True)
    print(
        labeled.groupby('query_id').size().describe()
    )
    train_df, val_df = split_by_query(labeled, query_id_col='query_id', val_frac=0.1)
    # ======= QUICK SANITY BEFORE TRAINING =======
    # ensure consistent p_id format
    train_df['p_id'] = train_df['p_id'].astype(str).str.strip()
    val_df['p_id'] = val_df['p_id'].astype(str).str.strip()

    # 1) check label distribution in train and val
    print("TRAIN label counts:\n", train_df['label'].value_counts(dropna=False))
    print("VAL   label counts:\n", val_df['label'].value_counts(dropna=False))

    # 2) check number of positive queries in train/val
    train_pos_q = (train_df[train_df['label'].astype(int) == 1]['query_id'].astype(str).nunique())
    val_pos_q   = (val_df[val_df['label'].astype(int) == 1]['query_id'].astype(str).nunique())
    print(f"Queries with >=1 positive -> train: {train_pos_q}, val: {val_pos_q}")

    # 3) sample a few positive rows to inspect raw features
    print("Sample positive rows (train) head:")
    print(train_df[train_df['label'].astype(int) == 1].head(5).to_string(index=False))

    # 4) check that train/val feature columns are present and not all-zero for at least a few rows
    nonzero_counts_train = (train_df[feat_cols] != 0).sum()
    nonzero_counts_val   = (val_df[feat_cols] != 0).sum()
    print("Nonzero counts in train (per feature):")
    print(nonzero_counts_train[nonzero_counts_train > 0].sort_values(ascending=False).head(10))
    print("Nonzero counts in val (per feature):")
    print(nonzero_counts_val[nonzero_counts_val > 0].sort_values(ascending=False).head(10))

    # If we have no positives in train, abort early — this is fatal
    if train_df['label'].astype(int).sum() == 0:
        raise RuntimeError("No positive labels found in train_df — training will be useless. Check prepare_training_data mapping.")
    # ======= END SANITY =======
    val_qids = set(val_df['query_id'].astype(str))
    full_val_feats = feats[
        feats['query_id'].astype(str).isin(val_qids)
    ].copy()
    print(
        "Avg candidates per val query:",
        full_val_feats.groupby('query_id').size().mean()
    )
    feature_cols = [c for c in feat_cols]

    print("Train rows:", len(train_df), "Val rows:", len(val_df))
    if HAVE_LGB:
        model, res = train_lightgbm_ranker(train_df, val_df, feature_cols, group_col='query_id', num_boost_round=500, early_stopping_rounds=30)
        save_model(model, 'models/lgb_ranker.txt')
    else:
        model, res = train_lightgbm_classifier(train_df, val_df, feature_cols)
        save_model(model, 'models/lr_classifier.pkl')

    # for evaluation: predict on validation candidate set using features in val_df and compute P@1/P@10
    # build preds dict
    # preds = predict_and_rank(model, val_df[['query_id','p_id'] + feature_cols], feature_cols, model_type='ranker' if HAVE_LGB else 'classifier')
    
    # -------------------------
    # Diagnostics & sanity checks (NEW)
    # -------------------------
    # We will:
    # 1) verify that full_val_feats contains the positive pids from val_df
    # 2) compute model score statistics, and baseline rankings by 'tfidf_cosine' and 'bm25'
    # 3) print sample per-query tables for failing queries
    print("\n=== Running post-train diagnostics ===")

    # ensure val_df, full_val_feats, feature_cols are available here (they are above)
    # Basic presence checks
    val_qids = set(val_df['query_id'].astype(str))
    full_val_qids = set(full_val_feats['query_id'].astype(str))
    if not val_qids.issubset(full_val_qids):
        missing_q = sorted(list(val_qids - full_val_qids))[:10]
        print("WARNING: Some val query_ids are NOT present in full_val_feats (sample):", missing_q)
    else:
        print("All val query_ids present in full_val_feats.")

    # check pids presence: do any positive pids in val_df missing from full_val_feats?
    val_pos = val_df[val_df['label'].astype(int) == 1].copy()
    missing_pids = set(val_pos['p_id'].astype(str)) - set(full_val_feats['p_id'].astype(str))
    if missing_pids:
        print("WARNING: Positive p_ids missing in full candidate set (sample up to 20):", list(missing_pids)[:20])
    else:
        print("All positive p_ids from val are present in full_val_feats.")

    # Make sure feature columns align and are numeric
    for c in feature_cols:
        if c not in full_val_feats.columns:
            raise RuntimeError(f"Feature column {c} not present in full_val_feats - verification failed.")
    # convert numeric safety
    full_val_feats_num = full_val_feats.copy()
    for c in feature_cols:
        full_val_feats_num[c] = normalize_numeric_series(full_val_feats_num[c])

    # compute model scores on full set
    X_full = full_val_feats_num[feature_cols].values
    try:
        scores_full = model.predict(X_full, num_iteration=getattr(model, 'best_iteration', None))
    except Exception as e:
        # fallback to raw predict without num_iteration
        scores_full = model.predict(X_full)

    full_val_feats_num['__score'] = scores_full.astype(float)
    print("Model score stats on full validation candidates: min, mean, max ->",
          float(full_val_feats_num['__score'].min()), float(full_val_feats_num['__score'].mean()), float(full_val_feats_num['__score'].max()))
    print("Unique score count (how many distinct scores):", full_val_feats_num['__score'].nunique())

    # Baseline: tfidf_cosine and bm25 (if present)
    baselines = {}
    if 'tfidf_cosine' in full_val_feats_num.columns:
        baselines['tfidf_cosine'] = full_val_feats_num.copy().sort_values(['query_id','tfidf_cosine'], ascending=[True, False])
    if 'bm25' in full_val_feats_num.columns:
        baselines['bm25'] = full_val_feats_num.copy().sort_values(['query_id','bm25'], ascending=[True, False])

    # Build preds dicts for model and baselines
    def build_preds_df(df_scores):
        preds = {}
        for q, g in df_scores.groupby('query_id'):
            preds[str(q)] = g.sort_values('__score', ascending=False)['p_id'].astype(str).tolist()
        return preds

    preds_model = build_preds_df(full_val_feats_num[['query_id','p_id','__score'] + ([] if 'query' not in full_val_feats_num.columns else ['query'])])

    preds_baseline = {}
    for name, dfb in baselines.items():
        dfb2 = dfb.copy()
        dfb2['__score'] = dfb2[name].astype(float)
        preds_baseline[name] = build_preds_df(dfb2[['query_id','p_id','__score']])

    # Build GT mapping from val_df
    gt = defaultdict(set)
    for _, r in val_df.iterrows():
        if int(r['label']) == 1:
            gt[str(r['query_id'])].add(str(r['p_id']))

    # Compute metrics for model and baselines
    print("\nMetrics on full candidate set:")
    model_p1 = precision_at_k(preds_model, gt, k=1)
    model_p10 = precision_at_k(preds_model, gt, k=10)
    model_ndcg10 = ndcg_mean(preds_model, gt, k=10)
    print(f"Model -> P@1: {model_p1:.6f}, P@10: {model_p10:.6f}, NDCG@10: {model_ndcg10:.6f}")

    for name, pb in preds_baseline.items():
        p1b = precision_at_k(pb, gt, k=1)
        p10b = precision_at_k(pb, gt, k=10)
        ndcgb = ndcg_mean(pb, gt, k=10)
        print(f"Baseline ({name}) -> P@1: {p1b:.6f}, P@10: {p10b:.6f}, NDCG@10: {ndcgb:.6f}")

    # If model metrics are zero but baseline is >0, that points to model training/feature mismatch
    if (model_p1 == 0.0 and any(precision_at_k(pb, gt, k=1) > 0 for pb in preds_baseline.values())):
        print("DIAGNOSTIC: Baseline has >0 P@1 but model has 0 P@1 -> model is not learning/useful or feature order mismatch.")
    elif model_p1 == 0.0:
        print("DIAGNOSTIC: Both model and baselines show 0 P@1; likely GT mismatch or pos pids missing or labeling issue.")
    else:
        print("DIAGNOSTIC: Model achieves non-zero metrics on full candidate set (good).")

    # Show per-query examples where model failed but baseline succeeded (if any)
    sample_failed = []
    for q in list(gt.keys())[:200]:  # check up to first 200 val queries quickly
        model_top = preds_model.get(q, [])[:10]
        baseline_top = {name: preds_baseline[name].get(q, [])[:10] for name in preds_baseline}
        if len(gt.get(q, set())) > 0:
            if not any(p in gt[q] for p in model_top):  # model failed top10
                if any(any(p in gt[q] for p in baseline_top[name]) for name in baseline_top):  # baseline hit
                    sample_failed.append(q)
        if len(sample_failed) >= 10:
            break

    if sample_failed:
        print("\nSample queries where model misses but baseline hits (up to 10):", sample_failed[:10])
        # print detailed table for first sample
        q0 = sample_failed[0]
        dq = full_val_feats_num[full_val_feats_num['query_id'].astype(str) == str(q0)].copy()
        # merge label info
        dq = dq.merge(val_df[['query_id','p_id','label']].drop_duplicates(), left_on=['query_id','p_id'], right_on=['query_id','p_id'], how='left')
        dq['label'] = dq['label'].fillna(0).astype(int)
        print(f"\nDetailed table for failing query {q0} (top 20 by model score):")
        print(dq.sort_values('__score', ascending=False)[['p_id','label','__score'] + (feature_cols[:10] if len(feature_cols)>10 else feature_cols)].head(20).to_string(index=False))
    else:
        print("No sample-failures where baseline succeeded found in first 200 queries (or none detected).")

    print("=== Diagnostics complete ===\n")


    preds = predict_and_rank(
        model,
        full_val_feats[['query_id','p_id'] + feature_cols],
        feature_cols,
        model_type='ranker' if HAVE_LGB else 'classifier'
    )
    # build gt: mapping query->set(positives)
    gt = defaultdict(set)
    for _, r in val_df.iterrows():
        if int(r['label']) == 1:
            gt[str(r['query_id'])].add(str(r['p_id']))

    p1 = precision_at_k(preds, gt, k=1)
    p10 = precision_at_k(preds, gt, k=10)
    ndcg10 = ndcg_mean(preds, gt, k=10)
    print(f"Validation Precision@1: {p1:.4f}, Precision@10: {p10:.4f}, NDCG@10: {ndcg10:.4f}")

    # save metrics
    metrics = {'p1': p1, 'p10': p10, 'ndcg10': ndcg10}
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('Saved model and metrics.')
