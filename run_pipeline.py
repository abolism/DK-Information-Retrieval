import os
import sys
import pandas as pd

from normalizer import PersianNormalizer
from data_preprocessing import Preprocessor
from data_loader import load_train_pairs, load_test_queries

DATA_DIR = "data"
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_query_product_pairs.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_queries.csv")


def main():
    # files check
    for p in (PRODUCTS_CSV, TRAIN_CSV, TEST_CSV):
        if not os.path.exists(p):
            print(f"Missing file: {p}")
            sys.exit(1)

    # instantiate normalizer + preprocessor (use your implementations)
    normalizer = PersianNormalizer()
    pre = Preprocessor(normalizer=normalizer, map_digits=True, max_text_chars=1024)

    # --- Load raw products via pandas (force full raw fields to be available) ---
    # This guarantees 'attributes', 'title', etc. are present if CSV contains them.
    products_raw = pd.read_csv(PRODUCTS_CSV, dtype=str, encoding="utf-8")
    # canonicalize p_id in raw (just in case)
    if 'p_id' in products_raw.columns:
        products_raw['p_id'] = products_raw['p_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

    # load train/test via your loaders (they already canonicalize where needed)
    train_raw = load_train_pairs(TRAIN_CSV)
    test_raw = load_test_queries(TEST_CSV, normalizer)

    # --- Print raw previews and counts ---
    print("\n--- RAW SAMPLES / COUNTS ---")
    print(f"products rows: {len(products_raw)}")
    print(f"train pairs: {len(train_raw)}")
    print(f"test queries: {len(test_raw)}\n")

    print("product raw sample (head):")
    print(products_raw.head(1).to_string(index=False))

    print("\ntrain raw sample (head):")
    print(train_raw.head(5).to_string(index=False))

    print("\ntest raw sample (head):")
    print(test_raw.head(5).to_string(index=False))

    # --- Preprocess products (full cleaned set, keep raw attributes) ---
    try:
        products_clean_full = pre.clean_products_df(products_raw)
    except Exception as e:
        print(f"\nERROR: Preprocessor.clean_products_df failed: {e}")
        sys.exit(1)

    # show key preprocessed columns (if available)
    print("\n--- PREPROCESSED PRODUCTS (sample columns) ---")
    desired_cols = ['p_id', 'text_norm', 'title_norm', 'brand_norm', 'attributes_list']
    available = [c for c in desired_cols if c in products_clean_full.columns]
    print(products_clean_full[available].head(6).to_string(index=False))

    # deduplicate for indexing (we still keep full set for train mapping)
    products_dedup, dedup_stats = pre.remove_exact_duplicates(products_clean_full)
    print("\nDedup stats:", dedup_stats)
    if 'text_norm' in products_dedup.columns:
        print("products_dedup sample (p_id, text_norm):")
        print(products_dedup[['p_id', 'text_norm']].head(3).to_string(index=False))

    # --- Preprocess test queries (consistent normalization) ---
    try:
        tests_clean = pre.preprocess_queries_df(test_raw, query_col='query', id_col='query_id')
    except Exception as e:
        print(f"\nERROR: Preprocessor.preprocess_queries_df failed: {e}")
        sys.exit(1)

    print("\n--- TEST QUERIES (normalized) ---")
    print(tests_clean.head(6).to_string(index=False))

    # --- Preprocess train queries and attach product text (use full products for mapping) ---
    train_proc = train_raw.copy()
    train_proc['p_id'] = train_proc['p_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    train_proc['query_norm'] = train_proc['query'].map(lambda s: pre.normalize_text(s) if pd.notna(s) else '')

    # map using the full cleaned products (so training positives do not disappear due to dedup)
    pid_to_text_full = products_clean_full.set_index('p_id')['text_norm'].to_dict()
    train_proc['product_text_norm'] = train_proc['p_id'].map(pid_to_text_full)

    print("\n--- TRAIN (normalized + mapped) sample ---")
    display_cols = [c for c in ['query', 'query_norm', 'p_id', 'product_text_norm'] if c in train_proc.columns]
    print(train_proc[display_cols].head(6).to_string(index=False))

    # --- Sanity checks ---
    missing_products = int(train_proc['product_text_norm'].isna().sum())
    print(f"\nTraining rows whose p_id not found in product catalog: {missing_products}")
    if missing_products:
        missing_pids = sorted(list(set(train_proc.loc[train_proc['product_text_norm'].isna(), 'p_id'])))
        print("Sample missing pids (up to 20):", missing_pids[:20])
        print("\nTrain rows with missing product mappings (sample):")
        print(train_proc[train_proc['product_text_norm'].isna()].head(10).to_string(index=False))
    else:
        print("All train p_ids mapped successfully using the full product table.")

    print("\nUnique products (full):", products_clean_full['p_id'].nunique())
    print("Unique products referenced in train:", train_proc['p_id'].nunique())

    # --- Save cleaned artifacts for fast iteration later ---
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        products_clean_full.to_pickle(os.path.join(DATA_DIR, "products_clean_full.pkl"))
        products_dedup.to_pickle(os.path.join(DATA_DIR, "products_dedup.pkl"))
        print(f"\nSaved products_clean_full.pkl and products_dedup.pkl to {DATA_DIR}/")
    except Exception as e:
        print(f"\nWarning: failed to save pickles: {e}")

    print("\nPreprocessing inspection completed successfully.")


if __name__ == "__main__":
    main()