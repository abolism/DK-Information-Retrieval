# #!/usr/bin/env python3
# """
# Check format parity between predictions file and sample submission.

# Usage:
#     python check_submission_format.py predictions_top10.csv sample_submission.csv
# """

# import sys
# import csv
# import unicodedata
# import re
# from collections import Counter, defaultdict

# EXPECTED_PID_COLUMNS = [f"pid{i}" for i in range(1, 11)]
# EXPECTED_COL_COUNT = 1 + len(EXPECTED_PID_COLUMNS)  # query + pid1..pid10

# def read_raw(path):
#     with open(path, "rb") as f:
#         raw = f.read()
#     return raw

# def detect_bom(raw):
#     boms = {
#         "UTF-8 BOM": b"\xef\xbb\xbf",
#         "UTF-16 LE BOM": b"\xff\xfe",
#         "UTF-16 BE BOM": b"\xfe\xff",
#         "UTF-32 LE BOM": b"\xff\xfe\x00\x00",
#         "UTF-32 BE BOM": b"\x00\x00\xfe\xff",
#     }
#     found = [name for name, sig in boms.items() if raw.startswith(sig)]
#     return found

# def first_line_repr(raw, encoding="utf-8", maxchars=400):
#     try:
#         text = raw.decode(encoding)
#     except Exception:
#         text = raw.decode(encoding, errors="replace")
#     lines = text.splitlines()
#     first = lines[0] if lines else ""
#     return repr(first[:maxchars])

# def detect_delimiter(sample_text):
#     # Use csv.Sniffer to guess common delimiters; fallback to comma
#     try:
#         sniffer = csv.Sniffer()
#         dialect = sniffer.sniff(sample_text[:4096], delimiters=[",", ";", "\t", "|"])
#         return dialect.delimiter
#     except Exception:
#         # fallback heuristics
#         if "\t" in sample_text:
#             return "\t"
#         if ";" in sample_text and sample_text.count(";") > sample_text.count(","):
#             return ";"
#         return ","

# def read_table(path, delimiter=None):
#     # Read as text (utf-8 with replace to avoid crashes) and parse csv.DictReader
#     raw = read_raw(path)
#     try:
#         text = raw.decode("utf-8")
#     except Exception:
#         text = raw.decode("utf-8", errors="replace")
#     if delimiter is None:
#         delimiter = detect_delimiter(text)
#     reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
#     rows = list(reader)
#     header = reader.fieldnames
#     return header, rows, delimiter, raw

# def has_invisible_chars(s):
#     # detect zero-width / control / non-printable characters except normal whitespace
#     if s is None:
#         return False
#     for ch in s:
#         cat = unicodedata.category(ch)
#         # Cc = control, Cf = format (zero width), Cs = surrogate, Co = private use
#         if cat in ("Cc", "Cf", "Cs", "Co"):
#             # allow standard printable whitespace categories (space, newline are not in these)
#             return True
#     return False

# def non_printables_positions(s):
#     pos = []
#     for i,ch in enumerate(s):
#         cat = unicodedata.category(ch)
#         if cat in ("Cc", "Cf", "Cs", "Co"):
#             pos.append((i, ch, hex(ord(ch))))
#     return pos

# def analyze(pred_path, sample_path):
#     print(f"Reading prediction file: {pred_path}")
#     phdr, prows, pdelim, praw = read_table(pred_path)
#     print(f"  Detected delimiter: {repr(pdelim)}")
#     print(f"  Header (repr): {first_line_repr(praw)}")
#     print(f"  Parsed header columns ({len(phdr) if phdr else 0}): {phdr}")

#     print(f"\nReading sample file: {sample_path}")
#     shdr, srows, sdelim, sraw = read_table(sample_path)
#     print(f"  Detected delimiter: {repr(sdelim)}")
#     print(f"  Header (repr): {first_line_repr(sraw)}")
#     print(f"  Parsed header columns ({len(shdr) if shdr else 0}): {shdr}")

#     # BOM check
#     pboms = detect_bom(praw)
#     sboms = detect_bom(sraw)
#     print(f"\nBOMs detected: predictions={pboms or 'none'}, sample={sboms or 'none'}")

#     issues = []
#     # Header exact equality check (byte-significant): compare normalized strings
#     def normalize_header_list(hdr):
#         if hdr is None:
#             return []
#         return [h if h is not None else "" for h in hdr]

#     if normalize_header_list(phdr) != normalize_header_list(shdr):
#         issues.append("Header mismatch between prediction file and sample file.")
#         # show differences
#         print("\nHeader difference detail:")
#         print("  In predictions but not in sample:", [c for c in phdr if c not in shdr])
#         print("  In sample but not in predictions:", [c for c in shdr if c not in phdr])
#     else:
#         print("\nHeaders match exactly (string equality).")

#     # Check header contains expected columns (query + pid1..pid10)
#     if phdr is not None:
#         missing_expected = [c for c in (["query"] + EXPECTED_PID_COLUMNS) if c not in phdr]
#         extra_cols = [c for c in phdr if c not in (["query"] + EXPECTED_PID_COLUMNS)]
#         if missing_expected:
#             issues.append(f"Missing expected columns: {missing_expected}")
#         else:
#             print("All expected columns (query + pid1..pid10) are present.")
#         if extra_cols:
#             print("Extra columns detected in prediction file:", extra_cols)

#     # Row counts
#     print(f"\nRow counts: predictions={len(prows)}, sample={len(srows)}")
#     if len(prows) != len(srows):
#         issues.append("Row count differs between predictions and sample submission. (Order/rows may not match.)")

#     # Query comparisons: order sensitive and set sensitive
#     pred_queries = [ (r.get("query") or "").strip() for r in prows ]
#     sample_queries = [ (r.get("query") or "").strip() for r in srows ]

#     # Check for invisible chars in sample vs pred queries
#     invisible_in_pred_queries = [(i,q, non_printables_positions(q)) for i,q in enumerate(pred_queries) if has_invisible_chars(q)]
#     invisible_in_sample_queries = [(i,q, non_printables_positions(q)) for i,q in enumerate(sample_queries) if has_invisible_chars(q)]
#     if invisible_in_pred_queries:
#         print("\nFound invisible/control chars in predictions' queries (index, repr, positions):")
#         for item in invisible_in_pred_queries[:10]:
#             print(" ", item[0], repr(item[1]), item[2])
#         issues.append("Invisible/control characters found in some prediction queries.")
#     if invisible_in_sample_queries:
#         print("\nFound invisible/control chars in sample queries (index, repr, positions):")
#         for item in invisible_in_sample_queries[:10]:
#             print(" ", item[0], repr(item[1]), item[2])

#     # Order-sensitive comparison
#     order_mismatches = []
#     for i,(pq,sq) in enumerate(zip(pred_queries, sample_queries)):
#         if pq != sq:
#             order_mismatches.append((i, pq, sq))
#     if order_mismatches:
#         print(f"\nOrder mismatches for {len(order_mismatches)} rows (showing up to 10):")
#         for item in order_mismatches[:10]:
#             print(" ", item[0], "pred:", repr(item[1]), "sample:", repr(item[2]))
#         issues.append("Query order differs or some queries differ when compared row-by-row.")
#     else:
#         print("\nQuery ORDER and string values match exactly for rows aligned by position.")

#     # Set-sensitive check (are they the same set regardless of order)
#     pred_set = Counter(pred_queries)
#     sample_set = Counter(sample_queries)
#     only_in_pred = [q for q in pred_set if q not in sample_set]
#     only_in_sample = [q for q in sample_set if q not in pred_set]
#     dup_in_pred = [q for q,c in pred_set.items() if c>1]
#     dup_in_sample = [q for q,c in sample_set.items() if c>1]
#     if only_in_pred:
#         print(f"\nQueries present in predictions but not in sample (count {len(only_in_pred)}). Example:", only_in_pred[:5])
#         issues.append("Some queries in predictions are not present in sample.")
#     if only_in_sample:
#         print(f"\nQueries present in sample but not in predictions (count {len(only_in_sample)}). Example:", only_in_sample[:5])
#         issues.append("Some queries in sample are missing from predictions.")
#     if dup_in_pred:
#         print("\nDuplicate queries found in predictions (count):", {q:pred_set[q] for q in dup_in_pred})
#         issues.append("Duplicate query rows in predictions.")
#     if dup_in_sample:
#         print("\nDuplicate queries found in sample (count):", {q:sample_set[q] for q in dup_in_sample})

#     # Per-row pid checks
#     print("\nPer-row PID checks (will report up to first 20 problematic rows):")
#     pid_row_problems = []
#     for ridx, row in enumerate(prows):
#         q = (row.get("query") or "")
#         pids = [row.get(f"pid{i}", None) for i in range(1, 11)]
#         # Count nulls/empties
#         empty_pids = [i+1 for i,p in enumerate(pids) if p is None or str(p).strip()==""]
#         # Leading/trailing whitespace
#         ltw = [ (i+1, p) for i,p in enumerate(pids) if p is not None and (str(p) != str(p).strip()) ]
#         # Duplicates
#         stripped = ["" if p is None else str(p).strip() for p in pids]
#         duped = [p for p,c in Counter(stripped).items() if c>1 and p!=""]
#         # Non-printable chars in pids
#         invis_pid = []
#         non_digit_like = []
#         for i,p in enumerate(stripped):
#             if has_invisible_chars(p):
#                 invis_pid.append((i+1, repr(p), non_printables_positions(p)))
#             # check if pid is purely digits (many platforms use numeric ids) - we only warn
#             if p!="" and not re.match(r'^[0-9]+$', p):
#                 non_digit_like.append((i+1, p))
#         if empty_pids or ltw or duped or invis_pid or non_digit_like:
#             pid_row_problems.append({
#                 "row_index": ridx,
#                 "query": q,
#                 "empty_pids": empty_pids,
#                 "leading_trailing_ws": ltw,
#                 "duplicates_in_row": duped,
#                 "pids_with_invisibles": invis_pid,
#                 "pids_non_digit_like": non_digit_like,
#             })
#         if len(pid_row_problems) >= 20:
#             break

#     if pid_row_problems:
#         print("Found problematic PID rows (first 20):")
#         for p in pid_row_problems:
#             print(" Row", p["row_index"], "query repr:", repr(p["query"]))
#             if p["empty_pids"]:
#                 print("   Empty pid positions:", p["empty_pids"])
#             if p["leading_trailing_ws"]:
#                 print("   Leading/trailing whitespace in pid positions (pos,value):", p["leading_trailing_ws"])
#             if p["duplicates_in_row"]:
#                 print("   Duplicate pid values in this row:", p["duplicates_in_row"])
#             if p["pids_with_invisibles"]:
#                 print("   PIDs with invisible chars:", p["pids_with_invisibles"])
#             if p["pids_non_digit_like"]:
#                 print("   PIDs that are not purely digits (warn):", p["pids_non_digit_like"])
#         issues.append("Problems found in pid columns (empty/whitespace/duplicates/non-printable chars).")
#     else:
#         print("No per-row PID problems detected in the first inspected rows.")

#     # Final summary
#     print("\n=== SUMMARY ===")
#     if not issues:
#         print("PASS: Prediction file matches sample file format for all checked criteria.")
#     else:
#         print("FAIL: Found issues (total {})".format(len(issues)))
#         for it in issues:
#             print(" -", it)
#         print("\nSuggested immediate actions:")
#         print(" * Re-save predictions file as UTF-8 without BOM.")
#         print(" * Ensure header exactly: 'query,pid1,pid2,...,pid10' (no extra spaces or hidden chars).")
#         print(" * Ensure queries are identical (same text, same order) to sample submission if platform expects row-aligned submissions.")
#         print(" * Strip whitespace from all pid cells and ensure 10 unique pid columns per row.")
#         print(" * If product IDs are from a different namespace, remap them to the platform's product IDs.")
#     return issues

# if __name__ == "__main__":
#     # if len(sys.argv) < 3:
#     #     print("Usage: python check_submission_format.py <predictions.csv> <sample_submission.csv>")
#     #     sys.exit(2)
#     pred_path = "predictions_top10.csv"
#     sample_path = "data\sample_submission.csv"
#     analyze(pred_path, sample_path)







#!/usr/bin/env python3
"""
Align predictions file to the sample_submission order and report mismatches.

Usage:
    python align_predictions_to_sample.py predictions_top10.csv sample_submission.csv

Outputs:
 - predictions_aligned.csv    # same header, rows in sample order; unmatched rows have empty pid columns
 - alignment_report.txt       # human-readable report of matching stats and unmatched queries
"""
import sys
import pandas as pd
import unicodedata
import csv
from collections import defaultdict

ZWNJ = '\u200c'

def normalize_query(q):
    if q is None:
        return ""
    # Normalize unicode (NFC), strip whitespace
    qn = unicodedata.normalize("NFC", str(q)).strip()
    return qn

def remove_zwnj(q):
    return q.replace(ZWNJ, "")

def casefold(q):
    return q.casefold()

def build_match_keys(q):
    """
    Return a list of alternative keys to attempt matching in order of preference.
    """
    q0 = normalize_query(q)
    keys = []
    keys.append(("exact", q0))
    keys.append(("no_zwnj", remove_zwnj(q0)))
    keys.append(("casefold", casefold(q0)))
    keys.append(("casefold_no_zwnj", casefold(remove_zwnj(q0))))
    # further strategies could be added if needed
    return keys

def load_df(path):
    # read as strings to preserve leading zeros etc.
    df = pd.read_csv(path, dtype=str).fillna("")
    return df

def main(pred_path, sample_path):
    preds = load_df(pred_path)
    sample = load_df(sample_path)

    # Basic header check
    expected_cols = ["query"] + [f"pid{i}" for i in range(1,11)]
    missing_cols = [c for c in expected_cols if c not in preds.columns]
    if missing_cols:
        raise SystemExit(f"Predictions file missing expected columns: {missing_cols}")

    # Build index for predictions by alternative keys
    key_to_index = defaultdict(list)
    for idx, row in preds.iterrows():
        q = row.get("query", "")
        for kname, key in build_match_keys(q):
            key_to_index[(kname, key)].append(idx)
        # also store an 'any' key for exact original
        key_to_index[("orig", q)].append(idx)

    # For quick lookup, also keep a mapping for exact normalized -> idx (prefer first occurrence)
    norm_exact_map = {}
    for idx, row in preds.iterrows():
        q = row.get("query", "")
        qn = normalize_query(q)
        if qn not in norm_exact_map:
            norm_exact_map[qn] = idx

    aligned_rows = []
    report_lines = []
    unmatched_queries = []
    matched_by_strategy = defaultdict(int)
    used_pred_indices = set()

    for s_idx, srow in sample.iterrows():
        s_q = srow.get("query", "")
        matched_idx = None
        matched_strategy = None

        # Try strategies in order
        for strat, key in build_match_keys(s_q):
            candidate_indices = key_to_index.get((strat, key), [])
            if candidate_indices:
                # pick the first unused candidate index if possible
                pick = None
                for ci in candidate_indices:
                    if ci not in used_pred_indices:
                        pick = ci
                        break
                if pick is None:
                    # fallback to first one (duplicate queries in preds)
                    pick = candidate_indices[0]
                matched_idx = pick
                matched_strategy = strat
                break

        if matched_idx is not None:
            matched_by_strategy[matched_strategy] += 1
            used_pred_indices.add(matched_idx)
            aligned_rows.append(preds.loc[matched_idx, expected_cols].to_dict())
            report_lines.append(f"ROW {s_idx}: matched strategy={matched_strategy} sample_query_repr={repr(s_q)} pred_idx={matched_idx}")
        else:
            # no match found: create an empty row with the sample query so rows align
            empty_row = {"query": s_q}
            for i in range(1,11):
                empty_row[f"pid{i}"] = ""
            aligned_rows.append(empty_row)
            unmatched_queries.append(s_q)
            report_lines.append(f"ROW {s_idx}: NO MATCH FOUND for sample_query_repr={repr(s_q)}")

    # Create output DataFrame
    aligned_df = pd.DataFrame(aligned_rows, columns=expected_cols)

    # Save aligned CSV
    out_path = "predictions_aligned.csv"
    aligned_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    # Build report
    total = len(sample)
    matched_total = sum(matched_by_strategy.values())
    with open("alignment_report.txt", "w", encoding="utf-8") as rf:
        rf.write(f"Predictions file: {pred_path}\n")
        rf.write(f"Sample file: {sample_path}\n\n")
        rf.write(f"Sample rows total: {total}\n")
        rf.write(f"Predictions rows total: {len(preds)}\n\n")
        rf.write("Match counts by strategy:\n")
        for k,v in matched_by_strategy.items():
            rf.write(f"  {k}: {v}\n")
        rf.write(f"\nTotal matched: {matched_total}\n")
        rf.write(f"Total unmatched (will be empty rows in {out_path}): {len(unmatched_queries)}\n\n")
        rf.write("First 200 report lines:\n")
        for line in report_lines[:200]:
            rf.write(line + "\n")
        rf.write("\n\nUnmatched queries (first 50):\n")
        for q in unmatched_queries[:50]:
            rf.write(repr(q) + "\n")

    # Print short summary to stdout
    print("=== ALIGNMENT SUMMARY ===")
    print(f"Sample rows: {total}   Pred rows: {len(preds)}")
    print("Matched by strategy (counts):")
    for k,v in matched_by_strategy.items():
        print(f"  {k}: {v}")
    print(f"Total matched: {matched_total}")
    print(f"Total unmatched: {len(unmatched_queries)}")
    print(f"Aligned file written to: {out_path}")
    print("Detailed report written to: alignment_report.txt")
    if unmatched_queries:
        print("Note: unmatched queries listed in alignment_report.txt (you should inspect & fill these).")
    else:
        print("All sample queries found and aligned. You can re-upload predictions_aligned.csv to the platform.")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python align_predictions_to_sample.py predictions_top10.csv sample_submission.csv")
    #     sys.exit(1)
    pred_path = "predictions_top10.csv"
    sample_path = "data\sample_submission.csv"
    main(pred_path, sample_path)

