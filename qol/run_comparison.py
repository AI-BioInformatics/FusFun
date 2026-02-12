#!/usr/bin/env python3
"""
run_comparison.py
=================
Compare InterPro-labeled IPR sets with FAISS-retrieved IPR sets for four peptide fields.

Overview
--------
Given two tables sharing a common fusion/event identifier, this script measures
how closely FAISS-retrieved domain accessions match those obtained from InterProScan.

Each table must contain:
  • id (or --id-col)
  • retained_domains1, lost_domains1, retained_domains2, lost_domains2
Each peptide field stores pipe-separated IPR accessions
(e.g. "IPR000719|IPR001245"). Empty values or '.' are treated as empty sets.

Workflow
--------
1) Load both input files (--events for InterPro results, --faiss for FAISS retrievals).
2) Join on the given ID column (string-cast inner join).
3) For every row × column pair, compute:
     - intersection, union, missing, extra
     - precision, recall, F1, and Jaccard index
     - whether InterPro ⊆ FAISS (all_interpro_in_faiss)
4) Save:
   • --out_rows  : detailed per-row comparisons
   • --out_stats : per-column and overall summary metrics

Outputs
-------
--out_rows  columns:
    id, col, iprs_interpro, iprs_faiss, n_ipro, n_faiss,
    n_intersection, n_union, n_missing, n_extra,
    precision, recall, f1, jaccard, all_interpro_in_faiss
--out_stats columns:
    col, n_rows, n_full_subset, pct_full_subset, n_partial_subset,
    mean_missing, mean_extra, precision_mean, recall_mean, f1_mean, jaccard_mean

Notes
-----
• IDs are cast to string before joining.
• Missing or mismatched columns trigger an error (exit code 2).
• A zero-row join triggers exit code 4.
• All statistics are computed independently for the four peptide columns and
  then averaged in the overall row.
"""

from __future__ import annotations
from pathlib import Path
import argparse, sys
import polars as pl
import re

from qol.utilities import load_file, stage_path, save_file  # CSV/TSV/.gz robust loader
from config.config import STATS

COLS = ["retained_domains1", "lost_domains1", "retained_domains2", "lost_domains2"]


_IPR_RE = re.compile(r"^IPR\d{6}$")
_PLACEHOLDERS = {".", "-", "", "NA", "N/A", "None", "null"}

def _split_iprs(cell) -> set[str]:
    """
    Parse a cell into a set of IPR accessions.

    Rules
    -----
    - Treat '.', '-', empty, and common NAs as empty tokens.
    - Accept comma or pipe separators.
    - Deduplicate.
    - Keep only tokens matching IPR\d{6}.
    """
    if cell is None:
        return set()
    s = str(cell).strip()
    if s in _PLACEHOLDERS:
        return set()

    toks = []
    for tok in s.replace(",", "|").split("|"):
        t = tok.strip()
        if not t or t in _PLACEHOLDERS:
            continue
        if _IPR_RE.match(t):
            toks.append(t)
        # else: silently drop non-IPR junk like stray hyphens
    return set(toks)



def _fmt_join(s: set[str]) -> str:
    """
    Format a set of IPR accessions back to a canonical string.

    Parameters
    ----------
    s : set[str]
        IPR set.

    Returns
    -------
    str
        'IPR0001|IPR0002' with tokens sorted, or '.' if the set is empty.
    """
    return "." if not s else "|".join(sorted(s))


def main():
    """
    CLI entry point: compare IPR sets between --events (InterPro) and --faiss.

    Steps
    -----
    1) Parse arguments (paths for events/faiss, outputs, join key).
    2) Load both tables with a robust CSV/TSV loader.
    3) Validate presence of the join column and the four peptide columns.
    4) Cast join key to string and inner-join on IDs.
    5) For each row and each peptide column:
       • Build InterPro and FAISS sets
       • Compute intersection/union, missing/extra
       • Derive precision, recall, F1, Jaccard, and subset indicator
    6) Write per-row results to --out_rows.
    7) Aggregate metrics per column and overall; write to --out_stats.

    Exits
    -----
    • Code 2 if inputs are unreadable or required columns are missing.
    • Code 4 if the join yields zero rows.
    """
    ap = argparse.ArgumentParser(description="Compare IPRs in events vs FAISS outputs for four peptide columns.")
    ap.add_argument("--ips", type=Path, required=True, help="Path to CSV/TSV file with InterPro domain sets")
    ap.add_argument("--faiss",  type=Path, required=True, help="Path to CSV/TSV file with FAISS retrieved domain sets")
    # Treat these as directories. Filenames are derived automatically.
    ap.add_argument("--out_rows",  type=Path, default=STATS, help="Output directory for per-row comparison TSV")
    ap.add_argument("--out_stats", type=Path, default=STATS, help="Output directory for aggregated stats TSV")
    ap.add_argument("--id-col", default="id", help="Join key present in both files (default: id)")
    ap.add_argument(
        "--name",
        type=str,
        help="Basename for outputs (overrides default derived from --events). Example: --name test_file",
    )
    args = ap.parse_args()

    # Load
    try:
        ev = load_file(args.events)
    except Exception as e:
        print(f"[ERROR] cannot read events: {args.events} → {e}", file=sys.stderr)
        sys.exit(2)
    try:
        fa = load_file(args.faiss)
    except Exception as e:
        print(f"[ERROR] cannot read faiss: {args.faiss} → {e}", file=sys.stderr)
        sys.exit(2)

    idc = args.id_col
    need = [idc] + COLS
    miss_ev = [c for c in need if c not in ev.columns]
    miss_fa = [c for c in need if c not in fa.columns]
    if miss_ev:
        print(f"[ERROR] events is missing columns: {miss_ev}", file=sys.stderr)
        sys.exit(2)
    if miss_fa:
        print(f"[ERROR] faiss is missing columns: {miss_fa}", file=sys.stderr)
        sys.exit(2)

    # Cast IDs to string for robust join
    ev = ev.with_columns(pl.col(idc).cast(pl.Utf8, strict=False))
    fa = fa.with_columns(pl.col(idc).cast(pl.Utf8, strict=False))

    merged = ev.join(fa.select(need), on=idc, how="inner", suffix="_faiss")
    if merged.height == 0:
        print("[ERROR] Join produced 0 rows. Check IDs overlap.", file=sys.stderr)
        sys.exit(4)
    print(f"Joined rows: {merged.height}")

    # Row-wise metrics
    out_rows = []
    for col in COLS:
        fa_col = f"{col}_faiss"
        for r in merged.iter_rows(named=True):
            idv = r[idc]
            a = _split_iprs(r.get(col))       # InterPro set
            b = _split_iprs(r.get(fa_col))    # FAISS set

            inter = a & b
            union = a | b
            missing = sorted(a - b)
            extra   = sorted(b - a)

            n_a, n_b = len(a), len(b)
            n_inter, n_union = len(inter), len(union)

            precision = (n_inter / n_b) if n_b else 0.0
            recall    = (n_inter / n_a) if n_a else 0.0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            jaccard   = (n_inter / n_union) if n_union else 0.0
            all_in    = int(len(missing) == 0)  # InterPro ⊆ FAISS

            out_rows.append({
                idc: idv,
                "col": col,
                "iprs_interpro": _fmt_join(a),
                "iprs_faiss": _fmt_join(b),
                "n_ipro": n_a,
                "n_faiss": n_b,
                "n_intersection": n_inter,
                "n_union": n_union,
                "n_missing": len(missing),
                "missing_in_faiss": "." if not missing else "|".join(missing),
                "n_extra": len(extra),
                "extra_in_faiss": "." if not extra else "|".join(extra),
                "all_interpro_in_faiss": all_in,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "jaccard": round(jaccard, 6),
            })

    df_rows = pl.DataFrame(out_rows)

    # Decide basename for outputs
    base_for_name = Path(args.name) if args.name else args.events

    # Ensure output dirs exist
    args.out_rows.mkdir(parents=True, exist_ok=True)
    args.out_stats.mkdir(parents=True, exist_ok=True)

    # Write per-row results
    df_rows_path = stage_path(args.out_rows, base_for_name, "comparison_rows", ".tsv")
    save_file(df_rows, df_rows_path)
    print(f"per-row comparison → {df_rows_path}")

    # Aggregates per column
    stats = (
        df_rows.group_by("col")
        .agg([
            pl.len().alias("n_rows"),
            pl.col("all_interpro_in_faiss").sum().alias("n_full_subset"),
            pl.col("n_missing").mean().alias("mean_missing"),
            pl.col("n_extra").mean().alias("mean_extra"),
            pl.col("precision").mean().alias("precision_mean"),
            pl.col("recall").mean().alias("recall_mean"),
            pl.col("f1").mean().alias("f1_mean"),
            pl.col("jaccard").mean().alias("jaccard_mean"),
        ])
        .with_columns([
            (pl.col("n_rows") - pl.col("n_full_subset")).alias("n_partial_subset"),
            (pl.col("n_full_subset") / pl.col("n_rows") * 100).round(2).alias("pct_full_subset"),
        ])
        .select([
            "col","n_rows","n_full_subset","pct_full_subset","n_partial_subset",
            "mean_missing","mean_extra","precision_mean","recall_mean","f1_mean","jaccard_mean"
        ])
    )

    # Overall row
    overall = (
        df_rows.select([
            pl.len().alias("n_rows"),
            pl.col("all_interpro_in_faiss").sum().alias("n_full_subset"),
            pl.col("n_missing").mean().alias("mean_missing"),
            pl.col("n_extra").mean().alias("mean_extra"),
            pl.col("precision").mean().alias("precision_mean"),
            pl.col("recall").mean().alias("recall_mean"),
            pl.col("f1").mean().alias("f1_mean"),
            pl.col("jaccard").mean().alias("jaccard_mean"),
        ])
        .with_columns([
            (pl.col("n_rows") - pl.col("n_full_subset")).alias("n_partial_subset"),
            (pl.col("n_full_subset") / pl.col("n_rows") * 100).round(2).alias("pct_full_subset"),
            pl.lit("ALL_COLUMNS").alias("col"),
        ])
        .select([
            "col","n_rows","n_full_subset","pct_full_subset","n_partial_subset",
            "mean_missing","mean_extra","precision_mean","recall_mean","f1_mean","jaccard_mean"
        ])
    )

    df_stats = pl.concat([stats, overall], how="vertical_relaxed")

    # Write aggregated stats
    df_stats_path = stage_path(args.out_stats, base_for_name, "comparison_stats", ".tsv")
    save_file(df_stats, df_stats_path)
    print(f"aggregated stats → {df_stats_path}")


if __name__ == "__main__":
    main()
