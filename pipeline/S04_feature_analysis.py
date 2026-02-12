"""
S04_feature_analysis.py
=======================
Stage 04 — Compute fusion-level feature flags from lookups and domain outputs.

Overview
--------
Consumes the Stage-02 table and derives boolean/ternary feature flags describing:
strand coherence, UTR retention, truncation, frame status, and loss of
key domain classes per partner. Optionally enriches rows with MANE matches
and gene IDs resolved from transcript IDs.

Inputs
------
• Stage-02 table with at least:
  id, gene1, gene2, transcript_id1, transcript_id2,
  chromosome1, chromosome2, breakpoint1, breakpoint2,
  retained_domains1, lost_domains1, retained_domains2, lost_domains2,
  fusion_strand1, gene_strand1, fusion_strand2, gene_strand2
  (strand columns may be missing; code handles absence)
• Lookups from config:
  - ALL_LOOKUP: transcript_id, gene_id, feature_type, chr, start, end, strand
  - MANE_LOOKUP: transcript_id, mane_transcript_id, mane_tss
  - CDS_LOOKUP: transcript_id, feature_type=='CDS', start, end, strand, length, cumulative_length

Outputs
-------
• TSV at: output/interim/features.tsv  (default; see DEFAULT_OUT)

Notes
-----
• Domain loss flags search substring matches against *_domains* columns.
• Frame flags use per-breakpoint CDS offsets modulo 3 and expose both side-specific
  and pairwise status.
• Missing data are handled defensively: absent columns default flags to 0 or 2
  as documented in the function-level docstrings.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import polars as pl

from config.config import (
    OUTPUT_DIR, ALL_LOOKUP, MANE_LOOKUP, CDS_LOOKUP,
    KINASES_KEY_DOMS, TS_KEY_DOMS, CANCER_KEY_DOMS,
    ACTIONABLE_KEY_DOMS, DRUGGABLE_KEY_DOMS,
    OUTPUT_EXT, INTERIM
)
from qol.utilities import load_file, save_file, ensure_dir, stage_path


# ------------------------- Column handling -------------------------
_SNAKE = {
    "gene1","gene2","transcript_id1","transcript_id2",
    "chromosome1","chromosome2","breakpoint1","breakpoint2",
    "fusion_strand1","fusion_strand2","gene_strand1","gene_strand2",
    "retained_domains1","lost_domains1","retained_domains2","lost_domains2",
    "5p_role","3p_role","id"
}
_CAMEL_TO_SNAKE = {
    "Gene1":"gene1","Gene2":"gene2",
    "TranscriptID1":"transcript_id1","TranscriptID2":"transcript_id2",
    "Chromosome1":"chromosome1","Chromosome2":"chromosome2",
    "Breakpoint1":"breakpoint1","Breakpoint2":"breakpoint2",
    "FusionStrand1":"fusion_strand1","FusionStrand2":"fusion_strand2",
    "GeneStrand1":"gene_strand1","GeneStrand2":"gene_strand2",
    "RetainedDomains1":"retained_domains1","RetainedDomains2":"retained_domains2",
    "LostDomains1":"lost_domains1","LostDomains2":"lost_domains2",
}

def _normalize_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize column names to snake_case expected by downstream logic.

    Parameters
    ----------
    df : pl.DataFrame
        Input table possibly mixing CamelCase and snake_case.

    Returns
    -------
    pl.DataFrame
        DataFrame with selected columns renamed using the _CAMEL_TO_SNAKE map
        or lower-cased when already present in _SNAKE.

    Notes
    -----
    Only known headers are renamed; unknown headers are left unchanged.
    """

    ren = {}
    for c in df.columns:
        if c in _CAMEL_TO_SNAKE:
            ren[c] = _CAMEL_TO_SNAKE[c]
        elif c.lower() in _SNAKE and c != c.lower():
            ren[c] = c.lower()
    if ren:
        df = df.rename(ren)
    return df


# ------------------------- Flag helper -------------------------
def _set_flag_from_ids(df: pl.DataFrame, ids_df: pl.DataFrame, flag: str, value: int = 1) -> pl.DataFrame:
    """
    Set or update a flag column for a set of id values.

    Parameters
    ----------
    df : pl.DataFrame
        Target table containing an 'id' column.
    ids_df : pl.DataFrame
        Table with an 'id' column identifying rows to flag.
    flag : str
        Name of the flag column to set. Created if missing.
    value : int, default 1
        Value to assign for matching ids.

    Returns
    -------
    pl.DataFrame
        DataFrame where 'flag' equals 'value' for ids in ids_df
        and preserves existing values elsewhere.
    """

    if ids_df.is_empty():
        return df
    if flag not in df.columns:
        df = df.with_columns(pl.lit(0).alias(flag))
    mark = ids_df.select("id").unique().with_columns(pl.lit(value).alias("_flag_set"))
    return (
        df.join(mark, on="id", how="left")
          .with_columns(
              pl.when(pl.col("_flag_set").is_not_null())
                .then(value)
                .otherwise(pl.col(flag))
                .alias(flag)
          )
          .drop("_flag_set")
    )


# ------------------------- Feature builders -------------------------
def add_metadata_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure presence of all feature flag columns with default zeros.

    Parameters
    ----------
    df : pl.DataFrame
        Input table.

    Returns
    -------
    pl.DataFrame
        Same table with all expected feature columns present.
        Adds a sequential 'id' if missing (1..N).

    Flags created
    -------------
    same_gene_fusion, same_gene_fusion_lost_dom, wrong_tr_dir_5p, wrong_tr_dir_3p,
    out_of_frame, lost_kinases_key_doms1/2, lost_ts_key_doms1/2,
    lost_cancer_key_doms1/2, lost_actionable_key_doms1/2, lost_druggable_key_doms1/2,
    truncated_5p, truncated_3p, promoter_hijacking_500, promoter_hijacking_1000,
    retained_5P_UTR1/2, retained_3P_UTR1/2, read_through.
    """

    cols = [
        "same_gene_fusion","same_gene_fusion_lost_dom",
        "wrong_tr_dir_5p","wrong_tr_dir_3p","out_of_frame",
        "lost_kinases_key_doms1","lost_kinases_key_doms2",
        "lost_ts_key_doms1","lost_ts_key_doms2",
        "lost_cancer_key_doms1","lost_cancer_key_doms2",
        "lost_actionable_key_doms1","lost_actionable_key_doms2",
        "lost_druggable_key_doms1","lost_druggable_key_doms2",
        "truncated_5p","truncated_3p",
        "promoter_hijacking_500","promoter_hijacking_1000",
        "retained_5P_UTR1","retained_3P_UTR1","retained_5P_UTR2","retained_3P_UTR2",
        "read_through"
    ]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        df = df.with_columns([pl.lit(0).alias(c) for c in miss])
    if "id" not in df.columns:
        df = df.with_columns(pl.int_range(1, df.height + 1).alias("id"))
    return df


def check_strand_coherence(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flag strand incoherence between fusion_strand* and gene_strand*.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain fusion_strand1/2 and gene_strand1/2 to operate.

    Returns
    -------
    pl.DataFrame
        With flags:
        • wrong_tr_dir_5p = 1 when fusion_strand1 != gene_strand1
        • wrong_tr_dir_3p = 1 when fusion_strand2 != gene_strand2
        • wrong_tr_dir_5p = 2 when fusion_strand1 is missing '.'
        • wrong_tr_dir_3p = 2 when fusion_strand2 is missing '.'

    Notes
    -----
    If required columns are absent, the input is returned unchanged.
    """

    if not {"fusion_strand1","gene_strand1","fusion_strand2","gene_strand2"}.issubset(df.columns):
        return df
    incoh_5 = df.filter((pl.col("fusion_strand1") != pl.col("gene_strand1")) & (pl.col("fusion_strand1") != ".")).select("id")
    df = _set_flag_from_ids(df, incoh_5, "wrong_tr_dir_5p", 1)
    incoh_3 = df.filter((pl.col("fusion_strand2") != pl.col("gene_strand2")) & (pl.col("fusion_strand2") != ".")).select("id")
    df = _set_flag_from_ids(df, incoh_3, "wrong_tr_dir_3p", 1)
    miss_5 = df.filter(pl.col("fusion_strand1") == ".").select("id")
    df = _set_flag_from_ids(df, miss_5, "wrong_tr_dir_5p", 2)
    miss_3 = df.filter(pl.col("fusion_strand2") == ".").select("id")
    df = _set_flag_from_ids(df, miss_3, "wrong_tr_dir_3p", 2)
    return df


def check_full_utr_retention(df: pl.DataFrame, gtf: pl.DataFrame, pos: int) -> pl.DataFrame:
    """
    Flag complete 5' and 3' UTR retention for one fusion partner.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with transcript_id{pos}, breakpoint{pos}, fusion_strand{pos}.
    gtf : pl.DataFrame
        Lookup with feature_type ∈ {'five_prime_utr','three_prime_utr'},
        and columns transcript_id, start, end.
    pos : int
        Partner index (1 or 2).

    Returns
    -------
    pl.DataFrame
        With retained_5P_UTR{pos} and retained_3P_UTR{pos} set to 1 when the
        breakpoint lies outside the respective UTR region per strand logic.

    Logic
    -----
    For '+' strand, retention requires breakpoint ≥ UTR.end.
    For '-' strand, retention requires breakpoint ≤ UTR.start.
    """

    if not {"transcript_id","feature_type","start","end"}.issubset(gtf.columns):
        return df
    if f"transcript_id{pos}" not in df.columns or f"breakpoint{pos}" not in df.columns or f"fusion_strand{pos}" not in df.columns:
        return df

    utr_df = gtf.filter(pl.col("feature_type").is_in(["five_prime_utr","three_prime_utr"]))
    merged = df.join(utr_df, left_on=f"transcript_id{pos}", right_on="transcript_id", how="left")

    full = merged.with_columns([
        pl.when(
            (pl.col("feature_type") == "five_prime_utr") &
            (
                ((pl.col(f"fusion_strand{pos}") == "+") & (pl.col(f"breakpoint{pos}") >= pl.col("end"))) |
                ((pl.col(f"fusion_strand{pos}") == "-") & (pl.col(f"breakpoint{pos}") <= pl.col("start")))
            )
        ).then(1).otherwise(0).alias("keep5"),
        pl.when(
            (pl.col("feature_type") == "three_prime_utr") &
            (
                ((pl.col(f"fusion_strand{pos}") == "+") & (pl.col(f"breakpoint{pos}") >= pl.col("end"))) |
                ((pl.col(f"fusion_strand{pos}") == "-") & (pl.col(f"breakpoint{pos}") <= pl.col("start")))
            )
        ).then(1).otherwise(0).alias("keep3"),
    ])
    agg = full.group_by("id").agg([pl.min("keep5").alias("k5"), pl.min("keep3").alias("k3")])

    df = _set_flag_from_ids(df, agg.filter(pl.col("k5") == 1).select("id"), f"retained_5P_UTR{pos}", 1)
    df = _set_flag_from_ids(df, agg.filter(pl.col("k3") == 1).select("id"), f"retained_3P_UTR{pos}", 1)
    return df


def update_truncation_flags(df: pl.DataFrame, gtf_df: pl.DataFrame, pos: int) -> pl.DataFrame:
    """
    Flag 5' or 3' truncation when breakpoint is before first 3'UTR base.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with transcript_id{pos}, breakpoint{pos}, fusion_strand{pos}.
    gtf_df : pl.DataFrame
        Lookup filtered to feature_type == 'three_prime_utr'.
    pos : int
        Partner index (1 or 2).

    Returns
    -------
    pl.DataFrame
        With 'truncated_5p' (pos==1) or 'truncated_3p' (pos==2) set to 1
        if breakpoint precedes UTR start on '+' or follows UTR start on '-'.

    Notes
    -----
    If required columns are missing, returns the input unchanged.
    """

    if not {"feature_type","start"}.issubset(gtf_df.columns):
        return df
    if f"fusion_strand{pos}" not in df.columns or f"breakpoint{pos}" not in df.columns or f"transcript_id{pos}" not in df.columns:
        return df

    utr3 = gtf_df.filter(pl.col("feature_type") == "three_prime_utr")
    merged = df.join(utr3, left_on=f"transcript_id{pos}", right_on="transcript_id", how="left")
    trunc = merged.filter(
        ((pl.col(f"fusion_strand{pos}") == "+") & (pl.col(f"breakpoint{pos}") < pl.col("start"))) |
        ((pl.col(f"fusion_strand{pos}") == "-") & (pl.col(f"breakpoint{pos}") > pl.col("start")))
    ).select("id").unique()
    col = f"truncated_{'5p' if pos == 1 else '3p'}"
    return _set_flag_from_ids(df, trunc, col, 1)


def add_mane_matches(df: pl.DataFrame, mane_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Attach MANE transcript matches and TSS coordinates for both partners.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with transcript_id1 and transcript_id2.
    mane_lookup : pl.DataFrame
        Columns: transcript_id, mane_transcript_id, mane_tss.

    Returns
    -------
    pl.DataFrame
        With columns: mane1, mane_tss1, mane2, mane_tss2 (nullable).
    """

    need = {"transcript_id","mane_transcript_id","mane_tss"}
    if not need.issubset(mane_lookup.columns):
        return df
    m = mane_lookup.select(["transcript_id","mane_transcript_id","mane_tss"]).unique()
    df = df.join(
        m.rename({"transcript_id":"transcript_id1","mane_transcript_id":"mane1","mane_tss":"mane_tss1"}),
        on="transcript_id1", how="left"
    )
    df = df.join(
        m.rename({"transcript_id":"transcript_id2","mane_transcript_id":"mane2","mane_tss":"mane_tss2"}),
        on="transcript_id2", how="left"
    )
    return df


def attach_gene_ids_from_tx(df: pl.DataFrame, full_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Resolve gene_id1 and gene_id2 from transcript IDs when missing.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table potentially missing gene_id1 and/or gene_id2.
    full_lookup : pl.DataFrame
        Columns: transcript_id, gene_id.

    Returns
    -------
    pl.DataFrame
        With gene_id1/gene_id2 filled where possible.
    """

    if not {"transcript_id","gene_id"}.issubset(full_lookup.columns):
        return df
    need_1 = "gene_id1" not in df.columns
    need_2 = "gene_id2" not in df.columns
    if not (need_1 or need_2):
        return df
    tx2gene = full_lookup.select(["transcript_id","gene_id"]).unique().drop_nulls()
    if need_1:
        df = df.join(
            tx2gene.rename({"transcript_id":"transcript_id1","gene_id":"gene_id1"}),
            on="transcript_id1", how="left"
        )
    if need_2:
        df = df.join(
            tx2gene.rename({"transcript_id":"transcript_id2","gene_id":"gene_id2"}),
            on="transcript_id2", how="left"
        )
    return df


def analyse_domains(df: pl.DataFrame) -> pl.DataFrame:
    """
    Set domain-loss flags based on substring matches in lost_domains* columns.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with retained_domains1/2 and lost_domains1/2.
        Role columns (5p_role, 3p_role) are optional.

    Returns
    -------
    pl.DataFrame
        With lost_*_key_doms{1,2} flags set to 1 when any configured domain
        name is found in the corresponding lost_domains* string.

    Behavior
    --------
    • If role columns are present, restrict checks to rows matching the role
    (e.g., 'Kinases', 'Suppressor', 'Actionable', 'Druggable', 'Cancer').
    • If role columns are absent, evaluate all rows.
    • Missing domain columns are created and filled with '.' before matching.
    """

    # Ensure presence
    for col in ["retained_domains1","lost_domains1","retained_domains2","lost_domains2"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(".").alias(col))

    # Role filters are optional
    def _maybe(mask_expr: pl.Expr) -> pl.DataFrame:
        try:
            return df.filter(mask_expr)
        except Exception:
            return df  # if role columns missing, act on all rows

    kin5 = _maybe(pl.col("5p_role").str.contains("(?i)Kinases", literal=False))
    kin3 = _maybe(pl.col("3p_role").str.contains("(?i)Kinases", literal=False))
    ts5  = _maybe(pl.col("5p_role").str.contains("(?i)Suppressor", literal=False))
    ts3  = _maybe(pl.col("3p_role").str.contains("(?i)Suppressor", literal=False))
    act5 = _maybe(pl.col("5p_role").str.contains("(?i)Actionable", literal=False))
    act3 = _maybe(pl.col("3p_role").str.contains("(?i)Actionable", literal=False))
    drug5= _maybe(pl.col("5p_role").str.contains("(?i)Druggable", literal=False))
    drug3= _maybe(pl.col("3p_role").str.contains("(?i)Druggable", literal=False))
    can5 = _maybe(pl.col("5p_role").str.contains("(?i)Cancer", literal=False))
    can3 = _maybe(pl.col("3p_role").str.contains("(?i)Cancer", literal=False))

    for name in KINASES_KEY_DOMS:
        df = _set_flag_from_ids(df, kin5.filter(pl.col("lost_domain_names1").str.contains(name)).select("id"), "lost_kinases_key_doms1", 1)
        df = _set_flag_from_ids(df, kin3.filter(pl.col("lost_domain_names2").str.contains(name)).select("id"), "lost_kinases_key_doms2", 1)
    for name in TS_KEY_DOMS:
        df = _set_flag_from_ids(df, ts5.filter(pl.col("lost_domain_names1").str.contains(name)).select("id"), "lost_ts_key_doms1", 1)
        df = _set_flag_from_ids(df, ts3.filter(pl.col("lost_domain_names2").str.contains(name)).select("id"), "lost_ts_key_doms2", 1)
    for name in ACTIONABLE_KEY_DOMS:
        df = _set_flag_from_ids(df, act5.filter(pl.col("lost_domain_names1").str.contains(name)).select("id"), "lost_actionable_key_doms1", 1)
        df = _set_flag_from_ids(df, act3.filter(pl.col("lost_domain_names2").str.contains(name)).select("id"), "lost_actionable_key_doms2", 1)
    for name in DRUGGABLE_KEY_DOMS:
        df = _set_flag_from_ids(df, drug5.filter(pl.col("lost_domain_names1").str.contains(name)).select("id"), "lost_druggable_key_doms1", 1)
        df = _set_flag_from_ids(df, drug3.filter(pl.col("lost_domain_names2").str.contains(name)).select("id"), "lost_druggable_key_doms2", 1)
    for name in CANCER_KEY_DOMS:
        df = _set_flag_from_ids(df, can5.filter(pl.col("lost_domain_names1").str.contains(name)).select("id"), "lost_cancer_key_doms1", 1)
        df = _set_flag_from_ids(df, can3.filter(pl.col("lost_domain_names2").str.contains(name)).select("id"), "lost_cancer_key_doms2", 1)
    return df


def compute_frame_flags(df: pl.DataFrame, cds_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Compute reading-frame status at breakpoints and pairwise frame agreement.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with transcript_id{1,2}, breakpoint{1,2}, gene_strand{1,2}.
    cds_lookup : pl.DataFrame
        CDS-only rows or mixed feature table containing:
        transcript_id, start, end, strand, length, cumulative_length.

    Returns
    -------
    pl.DataFrame
        Input joined with:
        frame1, frame2        # nt_offset % 3 at each breakpoint (Int64, nullable)
        coding_bp1, coding_bp2  # True if breakpoint falls within CDS
        in_frame, out_of_frame_pair  # pairwise status
        out_of_frame1, out_of_frame2 # side-specific: 0 in-frame, 1 out-of-frame, 2 non-coding

    Logic
    -----
    For each side, find the CDS exon covering the breakpoint. Compute nt_offset as:
    cumulative_length - length + within_exon_offset
    where within_exon_offset is bp - start on '+' or end - bp on '-'.
    The frame is nt_offset % 3.
    """

    cds = cds_lookup
    if "feature_type" in cds.columns:
        cds = cds.filter(pl.col("feature_type") == "CDS")
    need = {"transcript_id","start","end","strand","length","cumulative_length"}
    if not need.issubset(cds.columns):
        return df

    def per_side(pos: int) -> pl.DataFrame:
        tcol, bcol, scol = f"transcript_id{pos}", f"breakpoint{pos}", f"gene_strand{pos}"
        if not {tcol, bcol}.issubset(df.columns):
            # no data → mark non-coding
            return df.select("id").unique().with_columns([
                pl.lit(None).cast(pl.Int64).alias(f"frame{pos}"),
                pl.lit(False).alias(f"coding_bp{pos}")
            ])
        within = (
            df.join(cds, left_on=tcol, right_on="transcript_id", how="left")
              .filter((pl.col("start") <= pl.col(bcol)) & (pl.col("end") >= pl.col(bcol)))
              .with_columns(pl.col(scol).fill_null("+").alias("_strand"))
              .with_columns(
                  pl.when(pl.col("_strand") == "+")
                    .then(pl.col(bcol) - pl.col("start"))
                    .otherwise(pl.col("end") - pl.col(bcol))
                    .alias("_within_exon")
              )
              .with_columns(
                  (pl.col("cumulative_length") - pl.col("length") + pl.col("_within_exon")).alias("_nt_offset")
              )
              .with_columns(
                  (pl.col("_nt_offset") % 3).cast(pl.Int64).alias(f"frame{pos}")
              )
              .select(["id", f"frame{pos}"])
              .unique(subset=["id"], keep="first")
        )
        return (
            df.select("id").unique()
              .join(within, on="id", how="left")
              .with_columns(pl.col(f"frame{pos}").is_not_null().alias(f"coding_bp{pos}"))
        )

    j1 = per_side(1)
    j2 = per_side(2)

    tmp = (
        df.join(j1, on="id", how="left")
          .join(j2, on="id", how="left")
          .with_columns([
              (pl.col("coding_bp1") & pl.col("coding_bp2") & (pl.col("frame1") == pl.col("frame2"))).alias("in_frame"),
              (pl.col("coding_bp1") & pl.col("coding_bp2") & (pl.col("frame1") != pl.col("frame2"))).alias("out_of_frame_pair"),
          ])
          .with_columns([
              pl.when(~pl.col("coding_bp1")).then(2)
                .when(pl.col("out_of_frame_pair")).then(1)
                .when(pl.col("in_frame")).then(0)
                .otherwise(2).cast(pl.Int8).alias("out_of_frame1"),
              pl.when(~pl.col("coding_bp2")).then(2)
                .when(pl.col("out_of_frame_pair")).then(1)
                .when(pl.col("in_frame")).then(0)
                .otherwise(2).cast(pl.Int8).alias("out_of_frame2"),
          ])
    )
    return tmp


def compute_features(df: pl.DataFrame, full_lookup: pl.DataFrame, mane_lookup: pl.DataFrame, cds_lookup: pl.DataFrame) -> pl.DataFrame:
    """
    Run the full feature computation pipeline on a fusion table.

    Parameters
    ----------
    df : pl.DataFrame
        Stage-02 table.
    full_lookup : pl.DataFrame
        ALL_LOOKUP table.
    mane_lookup : pl.DataFrame
        MANE_LOOKUP table.
    cds_lookup : pl.DataFrame
        CDS_LOOKUP table.

    Returns
    -------
    pl.DataFrame
        Input table augmented with feature flags and annotations.

    Steps
    -----
    1) Limit GTF to UTR rows when available.
    2) Attach gene IDs from transcripts.
    3) Strand coherence flags.
    4) UTR retention flags for both partners.
    5) Truncation flags for both partners.
    6) MANE matches.
    7) Domain-loss flags by class.
    8) Frame flags and pairwise frame status.
    """

    utr_gtf = full_lookup.filter(pl.col("feature_type").is_in(["five_prime_utr","three_prime_utr"])) if "feature_type" in full_lookup.columns else full_lookup
    df = attach_gene_ids_from_tx(df, full_lookup)
    df = check_strand_coherence(df)
    df = check_full_utr_retention(df, utr_gtf, 1)
    df = check_full_utr_retention(df, utr_gtf, 2)
    df = update_truncation_flags(df, utr_gtf, 1)
    df = update_truncation_flags(df, utr_gtf, 2)
    df = add_mane_matches(df, mane_lookup)
    df = analyse_domains(df)
    df = compute_frame_flags(df, cds_lookup)
    return df


# ---------------- Orchestration ----------------

def run(input_path: Path) -> Path:
    """
Entry point: load inputs, compute features, and write default output.

Parameters
----------
input_path : Path
    Stage-02 TSV path.

Returns
-------
Path
    Path to DEFAULT_OUT (output/interim/features.tsv).

Side Effects
------------
Reads: input_path, ALL_LOOKUP, MANE_LOOKUP, CDS_LOOKUP.
Writes: DEFAULT_OUT under output/interim.
"""

    df = load_file(input_path)
    df = _normalize_cols(df)
    df = add_metadata_cols(df)

    full_lookup = load_file(ALL_LOOKUP)
    mane_lookup = load_file(MANE_LOOKUP)
    cds_lookup  = load_file(CDS_LOOKUP)

    out = compute_features(df, full_lookup, mane_lookup, cds_lookup)

    out_path = stage_path(INTERIM, input_path, "features", ".tsv")
    ensure_dir(out_path.parent)
    save_file(out, out_path)
    return out_path


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        raise SystemExit("Usage: S04_feature_analysis.py <stage_02_tsv>")

    print(run(Path(sys.argv[1])))

