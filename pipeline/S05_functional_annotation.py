"""
S05_functional_annotation.py
============================
Stage 05 — Event-level functional annotation of gene fusions.

Overview
--------
Computes event flags on top of Stage-04 features and upstream metadata:
• same_gene_fusion and same_gene_fusion_lost_dom
• read_through (strict) and read_through_loose
• promoter_hijacking_* variants (promoter, UTR, combined, 500/1000 windows)
• true_fusion_{TH} based on expression coverage thresholds (optional)

Inputs
------
• Stage-04/02 table with at least:
  id, gene1, gene2, transcript_id1/2, chromosome1/2, breakpoint1/2,
  gene_strand1/2, coverage1/2 (coverage only if true_fusion_* is desired)
• Lookups from config:
  - ALL_LOOKUP: full gene/transcript features
  - NEXT_GENE_LOOKUP: per-gene adjacency and TSS/TES anchors for neighbors
• Thresholds and windows from config.config:
  RT_MAX_GAP, RT_MAX_GENE_GAP, RT_BP_WINDOW, RT_USE_TX_ANCHORS, RT_PROMOTE_LOOSE,
  PH_WIN_500, PH_WIN_1000, PH_PROM_WIN, EXPRESSION_COVERAGE_THRESHOLDS

Outputs
-------
• INTERIM/<base>_annotated.tsv       — input rows + event flags
• INTERIM/<base>_event_counts.tsv    — counts by event category

Notes
-----
• "Strict" read-through checks adjacency and bp gap. "Loose" version uses gene-gene
  window plus breakpoint windows and can optionally promote to strict.
• true_fusion_{TH} requires both coverage1 and coverage2 > TH and no exclusion flags
  (same_gene, read_through, promoter_hijacking_500/1000).
"""


from __future__ import annotations
from pathlib import Path
import re
import polars as pl

from qol.utilities import load_file, save_file, ensure_dir, stage_path
from config.config import (
    INTERIM, ALL_LOOKUP, NEXT_GENE_LOOKUP,
    EXPRESSION_COVERAGE_THRESHOLDS as CFG_THRESH,
    RT_MAX_GAP, RT_MAX_GENE_GAP, RT_BP_WINDOW,
    RT_USE_TX_ANCHORS, RT_PROMOTE_LOOSE,
    PH_WIN_500, PH_WIN_1000, PH_PROM_WIN, FINAL
)

# -------------------------
# Helpers
# -------------------------
def _set_flag_from_ids(df: pl.DataFrame, ids_df: pl.DataFrame, flag: str, value: int = 1) -> pl.DataFrame:
    """
    Set a flag column to a value for a list of ids, creating the column if missing.

    Parameters
    ----------
    df : pl.DataFrame
        Target table with an 'id' column.
    ids_df : pl.DataFrame
        DataFrame with an 'id' column listing rows to update.
    flag : str
        Flag column name to set.
    value : int, default 1
        Value to assign for matching ids.

    Returns
    -------
    pl.DataFrame
        Updated DataFrame where 'flag' equals 'value' for ids in ids_df
        and is preserved elsewhere.
    """

    if flag not in df.columns:
        df = df.with_columns(pl.lit(0).alias(flag))
    if ids_df.is_empty():
        return df
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

def _pick(colnames: list[str], *candidates: str) -> str | None:
    """
    Set a flag column to a value for a list of ids, creating the column if missing.

    Parameters
    ----------
    df : pl.DataFrame
        Target table with an 'id' column.
    ids_df : pl.DataFrame
        DataFrame with an 'id' column listing rows to update.
    flag : str
        Flag column name to set.
    value : int, default 1
        Value to assign for matching ids.

    Returns
    -------
    pl.DataFrame
        Updated DataFrame where 'flag' equals 'value' for ids in ids_df
        and is preserved elsewhere.
    """

    for c in candidates:
        if c in colnames:
            return c
    return None

def _norm_str_col(s: pl.Expr) -> pl.Expr:
    """
    Normalize a string expression: cast, strip spaces, remove inner spaces, uppercase.

    Parameters
    ----------
    s : pl.Expr
        String expression.

    Returns
    -------
    pl.Expr
        Normalized uppercase string without spaces.
    """

    return (s.cast(pl.Utf8)
             .str.strip_chars()
             .str.replace_all(" ", "")
             .str.to_uppercase())

def _base_stem(p: Path) -> str:
    """
    Derive the base stem of an input file by removing common pipeline suffixes.

    Parameters
    ----------
    p : Path
        Input path (e.g., 'sample_features.tsv').

    Returns
    -------
    str
        Base stem with known suffixes removed (e.g., 'sample').
    """

    s = p.stem
    patterns = (
        r"(?:_features)$",
        r"(?:_domains_full)$",
        r"(?:_domains)$",
        r"(?:_ips_full)$",
        r"(?:_ips)$",
        r"(?:_peptides)$",
        r"(?:_peptides_domains_full)$",
        r"(?:_peptides_domains)$",
    )
    for pat in patterns:
        if re.search(pat, s):
            return re.sub(pat, "", s)
    return s

# -------------------------
# Rules
# -------------------------
def check_same_gene(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flag same-gene fusions and same-gene fusions with any lost domain.

    Behavior
    --------
    • same_gene_fusion = 1 when gene1 == gene2.
    • same_gene_fusion_lost_dom = 1 for same-gene rows where either
    lost_domains1 or lost_domains2 is not '.'.

    Parameters
    ----------
    df : pl.DataFrame

    Returns
    -------
    pl.DataFrame
        DataFrame with the two flags ensured and updated.
    """

    """same_gene_fusion = 1 if gene1 == gene2; same_gene_fusion_lost_dom = 1 if any lost_domains in same-gene rows."""
    for c in ("same_gene_fusion", "same_gene_fusion_lost_dom"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))
    sg = df.filter(pl.col("gene1") == pl.col("gene2")).select("id")
    df = _set_flag_from_ids(df, sg, "same_gene_fusion", 1)
    for col in ("lost_domains1", "lost_domains2"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(".").alias(col))
    sg_loss = df.filter(
        (pl.col("gene1") == pl.col("gene2")) &
        ((pl.col("lost_domains1") != ".") | (pl.col("lost_domains2") != "."))
    ).select("id")
    df = _set_flag_from_ids(df, sg_loss, "same_gene_fusion_lost_dom", 1)
    return df

def evaluate_promoter_hijacking(
    df: pl.DataFrame,
    gtf_df: pl.DataFrame,
    *,
    ph_win_500: int,
    ph_win_1000: int,
    ph_prom_win: int,
) -> pl.DataFrame:
    """
    Flag promoter hijacking based on distance to TSS and UTR positioning.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with transcript_id1 and breakpoint1.
    gtf_df : pl.DataFrame
        ALL_LOOKUP-like table with 'transcript' and 'CDS' features.
    ph_win_500 : int
        Upstream window size for promoter_hijacking_500.
    ph_win_1000 : int
        Upstream window size for promoter_hijacking_1000.
    ph_prom_win : int
        Upstream window for promoter_hijacking_prom.

    Returns
    -------
    pl.DataFrame
        With flags:
        promoter_hijacking_500 / _1000
        promoter_hijacking_prom
        promoter_hijacking_utr
        promoter_hijacking (combined prom ∪ utr)

    Logic
    -----
    • Compute transcript-anchored TSS and CDS start per strand.
    • Upstream distance = signed distance from bp1 to TSS on the '+' direction.
    • UTR flag when bp1 lies between TSS and CDS start per strand conventions.
    """

    for c in ("promoter_hijacking_prom","promoter_hijacking_utr","promoter_hijacking",
              "promoter_hijacking_500","promoter_hijacking_1000"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))

    tx = (
        gtf_df.filter(pl.col("feature_type") == "transcript")
              .select(["transcript_id","start","end","strand"])
              .unique("transcript_id")
              .with_columns(
                  pl.when(pl.col("strand")=="+").then(pl.col("start")).otherwise(pl.col("end")).alias("tss_tx")
              )
              .select(["transcript_id","tss_tx","strand"])
    )
    cds = gtf_df.filter(pl.col("feature_type")=="CDS").select(["transcript_id","start","end","strand"])
    cds_start = (
        cds.group_by(["transcript_id","strand"])
           .agg([pl.min("start").alias("min_start"),pl.max("end").alias("max_end")])
           .with_columns(
               pl.when(pl.col("strand")=="+").then(pl.col("min_start")).otherwise(pl.col("max_end")).alias("cds_start_tx")
           )
           .select(["transcript_id","cds_start_tx"])
    )

    j = (
        df.join(tx.rename({"transcript_id":"transcript_id1","strand":"t1_strand"}), on="transcript_id1", how="left")
          .join(cds_start.rename({"transcript_id":"transcript_id1"}), on="transcript_id1", how="left")
          .with_columns([
              pl.col("breakpoint1").cast(pl.Int64, strict=False).alias("bp1_i"),
              pl.col("tss_tx").cast(pl.Int64, strict=False).alias("tss_i"),
              pl.col("cds_start_tx").cast(pl.Int64, strict=False).alias("cds_i"),
          ])
    )

    upstream = pl.when(pl.col("t1_strand")=="+").then(pl.col("tss_i")-pl.col("bp1_i")).otherwise(pl.col("bp1_i")-pl.col("tss_i"))
    j = j.with_columns(upstream.alias("_upstream"))

    hij500_ids  = j.filter((pl.col("_upstream")>0)&(pl.col("_upstream")<=ph_win_500)).select("id").unique()
    hij1000_ids = j.filter((pl.col("_upstream")>0)&(pl.col("_upstream")<=ph_win_1000)).select("id").unique()
    prom_ids    = j.filter((pl.col("_upstream")>0)&(pl.col("_upstream")<=ph_prom_win)).select("id").unique()

    utr_pos = pl.when(pl.col("t1_strand")=="+") \
                .then((pl.col("bp1_i")>=pl.col("tss_i"))&(pl.col("bp1_i")<pl.col("cds_i"))) \
                .otherwise((pl.col("bp1_i")<=pl.col("tss_i"))&(pl.col("bp1_i")>pl.col("cds_i")))
    utr_ids = j.filter(pl.col("tss_i").is_not_null()&pl.col("cds_i").is_not_null()&utr_pos).select("id").unique()

    df = _set_flag_from_ids(df,hij500_ids,"promoter_hijacking_500",1)
    df = _set_flag_from_ids(df,hij1000_ids,"promoter_hijacking_1000",1)
    df = _set_flag_from_ids(df,prom_ids,"promoter_hijacking_prom",1)
    df = _set_flag_from_ids(df,utr_ids,"promoter_hijacking_utr",1)

    comb_ids = pl.concat([prom_ids,utr_ids]).unique()
    df = _set_flag_from_ids(df,comb_ids,"promoter_hijacking",1)
    return df

def detect_read_through(
    df: pl.DataFrame,
    nextgene_lookup: pl.DataFrame,
    gtf_full: pl.DataFrame,
    *,
    rt_max_gap: int,
) -> pl.DataFrame:
    """
    Detect strict read-through events between adjacent genes on same chr and strand.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table with gene and breakpoint info.
    nextgene_lookup : pl.DataFrame
        Table with columns: gene_id, chr, strand, next_gene_id.
    gtf_full : pl.DataFrame
        ALL_LOOKUP to map transcript_id → gene_id if missing.
    rt_max_gap : int
        Maximum absolute breakpoint distance allowed between adjacent genes.

    Returns
    -------
    pl.DataFrame
        With 'read_through' set to 1 for qualifying rows.

    Logic
    -----
    • Same chromosome and same strand.
    • gene2 equals next_gene_id of gene1.
    • |bp2 − bp1| ≤ rt_max_gap.
    """

    if "read_through" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("read_through"))

    need_gmap = ("gene_id1" not in df.columns) or ("gene_id2" not in df.columns)
    if need_gmap and {"transcript_id","gene_id"}.issubset(set(gtf_full.columns)):
        tx2gene = gtf_full.select(["transcript_id","gene_id"]).unique().drop_nulls()
        df = df.join(tx2gene.rename({"transcript_id":"transcript_id1","gene_id":"gene_id1"}), on="transcript_id1", how="left")
        df = df.join(tx2gene.rename({"transcript_id":"transcript_id2","gene_id":"gene_id2"}), on="transcript_id2", how="left")

    base = df.filter(
        (pl.col("chromosome1").cast(pl.Utf8) == pl.col("chromosome2").cast(pl.Utf8)) &
        (pl.col("gene_strand1") == pl.col("gene_strand2"))
    )

    adj = nextgene_lookup.select(["gene_id","chr","strand","next_gene_id"]).unique(maintain_order=True)
    step2 = (
        base.join(adj.rename({"gene_id":"gene_id1"}), on="gene_id1", how="left")
            .filter((pl.col("chromosome1").cast(pl.Utf8) == pl.col("chr").cast(pl.Utf8)) &
                    (pl.col("gene_strand1") == pl.col("strand")))
            .with_columns((pl.col("gene_id2") == pl.col("next_gene_id")).alias("are_subsequent"))
    )

    step3 = (
        step2.with_columns([
                pl.col("breakpoint1").cast(pl.Int64, strict=False).alias("bp1_i"),
                pl.col("breakpoint2").cast(pl.Int64, strict=False).alias("bp2_i"),
            ])
            .with_columns((pl.col("bp2_i") - pl.col("bp1_i")).abs().alias("_gap"))
            .filter(pl.col("are_subsequent") & (pl.col("_gap") <= rt_max_gap))
    )

    df = _set_flag_from_ids(df, step3.select("id").unique(), "read_through", 1)
    return df

def detect_read_through_loose(
    df: pl.DataFrame,
    nextgene_df: pl.DataFrame,
    gtf_full: pl.DataFrame,
    *,
    rt_max_gene_gap: int,
    rt_bp_window: int,
    rt_use_tx_anchors: int,
    rt_promote_loose: int,
) -> pl.DataFrame:
    """
    Detect loose read-through using gene-gene or transcript anchors and bp windows.

    Parameters
    ----------
    df : pl.DataFrame
        Fusion table.
    nextgene_df : pl.DataFrame
        NEXT_GENE_LOOKUP with gene adjacency and TSS/TES.
    gtf_full : pl.DataFrame
        ALL_LOOKUP; used if rt_use_tx_anchors is true to derive transcript TSS/TES.
    rt_max_gene_gap : int
        Max distance between TES(gene1) and TSS(gene2).
    rt_bp_window : int
        Window around TES/TSS within which bp1/bp2 must fall.
    rt_use_tx_anchors : int
        If non-zero, use transcript-level TES/TSS; else use gene-level anchors.
    rt_promote_loose : int
        If non-zero, promote loose hits to strict read_through=1.

    Returns
    -------
    pl.DataFrame
        With 'read_through_loose' set to 1 where conditions hold
        and optionally 'read_through' promoted to 1.
    """

    if "read_through_loose" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("read_through_loose"))

    USE_TX = bool(rt_use_tx_anchors)

    need_gmap = ("gene_id1" not in df.columns) or ("gene_id2" not in df.columns)
    if need_gmap and {"transcript_id","gene_id"}.issubset(set(gtf_full.columns)):
        tx2gene = gtf_full.select(["transcript_id","gene_id"]).unique().drop_nulls()
        df = df.join(tx2gene.rename({"transcript_id":"transcript_id1","gene_id":"gene_id1"}), on="transcript_id1", how="left")
        df = df.join(tx2gene.rename({"transcript_id":"transcript_id2","gene_id":"gene_id2"}), on="transcript_id2", how="left")

    base = df.filter(
        (pl.col("chromosome1").cast(pl.Utf8) == pl.col("chromosome2").cast(pl.Utf8)) &
        (pl.col("gene_strand1") == pl.col("gene_strand2"))
    )

    adj = nextgene_df.select(["gene_id","chr","strand","tss","tes","next_gene_id","next_tss","next_tes"]).unique(maintain_order=True)
    j = (
        base.join(adj.rename({"gene_id":"gene_id1"}), on="gene_id1", how="left")
           .filter((pl.col("chromosome1").cast(pl.Utf8) == pl.col("chr").cast(pl.Utf8)) &
                   (pl.col("gene_strand1") == pl.col("strand")))
           .with_columns((pl.col("gene_id2") == pl.col("next_gene_id")).alias("_is_next"))
    )

    if USE_TX and {"feature_type","transcript_id","start","end","strand"}.issubset(set(gtf_full.columns)):
        tx = (gtf_full.filter(pl.col("feature_type") == "transcript")
                      .select(["transcript_id","start","end","strand"])
                      .unique("transcript_id"))
        tx1 = tx.rename({"transcript_id":"transcript_id1","start":"t1_start","end":"t1_end","strand":"t1_strand"})
        tx2 = tx.rename({"transcript_id":"transcript_id2","start":"t2_start","end":"t2_end","strand":"t2_strand"})

        j = (j.join(tx1, on="transcript_id1", how="left")
               .join(tx2, on="transcript_id2", how="left")
               .with_columns([
                   pl.when(pl.col("t1_strand") == "+").then(pl.col("t1_end")).otherwise(pl.col("t1_start")).cast(pl.Int64, strict=False).alias("tes1_tx"),
                   pl.when(pl.col("t2_strand") == "+").then(pl.col("t2_start")).otherwise(pl.col("t2_end")).cast(pl.Int64, strict=False).alias("tss2_tx"),
               ]))
        tes_anchor  = pl.col("tes1_tx")
        tss_anchor2 = pl.col("tss2_tx")
    else:
        tes_anchor  = pl.col("tes").cast(pl.Int64, strict=False)
        tss_anchor2 = pl.col("next_tss").cast(pl.Int64, strict=False)

    j = j.with_columns([
        pl.col("breakpoint1").cast(pl.Int64, strict=False).alias("bp1_i"),
        pl.col("breakpoint2").cast(pl.Int64, strict=False).alias("bp2_i"),
        tes_anchor.alias("tes_i"),
        tss_anchor2.alias("tss2_i"),
    ])

    j = j.with_columns((pl.col("tss2_i") - pl.col("tes_i")).abs().alias("_gene_gap"))
    j_gap = j.filter(pl.col("_is_next") & (pl.col("_gene_gap") <= rt_max_gene_gap))

    near5 = ((pl.col("bp1_i") >= (pl.col("tes_i") - rt_bp_window)) & (pl.col("bp1_i") <= (pl.col("tes_i") + rt_bp_window)))
    near3 = ((pl.col("bp2_i") >= (pl.col("tss2_i") - rt_bp_window)) & (pl.col("bp2_i") <= (pl.col("tss2_i") + rt_bp_window)))
    hits = j_gap.filter(near5 & near3).select("id").unique()

    df = _set_flag_from_ids(df, hits, "read_through_loose", 1)
    if bool(rt_promote_loose):
        df = _set_flag_from_ids(df, hits, "read_through", 1)
    return df

def check_true_fusion(df: pl.DataFrame, th_list: list[float]) -> pl.DataFrame:
    """
    Mark true_fusion_{TH} for rows with sufficient coverage and no exclusion flags.

    Parameters
    ----------
    df : pl.DataFrame
        Must provide coverage1, coverage2 and exclusion flags.
    th_list : list[float]
        Coverage thresholds to evaluate.

    Returns
    -------
    pl.DataFrame
        With one boolean flag per threshold: true_fusion_{TH} ∈ {0,1}.

    Criteria
    --------
    • coverage1 > TH and coverage2 > TH
    • same_gene_fusion == 0
    • read_through == 0
    • promoter_hijacking_500 == 0 and promoter_hijacking_1000 == 0
    """

    for c in ("coverage1", "coverage2"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    for c in ("same_gene_fusion","read_through","promoter_hijacking_500","promoter_hijacking_1000"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))

    candidate = df.filter(
        (pl.col("same_gene_fusion") == 0) &
        (pl.col("read_through") == 0) &
        (pl.col("promoter_hijacking_500") == 0) &
        (pl.col("promoter_hijacking_1000") == 0)
    ).select("id","coverage1","coverage2")

    for th in th_list:
        col = f"true_fusion_{th}"
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))

    for th in th_list:
        col = f"true_fusion_{th}"
        ids = candidate.filter((pl.col("coverage1") > th) & (pl.col("coverage2") > th)).select("id")
        df = _set_flag_from_ids(df, ids, col, 1)
    return df

def count_fusion_categories(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write a table with counts per event category and a 'none' bucket.

    Parameters
    ----------
    df : pl.DataFrame
        Annotated table including event flags.
    output_path : Path
        Destination TSV path.

    Returns
    -------
    None

    Behavior
    --------
    • Counts true_fusion_{TH} for all configured thresholds.
    • Counts promoter_hijacking_500/_1000, read_through, read_through_loose, same_gene_fusion.
    • 'none' counts rows with none of the above flags set to 1.
    """

    rows = []
    for th in CFG_THRESH:
        col = f"true_fusion_{th}"
        if col in df.columns:
            rows.append({"type": col, "count": df.filter(pl.col(col) == 1).height})
    for col in ["promoter_hijacking_500","promoter_hijacking_1000","read_through","read_through_loose","same_gene_fusion"]:
        if col in df.columns:
            rows.append({"type": col, "count": df.filter(pl.col(col) == 1).height})

    flags = [c for c in df.columns if c.startswith("true_fusion_")] + \
            ["promoter_hijacking_500","promoter_hijacking_1000","read_through","read_through_loose","same_gene_fusion"]
    if flags:
        any_flag = pl.lit(False)
        for c in flags:
            any_flag = any_flag | (pl.col(c) == 1)
        none_count = df.filter(~any_flag).height
    else:
        none_count = df.height
    rows.append({"type": "none", "count": none_count})

    ensure_dir(output_path.parent)
    save_file(pl.DataFrame(rows), output_path)

def annotate_events(
    df: pl.DataFrame,
    full_lookup: pl.DataFrame,
    nextgene_lookup: pl.DataFrame,
    *,
    expression: bool,                      # NEW
    coverage_thresholds: list[float],
    rt_max_gap: int,
    rt_max_gene_gap: int,
    rt_bp_window: int,
    rt_use_tx_anchors: int,
    rt_promote_loose: int,
    ph_win_500: int,
    ph_win_1000: int,
    ph_prom_win: int,
    counts_out: Path | None = None
) -> pl.DataFrame:
    """
    Apply all event rules and optionally compute true_fusion thresholds.

    Parameters
    ----------
    df : pl.DataFrame
        Input fusion table.
    full_lookup : pl.DataFrame
        ALL_LOOKUP table.
    nextgene_lookup : pl.DataFrame
        NEXT_GENE_LOOKUP table.
    expression : bool
        If True, evaluate true_fusion_{TH} using coverage1/2 and thresholds.
    coverage_thresholds : list[float]
        Coverage thresholds TH to evaluate when expression=True.
    rt_max_gap : int
        Strict read-through bp gap.
    rt_max_gene_gap : int
        Loose read-through gene-gene gap.
    rt_bp_window : int
        Bp window around TES/TSS for loose read-through.
    rt_use_tx_anchors : int
        Use transcript anchors if non-zero.
    rt_promote_loose : int
        Promote loose hits to strict if non-zero.
    ph_win_500 : int
        Upstream window for promoter_hijacking_500.
    ph_win_1000 : int
        Upstream window for promoter_hijacking_1000.
    ph_prom_win : int
        Upstream window for promoter_hijacking_prom.
    counts_out : Path | None
        If set, write event counts to this path.

    Returns
    -------
    pl.DataFrame
        Annotated DataFrame with event flags added.
    """

    df = check_same_gene(df)
    df = detect_read_through(df, nextgene_lookup, full_lookup, rt_max_gap=rt_max_gap)
    df = detect_read_through_loose(
        df, nextgene_lookup, full_lookup,
        rt_max_gene_gap=rt_max_gene_gap,
        rt_bp_window=rt_bp_window,
        rt_use_tx_anchors=rt_use_tx_anchors,
        rt_promote_loose=rt_promote_loose,
    )
    df = evaluate_promoter_hijacking(
        df, full_lookup,
        ph_win_500=ph_win_500, ph_win_1000=ph_win_1000, ph_prom_win=ph_prom_win
    )
    if expression and coverage_thresholds:
        df = check_true_fusion(df, coverage_thresholds)
    if counts_out:
        count_fusion_categories(df, counts_out)
    return df


# -------------------------
# Orchestration
# -------------------------
def run(input_path: Path, expression: bool = False) -> Path:
    """
    Stage 05 entry point: load inputs, annotate events, and write outputs.

    Parameters
    ----------
    input_path : Path
        Input TSV path from a previous stage.
    expression : bool, default False
        If True, compute true_fusion_{TH} flags using coverage thresholds.

    Returns
    -------
    Path
        Path to INTERIM/<base>_annotated.tsv.

    Side Effects
    ------------
    Writes:
    • INTERIM/<base>_annotated.tsv
    • INTERIM/<base>_event_counts.tsv
    """

    df = load_file(input_path)
    full_lookup     = load_file(ALL_LOOKUP)
    nextgene_lookup = load_file(NEXT_GENE_LOOKUP)
    counts_out = stage_path(FINAL, input_path, "event_counts", ".tsv")

    out = annotate_events(
        df, full_lookup, nextgene_lookup,
        expression=expression,                                
        coverage_thresholds=list(CFG_THRESH) if CFG_THRESH else [],
        rt_max_gap=RT_MAX_GAP,
        rt_max_gene_gap=RT_MAX_GENE_GAP,
        rt_bp_window=RT_BP_WINDOW,
        rt_use_tx_anchors=RT_USE_TX_ANCHORS,
        rt_promote_loose=RT_PROMOTE_LOOSE,
        ph_win_500=PH_WIN_500,
        ph_win_1000=PH_WIN_1000,
        ph_prom_win=PH_PROM_WIN,
        counts_out=counts_out,
    )

    out_path = stage_path(INTERIM, input_path, "annotated", ".tsv")
    
    ensure_dir(out_path.parent)
    save_file(out, out_path)
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: S05_functional_annotation.py <input_tsv>")
    print(run(Path(sys.argv[1])))
