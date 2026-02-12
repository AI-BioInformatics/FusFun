#!/usr/bin/env python3
"""
S06_oncogenicity_scoring.py
===========================

Purpose
-------
Perform Positive–Unlabeled (PU) graph diffusion to assign oncogenicity scores
to fusion events. Known driver fusions act as positive seeds; unlabeled fusions
receive propagated scores through a kNN graph built on domain-level features.

Modes
-----
• Production (EVAL = False)
    Graph = USER_FUSIONS ∪ KNOWN_POSITIVES
    Seeds = all KNOWN_POSITIVES
    → Outputs: FINAL/<stem>_scores.tsv

• Research (EVAL = True)
    Randomly split KNOWN_POSITIVES into train (seeds) and holdout (evaluation).
    Graph = USER_FUSIONS ∪ POS_TRAIN ∪ POS_HOLDOUT
    Seeds = POS_TRAIN
    → Outputs:
        FINAL/<stem>_scores.tsv
        FINAL/<stem>_holdout_scores.tsv
        STATS/pu_metrics.csv (recall@K, EF@K, NDCG@K, AP)

Input Requirements
------------------
• input_path : S05-like TSV with columns
    id, gene1, gene2,
    retained_domains1, lost_domains1,
    retained_domains2, lost_domains2,
    out_of_frame1, out_of_frame2
• KNOWN_POSITIVES : S05-like TSV of curated driver fusions with the same schema.

Configuration Keys
------------------
HASH_DIM, TFIDF_ON, TFIDF_NORM
KNN_K, ALPHA, MAX_ITER, TOL
KNOWN_POSITIVES, EVAL, SPLIT_HOLDOUT_FRAC, RANDOM_SEED
METRICS_K, FINAL, STATS

Outputs
-------
Production:
    FINAL/<stem>_scores.tsv
Research:
    FINAL/<stem>_scores.tsv
    FINAL/<stem>_holdout_scores.tsv
    STATS/pu_metrics.csv

Method Summary
---------------
1. Load and align USER_FUSIONS and KNOWN_POSITIVES tables.
2. Optionally split positives into train/holdout.
3. Convert domain columns to side-aware token strings.
4. Vectorize text (hashing + optional TF-IDF) and append numeric features.
5. Build symmetric kNN cosine-similarity graph.
6. Diffuse labels from positive seeds using α-restart propagation.
7. Normalize diffusion scores to [0,1] and save results.
"""


from __future__ import annotations
from pathlib import Path
import re, math, random
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, coo_matrix, diags, hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score

from qol.utilities import load_file, save_file, ensure_dir, stage_path

from config.config import (
    # outputs
    STATS, INTERIM, VISUALS,
    # graph/featurization knobs
    HASH_DIM, KNN_K, ALPHA, MAX_ITER, TOL,
    TFIDF_ON, TFIDF_NORM,
    # positives and mode
    KNOWN_POSITIVES,
    EVAL, SPLIT_HOLDOUT_FRAC, RANDOM_SEED,
    # metrics
    METRICS_K,
)

# ---------- helpers ----------

REQUIRED_DOMAIN_COLS = [
    "retained_domains1", "lost_domains1", "retained_domains2", "lost_domains2",
    "out_of_frame1", "out_of_frame2",
]

def _base_stem(p: Path) -> str:
    s = p.stem
    for pat in (r"(?:_annotated)$", r"(?:_features)$", r"(?:_domains(_full)?)$",
                r"(?:_ips(_full)?)$", r"(?:_peptides(_domains(_full)?)?)$"):
        s = re.sub(pat, "", s)
    return s

def _pick(cols: list[str], *cands: str) -> str | None:
    for c in cands:
        if c in cols:
            return c
    return None

def _clean_ipr_cell(x: str) -> str:

    """
    Normalize a domain-list cell into a canonical 'A | B | C' format.
    Removes extra spaces and placeholder tokens like '.'.
    Returns an empty string if no valid domains are present.
    """

    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if x == "." or x == "":
        return ""
    return " | ".join(tok.strip() for tok in x.split("|") if tok.strip())

def _sideaware_text(df: pl.DataFrame) -> list[str]:
    """
    Construct side-aware textual tokens for each fusion row.

    For each of the four domain columns:
        retained_domains1 → 'r1:IPRxxxx'
        lost_domains1     → 'l1:IPRxxxx'
        retained_domains2 → 'r2:IPRxxxx'
        lost_domains2     → 'l2:IPRxxxx'
    Returns a list of space-separated token strings, one per row.
    """
    r1 = df.get_column("retained_domains1") if "retained_domains1" in df.columns else pl.lit(".")
    l1 = df.get_column("lost_domains1")     if "lost_domains1"     in df.columns else pl.lit(".")
    r2 = df.get_column("retained_domains2") if "retained_domains2" in df.columns else pl.lit(".")
    l2 = df.get_column("lost_domains2")     if "lost_domains2"     in df.columns else pl.lit(".")
    out: list[str] = []
    for A, B, C, D in zip(r1.to_list(), l1.to_list(), r2.to_list(), l2.to_list()):
        A = _clean_ipr_cell(A); B = _clean_ipr_cell(B); C = _clean_ipr_cell(C); D = _clean_ipr_cell(D)
        toks = []
        if A: toks.append("r1:" + A.replace(" ", "_"))
        if B: toks.append("l1:" + B.replace(" ", "_"))
        if C: toks.append("r2:" + C.replace(" ", "_"))
        if D: toks.append("l2:" + D.replace(" ", "_"))
        out.append(" ".join(toks))
    return out

def _count_iprs(cell: str) -> int:
    """
    Count non-empty domain identifiers in a pipe-separated cell.
    Used to create numeric features for each fusion side.
    """
    if not isinstance(cell, str) or cell.strip() in ("", "."):
        return 0
    return sum(1 for p in cell.split("|") if p.strip())

def _numeric_block(df: pl.DataFrame) -> csr_matrix:
    """
    Build the numeric feature matrix (CSR format) containing:
        • out_of_frame1, out_of_frame2
        • counts of retained/lost domains per side
        • binary noncoding indicators (out_of_frame == 2)
    Returns a matrix of shape (N, 8).
    """
    of1 = (df.get_column("out_of_frame1") if "out_of_frame1" in df.columns else pl.lit(2)).fill_null(2).cast(pl.Int8).to_numpy()
    of2 = (df.get_column("out_of_frame2") if "out_of_frame2" in df.columns else pl.lit(2)).fill_null(2).cast(pl.Int8).to_numpy()
    n = len(of1)
    r1 = np.array([_count_iprs(x) for x in (df.get_column("retained_domains1").fill_null(".").to_list() if "retained_domains1" in df.columns else ["."]*n)], dtype=np.float32)
    l1 = np.array([_count_iprs(x) for x in (df.get_column("lost_domains1").fill_null(".").to_list()     if "lost_domains1"     in df.columns else ["."]*n)], dtype=np.float32)
    r2 = np.array([_count_iprs(x) for x in (df.get_column("retained_domains2").fill_null(".").to_list() if "retained_domains2" in df.columns else ["."]*n)], dtype=np.float32)
    l2 = np.array([_count_iprs(x) for x in (df.get_column("lost_domains2").fill_null(".").to_list()     if "lost_domains2"     in df.columns else ["."]*n)], dtype=np.float32)
    noncoding1 = (of1 == 2).astype(np.float32)
    noncoding2 = (of2 == 2).astype(np.float32)
    X = np.vstack([of1, of2, r1, l1, r2, l2, noncoding1, noncoding2]).T
    return csr_matrix(X)

def _kneighbors_graph_cosine(X: csr_matrix, k: int) -> csr_matrix:
    """
    Construct a symmetric k-nearest-neighbors graph using cosine similarity.
    Distance metric = 1 − cosine_similarity.
    Output: sparse (N×N) CSR adjacency matrix with zero diagonal.
    """
    n = X.shape[0]
    if n <= 1:
        return csr_matrix((n, n))
    k = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(X)
    Gdist = nn.kneighbors_graph(X, mode="distance").tocoo()
    sim = 1.0 - Gdist.data
    sim[sim < 0.0] = 0.0
    W = coo_matrix((sim, (Gdist.row, Gdist.col)), shape=Gdist.shape).tocsr()
    W = 0.5 * (W + W.T)
    W.setdiag(0.0); W.eliminate_zeros()
    return W

def _row_normalize(W: csr_matrix) -> csr_matrix:
    """
    Convert adjacency matrix W into a row-stochastic transition matrix P
    so that each row sums to 1. Used for diffusion propagation.
    """
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0.0] = 1.0
    return diags(1.0 / rs) @ W

def _diffuse(P: csr_matrix, y: np.ndarray, alpha: float, max_iter: int, tol: float) -> np.ndarray:
    """
    Perform label diffusion with restart on graph P.

    Recurrence:
        s_{t+1} = α * (P @ s_t) + (1 − α) * y

    Parameters
    ----------
    P : csr_matrix
        Row-normalized transition matrix.
    y : np.ndarray
        Initial seed vector (1 = positive, 0 = unlabeled).
    alpha : float
        Teleport parameter controlling diffusion retention.
    max_iter : int
        Maximum number of iterations.
    tol : float
        L1 convergence threshold.

    Returns
    -------
    np.ndarray
        Final diffusion scores (unnormalized).
    """
    s = y.astype(np.float64, copy=True)
    for _ in range(int(max_iter)):
        s_new = alpha * (P @ s) + (1.0 - alpha) * y
        if np.abs(s_new - s).sum() <= tol:
            return s_new
        s = s_new
    return s

def _align_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Harmonize legacy and current column names for compatibility
    between datasets (e.g. rename 'ipro_retained_peptide1_iprs' → 'retained_domains1').
    """

    ren = {}
    # legacy InterPro names
    for a, b in [
        ("ipro_retained_peptide1_iprs", "retained_domains1"),
        ("ipro_lost_peptide1_iprs",     "lost_domains1"),
        ("ipro_retained_peptide2_iprs", "retained_domains2"),
        ("ipro_lost_peptide2_iprs",     "lost_domains2"),
        ("out_of_frame_left",  "out_of_frame1"),
        ("out_of_frame_right", "out_of_frame2"),
        ("Gene1","gene1"), ("Gene2","gene2"), ("ID","id"),
    ]:
        if a in df.columns and b not in df.columns:
            ren[a] = b
    return df.rename(ren)

def _require_domain_schema(df: pl.DataFrame, label: str):
    """
    Validate that `df` contains all mandatory S05 columns required for diffusion.
    Raises ValueError with informative message if any are missing.
    """
    missing = [c for c in REQUIRED_DOMAIN_COLS if c not in df.columns]
    for base in ("gene1", "gene2", "id"):
        if base not in df.columns:
            missing.append(base)
    if missing:
        raise ValueError(
            f"[S06] {label} lacks required columns: {missing}. "
            "KNOWN_POSITIVES must be an S05-like table with domain and frame columns."
        )

# ---------- metrics (holdout vs others in union) ----------

def _compute_metrics_union(scores: np.ndarray, mask_holdout: np.ndarray, ks: list[int]) -> pl.DataFrame:
    """
    Evaluate model ranking quality when EVAL=True.

    Treat holdout nodes as positives and all others as unlabeled.
    Compute for each K in METRICS_K:
        • recall@K
        • enrichment factor (EF@K)
        • NDCG@K
    and overall Average Precision (AP).
    Returns a Polars DataFrame ready for CSV export.
    """
    n = scores.shape[0]
    if n == 0 or mask_holdout.sum() == 0:
        return pl.DataFrame({"K": ks, "recall_at_k": [0.0]*len(ks), "EF_at_k": [0.0]*len(ks),
                             "NDCG_at_k": [0.0]*len(ks), "AveragePrecision": [0.0]*len(ks)})
    order = np.argsort(-scores, kind="mergesort")
    rel_sorted = mask_holdout.astype(int)[order]
    P = int(mask_holdout.sum())
    base_rate = P / n
    rows = []
    for k in ks:
        K = max(1, min(int(k), n))
        tp = int(rel_sorted[:K].sum())
        recall_k = tp / P
        prec_k = tp / K
        ef_k = (prec_k / base_rate) if base_rate > 0 else 0.0
        # NDCG@K
        gains = rel_sorted[:K].astype(float)
        dcg = float(np.sum((2.0**gains - 1.0) / np.log2(np.arange(2, K + 2))))
        ideal = float(np.sum((2.0**np.sort(rel_sorted)[::-1][:K] - 1.0) / np.log2(np.arange(2, K + 2))))
        ndcg_k = 0.0 if ideal == 0.0 else dcg / ideal
        rows.append((K, recall_k, ef_k, ndcg_k))
    # AP over entire ranking
    y_true = mask_holdout.astype(int)
    ap = float(average_precision_score(y_true, scores))
    df = pl.DataFrame({
        "K": [r[0] for r in rows],
        "recall_at_k": [r[1] for r in rows],
        "EF_at_k": [r[2] for r in rows],
        "NDCG_at_k": [r[3] for r in rows],
    }).with_columns(pl.lit(ap).alias("AveragePrecision"))
    return df

# ---------- main ----------

def run(input_path: Path) -> tuple[Path, Path | None, Path | None]:
    """
    Main entry point.

    Parameters
    ----------
    input_path : Path
        Path to the user fusion file (<stem>_annotated.tsv).

    Returns
    -------
    tuple
        (sample_scores_path, holdout_scores_path | None, metrics_path | None)

    Workflow
    --------
    1. Load and verify USER_FUSIONS and KNOWN_POSITIVES tables.
    2. If EVAL=True, randomly split positives into train/holdout.
    3. Concatenate USER_FUSIONS + POS_TRAIN (+ POS_HOLDOUT if EVAL).
    4. Generate side-aware text features and numeric block.
    5. Build kNN graph, run label diffusion, min–max normalize scores.
    6. Save per-fusion scores; optionally compute holdout metrics.
    """
    # sample
    sample = _align_schema(load_file(input_path))
    _require_domain_schema(sample, "USER_FUSIONS")

    # positives
    pos_full = _align_schema(load_file(KNOWN_POSITIVES))
    _require_domain_schema(pos_full, "KNOWN_POSITIVES")

    # split if EVAL
    if bool(EVAL):
        frac = float(SPLIT_HOLDOUT_FRAC)
        rng = random.Random(int(RANDOM_SEED))
        # stable shuffle using Polars row index
        idx = list(range(pos_full.height))
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - frac))) if len(idx) > 1 else 1
        train_idx = set(idx[:cut]); hold_idx = set(idx[cut:])
        pos_train = pos_full.take(sorted(list(train_idx))) if train_idx else pos_full
        pos_hold  = pos_full.take(sorted(list(hold_idx)))  if hold_idx else pl.DataFrame(schema=pos_full.schema)
        pos_train = pos_train.with_columns(pl.lit("pos_train").alias("_src"))
        pos_hold  = pos_hold.with_columns(pl.lit("pos_holdout").alias("_src"))
        print(f"[S06] EVAL=True, positives split: train={pos_train.height}, holdout={pos_hold.height}")
    else:
        pos_train = pos_full.with_columns(pl.lit("pos_train").alias("_src"))
        pos_hold  = pl.DataFrame(schema=pos_full.schema).with_columns(pl.lit("pos_holdout").alias("_src"))
        print(f"[S06] EVAL=False, all positives used as seeds: {pos_train.height}")

    # tag sample
    sample = sample.with_columns(pl.lit("sample").alias("_src"))

    # align columns across frames
    cols_core = ["id","gene1","gene2","retained_domains1","lost_domains1","retained_domains2","lost_domains2","out_of_frame1","out_of_frame2","_src"]
    def _ensure_cols(df: pl.DataFrame) -> pl.DataFrame:
        for c in cols_core:
            if c not in df.columns:
                df = df.with_columns(pl.lit(None).alias(c))
        return df.select(cols_core)

    S = _ensure_cols(sample)
    PT = _ensure_cols(pos_train)
    PH = _ensure_cols(pos_hold)

    # union
    U = pl.concat([S, PT, PH], how="vertical_relaxed")

    # features
    texts = _sideaware_text(U)
    hv = HashingVectorizer(
        n_features=int(HASH_DIM),
        alternate_sign=False,
        lowercase=False,
        token_pattern=r"[^ \t\n\r\f\v]+",
        norm=None,
        binary=not bool(TFIDF_ON),
    )
    X_text_raw = hv.transform(texts)
    X_text = (
        TfidfTransformer(norm=(None if str(TFIDF_NORM).lower() == "none" else str(TFIDF_NORM)), use_idf=True)
        .fit_transform(X_text_raw)
        if bool(TFIDF_ON) else X_text_raw
    )
    X_num = _numeric_block(U)
    X = hstack([X_text, X_num], format="csr")

    # graph + diffusion
    W = _kneighbors_graph_cosine(X, k=int(KNN_K))
    P = _row_normalize(W)
    y = (U["_src"] == "pos_train").to_numpy().astype(np.float64)
    print(f"[S06] Seed nodes: {int(y.sum())}")
    s_raw = _diffuse(P, y, alpha=float(ALPHA), max_iter=int(MAX_ITER), tol=float(TOL))
    lo, hi = float(s_raw.min()), float(s_raw.max())
    s = np.zeros_like(s_raw, dtype=np.float32) if hi - lo < 1e-12 else ((s_raw - lo) / (hi - lo)).astype(np.float32)

    # slice
    mask_sample  = (U["_src"] == "sample").to_numpy()
    mask_holdout = (U["_src"] == "pos_holdout").to_numpy()
    S_sc = s[mask_sample]
    H_sc = s[mask_holdout]

    # outputs
    stem = _base_stem(Path(input_path))
    out_scores = stage_path(INTERIM, input_path, "scores", ".tsv")
    ensure_dir(out_scores.parent)
    save_file(S.select(["id","gene1","gene2"]).with_columns(pl.Series("driver_score", S_sc)), out_scores)

    out_hold = None
    metrics_path = None
    if bool(EVAL):
        # holdout scores file
        out_hold = stage_path(INTERIM, input_path, "holdout_scores", ".tsv")
        ensure_dir(out_hold.parent)
        save_file(PH.select(["id","gene1","gene2"]).with_columns(pl.Series("driver_score", H_sc)), out_hold)

        # metrics over union, positives=holdout nodes
        metrics = _compute_metrics_union(scores=s, mask_holdout=mask_holdout, ks=list(METRICS_K))
        ensure_dir(Path(STATS))
        metrics_path = Path(STATS) / "pu_metrics.csv"
        save_file(metrics, metrics_path)

        print(f"[S06] wrote: {out_scores}")
        print(f"[S06] wrote: {out_hold}")
        print(f"[S06] metrics: {metrics_path}")
    else:
        print(f"[S06] Final file saved: {out_scores}")

    return out_scores

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: S06_oncogenicity_scoring.py <annotated_tsv>")
    run(Path(sys.argv[1]))
