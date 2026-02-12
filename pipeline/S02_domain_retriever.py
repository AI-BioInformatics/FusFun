#!/usr/bin/env python3
"""
S02_domain_retriever.py
=======================
Stage 02 — FAISS-based protein domain retrieval from peptide fragments.

Overview
--------
Given the Stage-01 peptide table and a prebuilt FAISS index of domain-window
embeddings with accompanying metadata, this stage embeds each peptide fragment
(using ESM2), searches the FAISS index, aggregates top hits into InterPro IDs,
and writes, for each fusion:

  • retained_domains1,  lost_domains1,
    retained_domains2,  lost_domains2    (InterPro accessions)
  • retained_domain_names1,  lost_domain_names1,
    retained_domain_names2,  lost_domain_names2  (InterPro descriptions)

Inputs
------
• Stage-01 peptides TSV/CSV with:
  - ID column defined by config.ID_COL
  - REQUIRED_PEP = [retained_peptide1, lost_peptide1,
                    retained_peptide2, lost_peptide2]
• FAISS index path: config.FAISS_INDEX
• Embedding metadata JSON: config.META_FILE
    - May be in NEW or OLD format (see _load_index_meta docstring).
• DOMAIN_WINDOWS TSV:
    - Must contain: window_id, ipr_id, ipr_name (for NEW META_FILE format).

Method
------
1) Load peptides and sanitize missing values.
2) Load FAISS index.
3) Build index-aligned metadata (ipr_id, ipr_name) from META_FILE
   (and DOMAIN_WINDOWS when needed).
4) Embed all peptide sequences with ESM2 using mean pooling over residues,
   windowing very long sequences, and batching under token budgets.
5) L2-normalize embeddings and query FAISS for top-k neighbors.
6) Deduplicate InterPro IDs per query to K_UNIQUE by highest similarity,
   returning both IDs and names.
7) Write:
   - domains-only file: ID + four domain ID columns + four domain name columns
   - domains-full file: all Stage-01 columns + those eight columns

Outputs
-------
• `<FAISS_OUT>/..._domains.tsv`
• `<FAISS_OUT>/..._domains_full.tsv`

Notes
-----
• GPU usage is optional; DataParallel is used when multiple GPUs are visible.
• Stop codons should have been replaced upstream; empty fragments are handled.
• All paths and numeric parameters come from config.config.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
import numpy as np
import polars as pl
import torch
import faiss
from esm import pretrained

from qol.utilities import load_file, save_file, stage_path
from config.config import (
    FAISS_OUT, FAISS_INDEX, META_FILE, ID_COL,
    TOPK_RAW, K_UNIQUE, FAISS_THREADS,
    GPUS, VERY_LONG_AA, WINDOW_SIZE_AA, WINDOW_STEP_AA,
    LONG_AA_BATCH, BATCH_SHORT, BATCH_LONG, BUDGET_SHORT, BUDGET_LONG,
    SEARCH_BATCH, OUTPUT_DIR,
    DOMAIN_WINDOWS,
)

REQUIRED_PEP = [
    "retained_peptide1",
    "lost_peptide1",
    "retained_peptide2",
    "lost_peptide2",
]

OUT_COLS = [
    "retained_domains1",
    "lost_domains1",
    "retained_domains2",
    "lost_domains2",
]

OUT_NAME_COLS = [
    "retained_domain_names1",
    "lost_domain_names1",
    "retained_domain_names2",
    "lost_domain_names2",
]


# ---- ESM2 loader ----
def _load_esm(gpu_count: int | None):
    """
    Load the ESM2 model and batch converter, optionally on GPU.

    Parameters
    ----------
    gpu_count : int | None
        Number of GPUs to use. 0 forces CPU. None falls back to config.GPUS.

    Returns
    -------
    model : torch.nn.Module
        ESM2 t33-650M-UR50D in eval mode. On CUDA(0) if available. Wrapped in
        torch.nn.DataParallel when multiple GPUs are used.
    batch_converter : Callable
        Function from ESM alphabet that converts [(name, seq), ...] to tokens.
    """

    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval()

    use_cuda = torch.cuda.is_available()
    if not use_cuda or (gpu_count == 0):
        return model, alphabet.get_batch_converter()  # CPU

    visible = torch.cuda.device_count()
    n = gpu_count if gpu_count is not None else GPUS
    n = max(1, min(n, visible))

    model = model.cuda(0)
    if n > 1:
        model = torch.nn.DataParallel(model)  # uses all visible devices
    return model, alphabet.get_batch_converter()


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2-normalization with epsilon guard."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _windows(seq: str, size: int, step: int) -> List[str]:
    """
    Generate overlapping windows for very long amino-acid sequences.

    Parameters
    ----------
    seq : str
        Amino-acid sequence.
    size : int
        Window size in residues.
    step : int
        Step size between consecutive windows.

    Returns
    -------
    List[str]
        One or more sequence windows. If sequence length ≤ size, returns [seq].
    """

    L = len(seq)
    if L <= size:
        return [seq]
    out: List[str] = []
    i = 0
    while True:
        j = min(L, i + size)
        out.append(seq[i:j])
        if j == L:
            break
        i += step
    return out


def _pack_buckets(seqs: List[str]) -> List[List[int]]:
    """
    Pack sequence indices into batches under size and token budgets.

    Parameters
    ----------
    seqs : List[str]
        Raw sequences to embed.

    Returns
    -------
    List[List[int]]
        A list of batches, each a list of indices into `seqs`.

    Notes
    -----
    • Sequences are split into "short" and "long" by config.LONG_AA_BATCH.
    • Two packers run with (BATCH_SHORT, BUDGET_SHORT) and (BATCH_LONG, BUDGET_LONG).
    • Budgets are measured in amino-acid tokens prior to ESM tokenization.
    """

    short = [i for i, s in enumerate(seqs) if len(s) < LONG_AA_BATCH]
    long = [i for i, s in enumerate(seqs) if len(s) >= LONG_AA_BATCH]

    def mk(idxs: List[int], cap: int, budget: int) -> List[List[int]]:
        bks: List[List[int]] = []
        i = 0
        while i < len(idxs):
            b: List[int] = []
            tok = 0
            while i < len(idxs):
                k = idxs[i]
                L = len(seqs[k])
                if len(b) + 1 > cap or tok + L > budget:
                    break
                b.append(k)
                tok += L
                i += 1
            if not b:
                b = [idxs[i]]
                i += 1
            bks.append(b)
        return bks

    out: List[List[int]] = []
    out += mk(short, BATCH_SHORT, BUDGET_SHORT)
    out += mk(long, BATCH_LONG, BUDGET_LONG)
    #out += [[short[i] for i in b] for b in mk(short, BATCH_SHORT, BUDGET_SHORT)]
    # out += [[long[i] for i in b] for b in mk(long, BATCH_LONG, BUDGET_LONG)]
    return out


@torch.inference_mode()
def _embed_mean(seqs_raw: List[str], model, batch_converter) -> np.ndarray:
    """
    Embed sequences with ESM2 and mean-pool residue representations.

    Parameters
    ----------
    seqs_raw : List[str]
        Input sequences. Empty or None are treated as empty strings.
    model : torch.nn.Module
        ESM2 model returned by `_load_esm`.
    batch_converter : Callable
        Batch converter returned by `_load_esm`.

    Returns
    -------
    np.ndarray
        Shape (N, 1280). L2-normalized embeddings aligned to `seqs_raw`.
    """

    owners: List[int] = []
    flat: List[str] = []
    for i, s in enumerate(seqs_raw):
        s = (s or "").replace(" ", "")
        if len(s) > VERY_LONG_AA:
            for w in _windows(s, WINDOW_SIZE_AA, WINDOW_STEP_AA):
                flat.append(w)
                owners.append(i)
        else:
            flat.append(s)
            owners.append(i)

    D = 1280
    out = np.zeros((len(flat), D), np.float32)
    model_device = next(model.parameters()).device

    for b in _pack_buckets(flat):
        _, strs, toks = batch_converter([("q", flat[i]) for i in b])
        toks = toks.to(model_device, non_blocking=True)
        rep = model(tokens=toks, repr_layers=[33], return_contacts=False)["representations"][33]
        mean = rep[:, 1:-1].mean(1).detach().cpu().to(torch.float32).numpy()
        out[np.array(b)] = mean

    agg: Dict[int, List[np.ndarray]] = {}
    for i, own in enumerate(owners):
        agg.setdefault(own, []).append(out[i])

    Q = np.zeros((len(seqs_raw), D), np.float32)
    for own, vecs in agg.items():
        Q[own] = np.mean(np.stack(vecs, 0), 0)
    return _l2(Q)


# ---- Meta: META_FILE (+ DOMAIN_WINDOWS) → index-aligned (ipr_id, ipr_name) ----
def _load_index_meta() -> List[Dict]:
    """
    Build index-aligned metadata for FAISS.

    Supports two META_FILE formats:

    1) NEW format (from current S04_embed_domain_windows.py):
       {
         "model": ...,
         "device": ...,
         "windows": ["win_id_0", "win_id_1", ...],
         ...
       }
       In this case we:
         - read DOMAIN_WINDOWS (TSV)
         - map window_id -> (ipr_id, ipr_name)
         - return a list of dicts aligned to the 'windows' list.

    2) OLD format:
       [
         { "ipr_id": "...", "ipr_desc": "...", ... },
         ...
       ]
       In this case we:
         - use the list directly
         - derive 'ipr_id' and 'ipr_name' from the row itself
         - do NOT read DOMAIN_WINDOWS.
    """

    meta_path = Path(META_FILE)
    txt = meta_path.read_text(encoding="utf-8").strip()
    meta_obj = json.loads(txt)

    # --- Case 1: NEW format with "windows" key ---
    if isinstance(meta_obj, dict) and "windows" in meta_obj:
        windows = meta_obj["windows"]
        if not isinstance(windows, list):
            raise ValueError("[S02] META_FILE 'windows' must be a list.")

        dom_path = Path(DOMAIN_WINDOWS)
        df_win = pl.read_csv(dom_path, separator="\t", infer_schema_length=0)

        for col in ("window_id", "ipr_id"):
            if col not in df_win.columns:
                raise ValueError(f"[S02] DOMAIN_WINDOWS missing required column '{col}'.")

        has_name = "ipr_name" in df_win.columns

        # window_id -> (ipr_id, ipr_name)
        win_map: Dict[str, tuple[str, str]] = {}
        cols = ["window_id", "ipr_id"] + (["ipr_name"] if has_name else [])
        for row in df_win.select(cols).to_dicts():
            wid = row["window_id"]
            ipr_id = str(row["ipr_id"]) if row["ipr_id"] is not None else "."
            if has_name and row["ipr_name"] is not None:
                ipr_name = str(row["ipr_name"])
            else:
                ipr_name = ""
            win_map[wid] = (ipr_id, ipr_name)

        meta_rows: List[Dict] = []
        for wid in windows:
            ipr_id, ipr_name = win_map.get(wid, (".", ""))
            meta_rows.append({"ipr_id": ipr_id, "ipr_name": ipr_name})
        return meta_rows

    # --- Case 2: OLD format → list of dicts ---
    if isinstance(meta_obj, list):
        meta_rows: List[Dict] = []
        for row in meta_obj:
            if not isinstance(row, dict):
                continue
            ipr_id = (
                row.get("ipr_id")
                or row.get("IPR")
                or row.get("ipr")
                or row.get("signature_accession")
                or "."
            )
            ipr_name = (
                row.get("ipr_name")
                or row.get("ipr_desc")
                or row.get("signature_description")
                or ""
            )
            meta_rows.append({"ipr_id": str(ipr_id), "ipr_name": str(ipr_name)})
        return meta_rows

    raise ValueError("[S02] META_FILE JSON must be either a dict with 'windows' or a list of rows.")


def _dedup_ipr_with_names(
    Irow: np.ndarray,
    Drow: np.ndarray,
    meta: List[Dict],
) -> tuple[str, str]:
    """
    Aggregate FAISS neighbors into deduplicated InterPro ID and name strings.

    Parameters
    ----------
    Irow : np.ndarray
        Indices of nearest neighbors for one query. Shape (k,).
    Drow : np.ndarray
        Similarity scores (higher is better) for one query. Shape (k,).
    meta : List[Dict]
        Index-aligned metadata with keys 'ipr_id' and 'ipr_name'.

    Returns
    -------
    ids_str : str
        '|'-separated InterPro IDs ordered by descending score, up to K_UNIQUE.
        '.' if no valid IDs are found.
    names_str : str
        '|'-separated domain names/descriptions in the same order as ids_str.
        '.' if no valid names are found.
    """

    seen: Dict[str, tuple[float, str, str]] = {}

    for idx, sc in zip(Irow, Drow):
        if idx < 0:
            continue
        m = meta[int(idx)]
        ipr = (m.get("ipr_id") or "").strip()
        if not ipr:
            continue
        nm = (m.get("ipr_name") or "").strip()
        if ipr not in seen or sc > seen[ipr][0]:
            seen[ipr] = (float(sc), ipr, nm)
        if len(seen) >= K_UNIQUE:
            break

    if not seen:
        return ".", "."

    ordered = sorted(seen.values(), key=lambda t: t[0], reverse=True)
    ids_str = "|".join(v[1] for v in ordered)
    names_str = "|".join((v[2] or "NA") for v in ordered)
    return ids_str, names_str


# ---- Main API ----
def run(peptides_csv: Path, gpu_count: int | None = None) -> Path:
    """
    Execute Stage-02 FAISS domain retrieval end-to-end.

    Parameters
    ----------
    peptides_csv : Path
        Stage-01 output file with ID_COL and REQUIRED_PEP columns.
    gpu_count : int | None, default None
        Number of GPUs to use. 0 forces CPU. None uses config.GPUS.

    Returns
    -------
    Path
        Path to the domains-full TSV (all Stage-01 columns plus domain columns).
    """

    # load peptides
    df = load_file(peptides_csv, required=[ID_COL, *REQUIRED_PEP])

    # keep only what embedding needs (df still keeps all metadata)
    _ = df.select([
        ID_COL,
        "retained_peptide1", "lost_peptide1",
        "retained_peptide2", "lost_peptide2",
    ]).with_columns([
        pl.col("retained_peptide1").fill_null("."),
        pl.col("lost_peptide1").fill_null("."),
        pl.col("retained_peptide2").fill_null("."),
        pl.col("lost_peptide2").fill_null("."),
    ])

    # load index
    index = faiss.read_index(str(Path(FAISS_INDEX)))
    if isinstance(FAISS_THREADS, int) and FAISS_THREADS > 0:
        try:
            faiss.omp_set_num_threads(FAISS_THREADS)
        except Exception:
            pass

    # empty index guard: return dots for all ID and name cols
    if getattr(index, "ntotal", 0) == 0:
        out = pl.DataFrame({ID_COL: df[ID_COL]})
        for c in OUT_COLS + OUT_NAME_COLS:
            out = out.with_columns(pl.lit(".").alias(c))
        dom_csv = stage_path(FAISS_OUT, peptides_csv, "domains", ".tsv")
        save_file(out, dom_csv)
        return dom_csv

    # index has entries: build meta from META_FILE (+ DOMAIN_WINDOWS)
    meta = _load_index_meta()
    k = int(min(TOPK_RAW, index.ntotal))

    # embed per peptide column
    model, bc = _load_esm(gpu_count)
    Q_by_col: Dict[str, np.ndarray] = {}
    for col in REQUIRED_PEP:
        seqs = df[col].fill_null("").to_list()
        Q_by_col[col] = _embed_mean(seqs, model, bc)

    # FAISS search + IPR aggregation (IDs + names)
    out = pl.DataFrame({ID_COL: df[ID_COL]})
    for col, out_col, out_name_col in zip(REQUIRED_PEP, OUT_COLS, OUT_NAME_COLS):
        Q = Q_by_col[col]
        ipr_list: List[str] = []
        name_list: List[str] = []

        for i0 in range(0, Q.shape[0], SEARCH_BATCH):
            q = Q[i0: i0 + SEARCH_BATCH].astype(np.float32, order="C")
            D, I = index.search(q, k)
            for drow, irow in zip(D, I):
                ids_str, names_str = _dedup_ipr_with_names(irow, drow, meta)
                ipr_list.append(ids_str)
                name_list.append(names_str)

        out = out.with_columns([
            pl.Series(out_col, ipr_list),
            pl.Series(out_name_col, name_list),
        ])

    # 1) domains-only file (id + domain IDs + names)
    dom_only = out.select([ID_COL, *OUT_COLS, *OUT_NAME_COLS])
    dom_only_path = stage_path(FAISS_OUT, peptides_csv, "domains", ".tsv")
    save_file(dom_only, dom_only_path)

    # 2) full file = all Stage-01 columns + ID + name columns
    dom_full = df.join(dom_only, on=ID_COL, how="left")
    dom_full_path = stage_path(FAISS_OUT, peptides_csv, "domains_full", ".tsv")
    save_file(dom_full, dom_full_path)

    print(f"[S02] extracted domains:  {dom_only_path}")
    return dom_full_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: S02_domain_retriever.py <peptides_tsv>")
    run(Path(sys.argv[1]))
