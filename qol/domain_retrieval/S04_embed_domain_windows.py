# qol/domain_retrieval/s04_embed_domain_windows.py
#!/usr/bin/env python3
"""
S04_embed_domain_windows.py
===========================
Stage S04 of the domain-retrieval pipeline.  
Embeds all domain windows using an ESM-2 transformer model and writes:
  • PROTEIN_EMBEDDINGS (.npy) — float32 array of shape [N, D]
  • META_FILE (JSON) — model, device, and window provenance metadata

Overview
--------
Input:
  DOMAIN_WINDOWS (TSV)
      Columns: window_id, sequence
Process:
  1) Load domain windows and sequence lengths.
  2) Load the specified ESM-2 pretrained model.
  3) Batch sequences respecting ESM_MAX_TOKENS and ESM_BATCH.
  4) Encode each window, mean-pool token embeddings, and store results.
  5) Save embeddings and JSON metadata.

Output files
-------------
• PROTEIN_EMBEDDINGS — NumPy .npy array (float32)  
• META_FILE — JSON metadata aligned with embedding order

Config keys used
----------------
DOMAIN_WINDOWS, PROTEIN_EMBEDDINGS, META_FILE,  
ESM_MODEL_NAME, ESM_DEVICE, ESM_BATCH, ESM_MAX_TOKENS, USE_FP16

Notes
-----
• The model layer used is always the final hidden layer (`model.num_layers`).  
• Mixed precision is enabled when `USE_FP16=True`.  
• Sequence order in the .npy file matches the `windows` list in META_FILE.
"""

from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import polars as pl
import torch
import esm  # pip install fair-esm

from config.config import (
    DOMAIN_WINDOWS,
    PROTEIN_EMBEDDINGS,
    META_FILE,
    ESM_MODEL_NAME,
    ESM_DEVICE,
    ESM_BATCH,
    ESM_MAX_TOKENS,
    USE_FP16,
)

def _load_windows(path: Path) -> tuple[list[str], list[str]]:
    """
    Load window identifiers and sequences from DOMAIN_WINDOWS.

    Parameters
    ----------
    path : Path
        Path to the domain-windows TSV.

    Returns
    -------
    (list[str], list[str])
        Parallel lists of window_ids and amino-acid sequences.

    Raises
    ------
    ValueError
        If required columns ("window_id", "sequence") are missing.
    """

    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    req = ["window_id", "sequence"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"[S04] DOMAIN_WINDOWS missing columns: {miss}")
    # preserve order
    wids = df["window_id"].to_list()
    seqs = df["sequence"].to_list()
    return wids, seqs

def _load_model(name: str):
    """
    Load a pretrained ESM-2 model and its alphabet by name.

    Parameters
    ----------
    name : str
        Model identifier, e.g. "esm2_t33_650M_UR50D".

    Returns
    -------
    (model, alphabet, batch_converter)

    Raises
    ------
    ValueError
        If the requested model is not available in `esm.pretrained`.

    Notes
    -----
    • Moves model to `ESM_DEVICE`.
    • Converts to half precision if `USE_FP16=True`.
    • Sets the model to eval() mode.
    """

    # expects names like "esm2_t33_650M_UR50D"
    if not hasattr(esm.pretrained, name):
        raise ValueError(f"[S04] Unknown ESM model: {name}")
    model, alphabet = getattr(esm.pretrained, name)()
    model.eval()
    if USE_FP16:
        model.half()
    model.to(ESM_DEVICE)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

def _make_batches(lengths: list[int], max_seqs: int, max_tokens: int) -> list[tuple[int,int]]:
    """
    Greedy batching utility that packs sequences under token and count limits.

    Parameters
    ----------
    lengths : list[int]
        Raw sequence lengths.
    max_seqs : int
        Maximum number of sequences per batch.
    max_tokens : int
        Maximum total tokens (including BOS/EOS) per batch.

    Returns
    -------
    tuple[list[int], list[tuple[int, int]]]
        • order  – permutation indices sorted by descending length  
        • batches – list of (start_idx, end_idx) slices over that order
    """

    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches, start = [], 0
    while start < len(order):
        tot_tok = 0
        count = 0
        end = start
        while end < len(order):
            L = lengths[order[end]] + 2  # add BOS/EOS tokens
            if (count + 1) > max_seqs or (tot_tok + L) > max_tokens:
                break
            tot_tok += L
            count += 1
            end += 1
        if end == start:  # single very long sequence
            end = start + 1
        batches.append((start, end))
        start = end
    return order, batches

@torch.no_grad()
def embed():
    """
    Main embedding routine for Stage S04.

    Workflow
    --------
    1) Load all domain windows and compute sequence lengths.
    2) Initialize ESM-2 model, tokenizer, and batch converter.
    3) Build efficient batches via `_make_batches`.
    4) For each batch:
        • Tokenize and move to device.
        • Run the model, extract final-layer representations.
        • Mean-pool per sequence (excluding BOS/EOS).
    5) Save results:
        - NumPy array PROTEIN_EMBEDDINGS
        - JSON META_FILE with model info and window list.

    Outputs
    -------
    • PROTEIN_EMBEDDINGS (.npy)
    • META_FILE (.json)

    Notes
    -----
    All computations run under `torch.no_grad()`.
    Mean pooling yields one fixed-dimensional vector per window.
    """

    out_npy = Path(PROTEIN_EMBEDDINGS)
    out_meta = Path(META_FILE)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    wids, seqs = _load_windows(Path(DOMAIN_WINDOWS))
    lengths = [len(s) for s in seqs]

    model, alphabet, batch_converter = _load_model(ESM_MODEL_NAME)
    repr_layer = model.num_layers  # final layer

    order, batches = _make_batches(lengths, int(ESM_BATCH), int(ESM_MAX_TOKENS))

    D = model.embed_dim
    N = len(seqs)
    xb = np.empty((N, D), dtype=np.float32)

    # process batches
    for bstart, bend in batches:
        idxs = order[bstart:bend]
        data = [("win", seqs[i]) for i in idxs]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(ESM_DEVICE)
        if USE_FP16:
            tokens = tokens.half()

        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # [B, L, D]

        # mean-pooled per sequence over (1..L-2)
        for j, i in enumerate(idxs):
            L = lengths[i]
            rep = reps[j, 1 : 1 + L, :]  # strip BOS/EOS
            emb = rep.float().mean(dim=0)  # ensure float32 on CPU
            xb[i, :] = emb.detach().cpu().numpy()

    # write outputs
    np.save(out_npy, xb)

    meta = {
        "model": ESM_MODEL_NAME,
        "device": str(ESM_DEVICE),
        "fp16": bool(USE_FP16),
        "num_sequences": int(N),
        "dim": int(D),
        "windows": wids,  # index-aligned with embeddings.npy rows
        "source": str(DOMAIN_WINDOWS),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[S04] embeddings → {out_npy}  shape={xb.shape}")
    print(f"[S04] meta       → {out_meta}")

if __name__ == "__main__":
    embed()
