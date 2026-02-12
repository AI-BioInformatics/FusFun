# qol/domain_retrieval/s05_build_index.py
#!/usr/bin/env python3
"""
S05_build_index.py
==================
Stage S05 of the domain-retrieval pipeline.  
Constructs a FAISS similarity index from precomputed domain-window embeddings.

Overview
--------
Input:
  • PROTEIN_EMBEDDINGS (.npy) — float32 array of shape [N, D]
Process:
  1) Load embeddings from NumPy file.
  2) Initialize a FAISS index with the chosen metric:
       - "ip" → inner product (cosine similarity)
       - "l2" → Euclidean distance
  3) Add all vectors to the index.
  4) Write the serialized index to disk.

Output:
  • FAISS_DIR / <index_name>.index

Config keys used
----------------
PROTEIN_EMBEDDINGS, FAISS_DIR, FAISS_INDEX, FAISS_METRIC

Notes
-----
• IndexFlatIP gives exact cosine similarity if inputs are normalized.  
• The index filename is stored under FAISS_DIR and must be a plain name
  (no path separators).  
• Supports standalone CLI execution via `--index <filename>`.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import faiss

from config.config import (
    PROTEIN_EMBEDDINGS,
    FAISS_DIR,
    FAISS_INDEX,
    FAISS_METRIC,
)

def build_index(index_name: str):
  """
  Build a FAISS index from saved embedding vectors.

  Parameters
  ----------
  index_name : str
      Output index filename (stored under FAISS_DIR).  
      Must not contain directory separators.

  Raises
  ------
  ValueError
      If the index_name is invalid or contains a path.
  FileNotFoundError
      If the embeddings file does not exist.

  Behavior
  --------
  1) Loads NumPy array from PROTEIN_EMBEDDINGS (float32).  
  2) Chooses metric:
      • FAISS_METRIC == "ip" → IndexFlatIP  
      • otherwise → IndexFlatL2  
  3) Adds all vectors and writes index to FAISS_DIR/index_name.  
  4) Prints summary with vector count, dimension, and metric used.
  """

  if not index_name or "/" in index_name or "\\" in index_name:
      raise ValueError("S05: index_name must be a plain filename (no path).")
  out_path = Path(FAISS_DIR) / index_name
  out_path.parent.mkdir(parents=True, exist_ok=True)

  xb = np.load(PROTEIN_EMBEDDINGS).astype(np.float32, copy=False)
  d = xb.shape[1]
  index = faiss.IndexFlatIP(d) if FAISS_METRIC == "ip" else faiss.IndexFlatL2(d)
  index.add(xb)
  faiss.write_index(index, str(out_path))
  print(f"[S05] index → {out_path}  vectors={xb.shape[0]} dim={d} metric={FAISS_METRIC}")

