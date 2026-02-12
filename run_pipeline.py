#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import polars as pl

from pipeline.S01_preprocessing import run as run_pre
from pipeline.S02_domain_retriever import run as run_faiss
from pipeline.S02b_run_interpro import run as run_ips
from pipeline.S03_expression_analysis import run as run_expr
from pipeline.S04_feature_analysis import run as run_feat
from pipeline.S05_functional_annotation import run as run_event
from pipeline.S06_oncogenicity_scoring import run as run_score

from config.config import INTERIM, EXP_WRITE_DEBUG, EXP_IMG_DIR, FINAL
from qol.utilities import load_file, save_file, stage_path, _assert_unique_id, _join_new_columns

def parse_args():
    p = argparse.ArgumentParser(prog="run_pipeline")
    p.add_argument("--input", required=True, help="Input filename under input/ or absolute path")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--faiss", action="store_true", help="Use FAISS for domain retrieval (default)")
    g.add_argument("--ips",   action="store_true", help="Use InterProScan for domain retrieval")
    p.add_argument("--gpu", type=int, default=None,
                   help="Number of GPUs to use (Slurm sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--arriba", metavar="PDF_DIR", help="Folder containing Arriba plot PDFs to analyze")
    p.add_argument("--delinter", action="store_true", help="Delete all intermediate files under output/interim after the pipeline completes")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- S01
    peps_csv, peps_fasta = run_pre(args.input)
    print(f"[S01] peptides: {peps_csv}")
    print(f"[S01] fasta:    {peps_fasta}")

    # ---- S02
    if args.ips:
        print("[S02] mode: InterProScan")
        dom_csv = run_ips(peps_csv)
    else:
        print("[S02] mode: FAISS")
        dom_csv = run_faiss(peps_csv, gpu_count=args.gpu)

    # ---- S03 (optional)
    csv_path = None
    if args.arriba:
        csv_path, json_path = run_expr(Path(args.arriba))
        if csv_path:  print(f"[S03] CSV:   {csv_path}")
        if json_path: print(f"[S03] JSON:  {json_path}")
        if EXP_WRITE_DEBUG: print(f"[S03] DEBUG IMAGES: {EXP_IMG_DIR}")

    # ---- S04
    feat_csv = run_feat(dom_csv)
    print(f"[S04] features: {feat_csv}")

    # ---- S05
    event_csv = run_event(Path(feat_csv), expression=bool(args.arriba))
    print(f"[S05] annotated: {event_csv}")

    # ---- S06
    scores_csv = run_score(Path(event_csv))

    # ---- Load per-stage tables
    s1 = load_file(peps_csv,  ["id"])
    s2 = load_file(dom_csv,   ["id"])
    s4 = load_file(feat_csv,  ["id"])
    s5 = load_file(event_csv, ["id"])
    s6 = load_file(scores_csv,["id"])

    # Optional S03
    s3 = load_file(csv_path) if csv_path else None

    # ---- Build complete table: original input columns first (from s1), then add new cols
    # Start from s1 verbatim
    _assert_unique_id(s1, "S01/base")
    complete = s1

    # Add new columns stage by stage
    complete = _join_new_columns(complete, s2, "S02")
    if s3 is not None:
        complete = complete.join(s3, on=["gene1", "gene2", "transcript_id1", "transcript_id2", "chromosome1", "chromosome2", "breakpoint1", "breakpoint2"], how="left")
    complete = _join_new_columns(complete, s4, "S04")
    complete = _join_new_columns(complete, s5, "S05")
    complete = _join_new_columns(complete, s6, "S06")

    # ---- Save final CSV using your helpers and naming
    complete_path = stage_path(FINAL, args.input, "complete_analysis", ".tsv")
    save_file(complete, complete_path)
    print(f"[FINAL] complete analysis: {complete_path}")

    # ---- Optional deletion of intermediate files
    if args.delinter:
        interim_dir = INTERIM
        if interim_dir.exists():
            print(f"[CLEANUP] Deleting all files inside {interim_dir} (keeping folders).")
            for path in interim_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            print("[CLEANUP] All intermediate files deleted, folder structure preserved.")
        else:
            print(f"[CLEANUP] No interim folder found at {interim_dir}")

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
