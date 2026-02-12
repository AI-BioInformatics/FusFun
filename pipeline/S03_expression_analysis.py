"""
S03_expression_analysis
=======================
Stage 03 — Extract expression metrics from Arriba-like PDF plots.

Overview
--------
Processes PDF pages that contain two stacked histograms per page
(top = gene1, bottom = gene2). For each page:
1) Render the page to an RGB image at EXP_DPI.
2) Crop the fixed ROI [EXP_ROI_*], locate the central white gap,
   and split into two histograms.
3) Detect the blue breakpoint X within each histogram.
4) Normalize grayscale so that bar fill gray equals EXP_SIGNAL_GRAY
   within tolerance EXP_GRAY_TOL.
5) For each side of each histogram, segment contiguous exon bars on
   the bottom row and compute per-exon mean and median "reads" by
   rescaling bar heights to MAX_PIXELS and the page’s max coverage.
6) Aggregate retained vs lost means/medians using transcript strand
   and write compact CSV rows plus optional JSON with page metadata.

Inputs
------
• A folder of PDF plots.
• HGNC gene list (for token filtering).
• ALL_LOOKUP table providing transcript_id → strand.
• Config flags for batch sizing and outputs.

Outputs
-------
• CSV (coverage metrics per page) when EXP_WRITE_CSV=True.
• JSON (page metadata and per-page metrics) when EXP_WRITE_JSON=True.
• Debug PNGs with breakpoints and exon boundaries when EXP_WRITE_DEBUG=True.

Notes
-----
• EXP_ITR selects the batch slice: files[(EXP_ITR−1)*EXP_BATCH_SIZE : EXP_ITR*EXP_BATCH_SIZE].
• The algorithm assumes consistent plot styling and ROI coordinates.
• Gray normalization snaps pixels near EXP_SIGNAL_GRAY to that exact value.
"""

from __future__ import annotations
from pathlib import Path
import re, gc, json, os
import numpy as np
import polars as pl
import fitz, pdfplumber, cv2

# ---- vision constants ----
EXP_DPI          = 150
EXP_ROI_Y1       = 250
EXP_ROI_Y2       = 334
EXP_ROI_X1       = 220
EXP_ROI_X2       = 1600
EXP_GAP_MARGIN   = 33
EXP_BLUE_THRESH  = 100
EXP_SIGNAL_GRAY  = 171
EXP_GRAY_TOL     = 40
MAX_PIXELS       = 84
EXP_ITR          = 1  # iteration index (1-based)

from qol.utilities import load_file, save_file  # TSV-aware
from config.config import (
    EXP_BATCH_SIZE, EXP_WRITE_CSV, EXP_WRITE_JSON, EXP_WRITE_DEBUG,
    HGNC, EXP_JSON_DIR, EXP_IMG_DIR, EXP_CSV_DIR, ALL_LOOKUP,
)

def _normalize_gray(gray: np.ndarray, target: int, tol: int) -> np.ndarray:
    """
    Snap grayscale values within tolerance to a target gray level.

    Parameters
    ----------
    gray : np.ndarray
        2D grayscale image.
    target : int
        Gray value to snap to (e.g., EXP_SIGNAL_GRAY).
    tol : int
        Allowed deviation around target.

    Returns
    -------
    np.ndarray
        Grayscale image with pixels in [target−tol, target+tol] set to target.
    """

    m = (gray >= target - tol) & (gray <= target + tol)
    out = gray.copy()
    out[m] = target
    return out

def _compute_exon_coverage(gray_img: np.ndarray, strand: str | None,
                           start_idx: int, max_height: int | None) -> tuple[dict, list[tuple[int,int]], list[float], list[float], bool]:
    """
    Compute exon regions and per-exon coverage statistics from a histogram slice.

    Parameters
    ----------
    gray_img : np.ndarray
        2D grayscale image of a histogram half (left or right of breakpoint).
    strand : str | None
        Transcript strand ('+' or '-'). Affects exon numbering direction.
    start_idx : int
        First exon index for numbering on this side.
    max_height : int | None
        Page max coverage value used to scale pixel counts to reads.
        If None, reads are set to 0.0.

    Returns
    -------
    tuple
        (
        ex_means: dict[str, float],            # {'exon_<n>': mean_reads}
        regions: list[tuple[int, int]],        # [(x_start, x_end), ...] exon spans
        means: list[float],                    # mean reads per exon
        medians: list[float],                  # median reads per exon
        split_inside: bool                     # True if an exon spans both ends
        )

    Method
    ------
    • Identify exon columns on the bottom row by equality to EXP_SIGNAL_GRAY.
    • Find contiguous runs as exon regions.
    • Count vertical pixels equal to EXP_SIGNAL_GRAY per column, map to reads
    via: reads = count * (max_height / MAX_PIXELS) when max_height is set.
    • Number exons left→right on '+' and right→left on '-'.
    """

    sig = EXP_SIGNAL_GRAY
    H, W = gray_img.shape
    bottom = gray_img[-1, :]
    is_sig = (bottom == sig).astype(int)
    tr = np.diff(is_sig)
    starts = np.where(tr == 1)[0] + 1
    ends   = np.where(tr == -1)[0] + 1
    if is_sig[0] == 1:  starts = np.insert(starts, 0, 0)
    if is_sig[-1] == 1: ends   = np.append(ends, W)
    regions = list(zip(starts, ends))
    split_inside = is_sig[0] == 1 and is_sig[-1] == 1

    ex_means: dict[str,float] = {}
    means, medians = [], []
    is_minus = (strand == "-")
    n = len(regions)
    for i,(x1,x2) in enumerate(regions):
        exon_no = start_idx + (n - i - 1) if is_minus else start_idx + i
        sl = gray_img[:, x1:x2]
        col_counts = np.sum(sl == sig, axis=0)
        if max_height:
            reads = (col_counts * (max_height / MAX_PIXELS)).astype(float)
        else:
            reads = np.zeros_like(col_counts, dtype=float)
        mean_r = round(float(reads.mean()), 2) if reads.size else 0.0
        med_r  = round(float(np.median(reads)), 2) if reads.size else 0.0
        ex_means[f"exon_{exon_no}"] = mean_r
        means.append(mean_r); medians.append(med_r)
    return ex_means, regions, means, medians, split_inside

def _detect_blue_x(hist_rgb: np.ndarray) -> int:
    """
    Locate the x-coordinate of the blue breakpoint marker in a histogram.

    Parameters
    ----------
    hist_rgb : np.ndarray
        RGB crop of a single histogram.

    Returns
    -------
    int
        X position of the breakpoint. Falls back to the histogram center if not found.

    Notes
    -----
    Computes a crude blue score: B − mean(R,G). Picks columns above EXP_BLUE_THRESH.
    """

    b,g,r = [hist_rgb[:,:,i].astype(np.int16) for i in (0,1,2)]
    blue_score = b - ((r + g) // 2)
    mx = np.max(blue_score, axis=0)
    idx = np.where(mx > EXP_BLUE_THRESH)[0]
    return int(np.mean(idx)) if idx.size else hist_rgb.shape[1] // 2

def _crop_and_split(img_rgb: np.ndarray):
    """
    Crop page ROI, find the central white gap, split into two histograms,
    and prepare grayscale halves around each detected breakpoint.

    Parameters
    ----------
    img_rgb : np.ndarray
        Full page RGB image.

    Returns
    -------
    tuple
        (
        g1L, g1R, g2L, g2R,        # grayscale halves after gray normalization
        hist1_rgb, hist2_rgb,      # RGB crops for debug visualization
        x1_blue, x2_blue           # breakpoint x positions in hist1 and hist2
        )

    Raises
    ------
    ValueError
        If ROI is empty or central gap cannot be found.

    Details
    -------
    • ROI is img[y1:y2, x1:x2] with EXP_ROI_* constants.
    • Central gap = longest run of near-white columns. Split at its center
    and trim by EXP_GAP_MARGIN.
    • Blue breakpoint X is detected separately in each histogram.
    • Produces left and right grayscale halves excluding a 2-px gap around X.
    """

    y1,y2,x1,x2 = EXP_ROI_Y1, EXP_ROI_Y2, EXP_ROI_X1, EXP_ROI_X2
    roi = img_rgb[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        raise ValueError("Empty ROI")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    col_mean = np.mean(gray, axis=0)
    is_gap = col_mean > 240
    # widest gap
    max_len = 0; s = e = -1; i = 0
    while i < len(is_gap):
        if is_gap[i]:
            j = i
            while j < len(is_gap) and is_gap[j]:
                j += 1
            if j - i > max_len:
                max_len = j - i; s, e = i, j
            i = j
        else:
            i += 1
    if s == -1:
        raise ValueError("No central gap found")
    split_x = (s + e) // 2
    gap = EXP_GAP_MARGIN
    hist1 = roi[:, :split_x - gap].copy()
    hist2 = roi[:, split_x + gap:].copy()

    x1_blue = _detect_blue_x(hist1)
    x2_blue = _detect_blue_x(hist2)

    g1L = _normalize_gray(cv2.cvtColor(hist1[:, :x1_blue], cv2.COLOR_BGR2GRAY), EXP_SIGNAL_GRAY, EXP_GRAY_TOL)
    g1R = _normalize_gray(cv2.cvtColor(hist1[:, x1_blue+2:], cv2.COLOR_BGR2GRAY), EXP_SIGNAL_GRAY, EXP_GRAY_TOL)
    g2L = _normalize_gray(cv2.cvtColor(hist2[:, :x2_blue], cv2.COLOR_BGR2GRAY), EXP_SIGNAL_GRAY, EXP_GRAY_TOL)
    g2R = _normalize_gray(cv2.cvtColor(hist2[:, x2_blue+2:], cv2.COLOR_BGR2GRAY), EXP_SIGNAL_GRAY, EXP_GRAY_TOL)

    return g1L, g1R, g2L, g2R, hist1, hist2, x1_blue, x2_blue

def _extract_meta(pdf_path: str, page_idx: int, valid_genes: set[str]) -> dict:
    """
    Extract gene, transcript, chromosome, breakpoints, and max coverage from a PDF page.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_idx : int
        Zero-based page index.
    valid_genes : set[str]
        Upper-cased HGNC symbols used to filter tokens.

    Returns
    -------
    dict
        {
        'gene1','gene2','transcript_id1','transcript_id2',
        'chr1','chr2','breakpoint1','breakpoint2','max_height'
        }

    Notes
    -----
    • ENST ids detected via regex.
    • Genes filtered by membership in `valid_genes`.
    • Coordinates parsed as "<chr>:<pos>".
    • max_height is parsed heuristically from the fifth text line when present.
    """

    with pdfplumber.open(pdf_path) as pdf:
        if page_idx >= len(pdf.pages):
            return {}
        text = pdf.pages[page_idx].extract_text() or ""
    transcripts = re.findall(r"(ENST\d{11})", text)
    gene_tokens = re.findall(r"\b[A-Z0-9\-]{3,}\b", text.upper())
    genes = [g for g in gene_tokens if g in valid_genes]
    coords = re.findall(r"([\dXYM]{1,2}:\d{5,})", text)
    c1 = coords[0] if len(coords)>0 else None
    c2 = coords[1] if len(coords)>1 else None
    chr1=bp1=chr2=bp2=None
    if c1 and ":" in c1: chr1, bp1 = c1.split(":")[0], int(c1.split(":")[1])
    if c2 and ":" in c2: chr2, bp2 = c2.split(":")[0], int(c2.split(":")[1])
    max_height = None
    lines = text.splitlines()
    if len(lines) >= 5:
        m = re.search(r"\b\d+\b", lines[4])
        if m: max_height = int(m.group(0))
    return {
        "gene1": genes[0] if len(genes)>0 else None,
        "gene2": genes[1] if len(genes)>1 else None,
        "transcript_id1": transcripts[0] if len(transcripts)>0 else None,
        "transcript_id2": transcripts[1] if len(transcripts)>1 else None,
        "chr1": chr1, "chr2": chr2, "breakpoint1": bp1, "breakpoint2": bp2,
        "max_height": max_height
    }

def _safe_mean(v): 
    """Return mean(v) rounded to 3 decimals, or 0.0 when v is empty."""
    return round(float(np.mean(v)), 3) if v else 0.0

def _load_valid_genes(hgnc_path: Path) -> set[str]:
    """
    Load and upper-case a set of valid gene symbols from a text file.

    Parameters
    ----------
    hgnc_path : Path
        One symbol per line.

    Returns
    -------
    set[str]
        Upper-cased symbols with blank lines removed.
    """

    with open(hgnc_path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}

# --- public API ---
def run(pdf_root: Path) -> tuple[Path | None, Path | None]:
    """
    Process a batch of PDFs and emit coverage metrics and optional JSON details.

    Parameters
    ----------
    pdf_root : Path
        Directory containing input PDFs.

    Returns
    -------
    tuple[Path | None, Path | None]
        (csv_path, json_path). Each is None if the corresponding EXP_WRITE_* flag is False.

    Behavior
    --------
    • Select files for this iteration using EXP_ITR and EXP_BATCH_SIZE.
    • For each page:
    - Render at EXP_DPI, crop ROI, split histograms, detect breakpoints.
    - Read strand from ALL_LOOKUP using transcript_id.
    - Compute retained vs lost means/medians using strand orientation.
    - Optionally draw debug PNG with breakpoints and exon boundaries.
    • Write a unique CSV of compact metrics and an optional JSON with full metadata.

    Side Effects
    ------------
    Creates output folders: EXP_JSON_DIR, EXP_IMG_DIR, EXP_CSV_DIR.
    Writes PNGs only when EXP_WRITE_DEBUG is True.
    """

    pdf_root = Path(pdf_root)
    EXP_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    EXP_CSV_DIR.mkdir(parents=True, exist_ok=True)

    valid_genes = _load_valid_genes(Path(HGNC))
    gtf = load_file(ALL_LOOKUP)  # need transcript_id, strand

    # Determine slice by EXP_ITR
    start_idx = (EXP_ITR - 1) * EXP_BATCH_SIZE
    end_idx   = EXP_ITR * EXP_BATCH_SIZE

    files = [p for p in sorted(pdf_root.iterdir()) if p.suffix.lower() == ".pdf"]
    files = files[start_idx:end_idx]

    csv_rows: list[dict] = []
    json_rows: list[dict] = []

    for pdf_path in files:
        doc = fitz.open(str(pdf_path))
        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=EXP_DPI)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = img[:, :, :3]

                meta = _extract_meta(str(pdf_path), page_idx, valid_genes)
                max_h = meta.get("max_height")
                g1L, g1R, g2L, g2R, h1rgb, h2rgb, bx1, bx2 = _crop_and_split(img)

                strand1 = strand2 = None
                if meta.get("transcript_id1"):
                    strand1 = gtf.filter(pl.col("transcript_id") == meta["transcript_id1"]).get_column("strand").first()
                if meta.get("transcript_id2"):
                    strand2 = gtf.filter(pl.col("transcript_id") == meta["transcript_id2"]).get_column("strand").first()

                # gene1
                m1_ret = m1_lost = md1_ret = md1_lost = 0.0
                ex1L = ex1R = []
                regs1L = regs1R = []
                if meta.get("transcript_id1"):
                    exL, regs1L, valsL, medsL, _ = _compute_exon_coverage(g1L, strand1, 1, max_h)
                    last_idx = int(list(exL.keys())[-1].split("_")[1]) if exL else 1
                    exR, regs1R, valsR, medsR, _ = _compute_exon_coverage(g1R, strand1, last_idx, max_h)
                    if strand1 == "+":
                        m1_ret, md1_ret = _safe_mean(valsL), _safe_mean(medsL)
                        m1_lost, md1_lost = _safe_mean(valsR), _safe_mean(medsR)
                    elif strand1 == "-":
                        m1_ret, md1_ret = _safe_mean(valsR), _safe_mean(medsR)
                        m1_lost, md1_lost = _safe_mean(valsL), _safe_mean(medsL)

                # gene2
                m2_ret = m2_lost = md2_ret = md2_lost = 0.0
                ex2L = ex2R = []
                regs2L = regs2R = []
                if meta.get("transcript_id2"):
                    exL2, regs2L, valsL2, medsL2, _ = _compute_exon_coverage(g2L, strand2, 1, max_h)
                    last_idx2 = int(list(exL2.keys())[-1].split("_")[1]) if exL2 else 1
                    exR2, regs2R, valsR2, medsR2, _ = _compute_exon_coverage(g2R, strand2, last_idx2, max_h)
                    if strand2 == "+":
                        m2_ret, md2_ret = _safe_mean(valsR2), _safe_mean(medsR2)
                        m2_lost, md2_lost = _safe_mean(valsL2), _safe_mean(medsL2)
                    elif strand2 == "-":
                        m2_ret, md2_ret = _safe_mean(valsL2), _safe_mean(medsL2)
                        m2_lost, md2_lost = _safe_mean(valsR2), _safe_mean(medsR2)

                row_compact = {
                    "Gene1": meta.get("gene1"), "Gene2": meta.get("gene2"),
                    "Chromosome1": meta.get("chr1"), "Chromosome2": meta.get("chr2"),
                    "Breakpoint1": meta.get("breakpoint1"), "Breakpoint2": meta.get("breakpoint2"),
                    "transcript_id1": meta.get("transcript_id1"), "transcript_id2": meta.get("transcript_id2"),
                    "mean_retained1": m1_ret, "mean_lost1": m1_lost,
                    "median_retained1": md1_ret, "median_lost1": md1_lost,
                    "coverage1": _safe_mean([m1_ret / m1_lost]) if m1_ret and m1_lost else 0.0,
                    "mean_retained2": m2_ret, "mean_lost2": m2_lost,
                    "median_retained2": md2_ret, "median_lost2": md2_lost,
                    "coverage2": _safe_mean([m2_ret / m2_lost]) if m2_ret and m2_lost else 0.0,
                }
                csv_rows.append(row_compact)
                json_rows.append({**meta, "strand1": strand1, "strand2": strand2, **row_compact})

                if EXP_WRITE_DEBUG:
                    EXP_IMG_DIR.mkdir(parents=True, exist_ok=True)
                    gap = 20
                    spacer = np.full((h1rgb.shape[0], gap, 3), 255, dtype=np.uint8)
                    combo = np.concatenate([h1rgb, spacer, h2rgb], axis=1)
                    off = h1rgb.shape[1] + gap
                    cv2.line(combo, (bx1, 0), (bx1, combo.shape[0]), (0, 0, 255), 2)
                    cv2.line(combo, (bx2 + off, 0), (bx2 + off, combo.shape[0]), (0, 0, 255), 2)
                    if strand1 == "+":
                        for x1, x2 in regs1L:
                            cv2.line(combo, (x1, 0), (x1, combo.shape[0]), (0, 255, 0), 1)
                            cv2.line(combo, (x2, 0), (x2, combo.shape[0]), (0, 255, 0), 1)
                    elif strand1 == "-":
                        for x1, x2 in regs1R:
                            cv2.line(combo, (x1 + bx1 + 2, 0), (x1 + bx1 + 2, combo.shape[0]), (0, 255, 0), 1)
                            cv2.line(combo, (x2 + bx1 + 2, 0), (x2 + bx1 + 2, combo.shape[0]), (0, 255, 0), 1)
                    if strand2 == "+":
                        for x1, x2 in regs2R:
                            cv2.line(combo, (x1 + bx2 + 2 + off, 0), (x1 + bx2 + 2 + off, combo.shape[0]), (0, 255, 0), 1)
                            cv2.line(combo, (x2 + bx2 + 2 + off, 0), (x2 + bx2 + 2 + off, combo.shape[0]), (0, 255, 0), 1)
                    elif strand2 == "-":
                        for x1, x2 in regs2L:
                            cv2.line(combo, (x1 + off, 0), (x1 + off, combo.shape[0]), (0, 255, 0), 1)
                            cv2.line(combo, (x2 + off, 0), (x2 + off, combo.shape[0]), (0, 255, 0), 1)
                    out_png = EXP_IMG_DIR / f"{pdf_path.stem}_page{page_idx+1}.png"
                    cv2.imwrite(str(out_png), cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

                del img, pix, page
                gc.collect()

            except Exception as e:
                print(f"Skipping {pdf_path.name} page {page_idx+1}: {e}")
                continue

    csv_path = json_path = None
    folder_tag = pdf_root.name
    if EXP_WRITE_CSV:
        df = pl.DataFrame(csv_rows).unique(maintain_order=True)
        csv_path = Path(EXP_CSV_DIR) / f"{folder_tag}_coverage_metrics_{EXP_ITR}.tsv"
        save_file(df, csv_path)

    if EXP_WRITE_JSON:
        json_path = Path(EXP_JSON_DIR) / f"{folder_tag}_in_depth_{EXP_ITR}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, indent=2)

    return csv_path, json_path