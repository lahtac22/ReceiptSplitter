"""
single_receipt.py

Single-receipt application for the ReceiptSplitter project.

Purpose:
    Process images/PDFs that contain ONE receipt only and save one clean
    receipt image for each input.

This deliberately does NOT attempt to split multiple receipts. It reuses
the proven single-receipt detection/cropping components from splitter_42.py.

Project layout expected:

    ReceiptSplitter/
        single_receipt.py
        Input/
        SingleReceipts/
        utils/
        image/
        detectors/
        paper_boundary_detector_span_aware.py

Run from the ReceiptSplitter folder with:

    python single_receipt.py

The input files are taken from the existing project's Input folder.
Outputs are written to SingleReceipts as PNG files.
"""

import time
from pathlib import Path

import cv2

from utils.files import get_input_files
from image.image_loader import load_image
from detectors.colour_detector import detect

from paper_boundary_detector_span_aware import find_paper_regions

from splitter_42 import (
    _remove_nested_candidate_rectangles,
    _paper_region_for_candidate,
    _conservative_outer_crop_bounds,
    _make_paper_mask,
    _crop_box,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

OUTPUT_FOLDER = Path("SingleReceipts")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _candidate_xywh(rectangle):
    """Accept the rectangle formats used by the existing detector."""
    if isinstance(rectangle, dict):
        x = int(rectangle.get("x", rectangle.get("left", 0)))
        y = int(rectangle.get("y", rectangle.get("top", 0)))
        w = int(rectangle.get("width", rectangle.get("w", 0)))
        h = int(rectangle.get("height", rectangle.get("h", 0)))
        return x, y, w, h

    x, y, w, h = rectangle[:4]
    return int(x), int(y), int(w), int(h)


def _area(rectangle):
    x, y, w, h = _candidate_xywh(rectangle)
    return max(0, w) * max(0, h)


def _safe_output_name(input_file):
    return OUTPUT_FOLDER / f"{input_file.stem}.png"


def process_single_receipt(input_file):
    """
    Process one input file.

    The important rule is that this function produces exactly ONE output.
    It never attempts to split the image into multiple receipts.
    """

    print()
    print("=" * 70)
    print(f"[INFO] Loading {input_file.name}")

    image = load_image(input_file)

    if image is None or image.size == 0:
        print("[ERROR] Could not load image.")
        return False

    print(f"[INFO] Image size: {image.shape}")

    # --------------------------------------------------------
    # Detect the receipt candidate(s).
    # --------------------------------------------------------

    rectangles, mask, contours, stats = detect(image)

    rectangles = _remove_nested_candidate_rectangles(rectangles)

    print(f"[INFO] Candidate regions after nested filtering: {len(rectangles)}")

    # --------------------------------------------------------
    # Find the physical paper regions.
    # --------------------------------------------------------

    (
    paper_regions,
    paper_mask,
    cleaned_paper_mask,
    paper_diagnostics,
) = find_paper_regions(image)

    print(f"[INFO] Paper regions found: {len(paper_regions)}")

    # --------------------------------------------------------
    # Select the strongest candidate.
    #
    # For this application the input is explicitly assumed to contain
    # one receipt. If the detector sees several pieces, we choose the
    # largest candidate rather than treating them as separate receipts.
    # --------------------------------------------------------

    if rectangles:
        candidate = max(rectangles, key=_area)
        cx, cy, cw, ch = _candidate_xywh(candidate)

        print(
            f"[INFO] Selected candidate: "
            f"x={cx}–{cx+cw}, y={cy}–{cy+ch}"
        )
    else:
        # Safe fallback: use the complete image.
        cx, cy, cw, ch = 0, 0, image.shape[1], image.shape[0]
        print("[INFO] No candidate detected; using complete image.")

    # --------------------------------------------------------
    # Prefer the paper region that best matches the candidate.
    #
    # This follows the philosophy already used in Splitter 42:
    # physical paper bounds are more reliable than text/contour bounds
    # for preserving the complete receipt.
    # --------------------------------------------------------

    paper_region = None
    paper_overlap = 0.0

    if paper_regions:
        paper_region, paper_overlap = _paper_region_for_candidate(
            paper_regions,
            (cx, cy, cw, ch),
        )

    if paper_region is not None and paper_overlap >= 0.50:
        px = int(paper_region.get("x", paper_region.get("left", 0)))
        py = int(paper_region.get("y", paper_region.get("top", 0)))
        pw = int(paper_region.get("width", 0))
        ph = int(paper_region.get("height", 0))

        cell_left = max(0, px)
        cell_top = max(0, py)
        cell_right = min(image.shape[1], px + pw)
        cell_bottom = min(image.shape[0], py + ph)

        print(
            f"[INFO] Using paper region: "
            f"x={cell_left}–{cell_right}, "
            f"y={cell_top}–{cell_bottom}, "
            f"overlap={paper_overlap:.3f}"
        )
    else:
        cell_left = cx
        cell_top = cy
        cell_right = cx + cw
        cell_bottom = cy + ch

        print("[INFO] Using candidate bounds for the receipt cell.")

    # --------------------------------------------------------
    # Conservative outer crop.
    #
    # This is intentionally the same conservative strategy used by
    # Splitter 42: trimming obvious background is acceptable, but
    # losing receipt content is not.
    # --------------------------------------------------------

    paper_mask = _make_paper_mask(image)

    left, top, right, bottom = _conservative_outer_crop_bounds(
        image,
        cell_left,
        cell_top,
        cell_right,
        cell_bottom,
        paper_mask=paper_mask,
    )

    crop = _crop_box(
        image,
        left,
        top,
        right,
        bottom,
    )

    if crop is None:
        print("[ERROR] Failed to create receipt crop.")
        return False

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_file = _safe_output_name(input_file)

    if not cv2.imwrite(str(output_file), crop):
        print(f"[ERROR] Could not write {output_file}")
        return False

    print(
        f"[OK] Saved single receipt: {output_file}"
    )
    print(
        f"[OK] Output size: {crop.shape[1]} x {crop.shape[0]}"
    )

    return True


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = get_input_files()

    if not files:
        print()
        print("[INFO] No input files found in Input.")
        return

    print()
    print(f"Found {len(files)} input file(s).")
    print("Single-receipt mode: one input -> one output.")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")

    start = time.perf_counter()

    succeeded = 0
    failed = 0

    for input_file in files:
        try:
            if process_single_receipt(input_file):
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(
                f"[ERROR] {input_file.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("[SUMMARY]")
    print(f"Successful: {succeeded}")
    print(f"Failed:     {failed}")
    print(f"Total time: {elapsed:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()


