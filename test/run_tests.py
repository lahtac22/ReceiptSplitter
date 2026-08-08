from pathlib import Path
import sys

import cv2


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# Imports
# ============================================================

from image.image_loader import load_image

from detectors.colour_detector import detect

from detectors.paper_boundary_detector import (
    find_paper_regions,
    find_internal_boundaries,
    draw_paper_regions,
    draw_internal_boundaries
)

from utils.scoring import score_candidate


# ============================================================
# Folders
# ============================================================

TEST_IMAGE_FOLDER = (
    Path(__file__).parent / "images"
)

TEST_RESULTS_FOLDER = (
    Path(__file__).parent / "results"
)

PAPER_BOUNDARY_RESULTS_FOLDER = (
    TEST_RESULTS_FOLDER
    / "paper_boundaries"
)


# ============================================================
# Existing candidate settings
# ============================================================

MIN_CONTOUR_AREA = 10000


# ============================================================
# Candidate reporting
# ============================================================

def print_measurements(
    index,
    measurements
):

    print(
        f"       Candidate {index}"
    )

    print(
        f"           Area           : "
        f"{measurements['area']:,.0f} px²"
    )

    print(
        f"           Width          : "
        f"{measurements['width']} px"
    )

    print(
        f"           Height         : "
        f"{measurements['height']} px"
    )

    print(
        f"           Aspect Ratio   : "
        f"{measurements['aspect_ratio']:.3f}"
    )

    print(
        f"           Orientation AR : "
        f"{measurements['orientation_aspect_ratio']:.3f}"
    )

    print(
        f"           Angle          : "
        f"{measurements['angle']:.2f}°"
    )

    print(
        f"           Fill Ratio     : "
        f"{measurements['fill_ratio']:.3f}"
    )

    print(
        f"           Rectangularity : "
        f"{measurements['rectangularity']:.3f}"
    )

    score = score_candidate(
        measurements
    )

    print(
        f"           Score          : "
        f"{score}/100"
    )

    print()


# ============================================================
# Paper-boundary reporting
# ============================================================

def print_paper_regions(
    regions
):

    print(
        f"       Paper regions: "
        f"{len(regions)}"
    )

    for index, region in enumerate(
        regions,
        start=1
    ):

        print(
            f"           Paper {index}: "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"w={region['width']}, "
            f"h={region['height']}, "
            f"area={region['area']:,.0f}, "
            f"score={region['score']:.3f}"
        )

    print()


def print_boundaries(
    boundaries
):

    print(
        f"       Paper boundaries: "
        f"{len(boundaries)}"
    )

    for index, boundary in enumerate(
        boundaries,
        start=1
    ):

        print(
            f"           Boundary {index}: "
            f"{boundary['orientation']}, "
            f"position={boundary['position']}, "
            f"width={boundary['width']}, "
            f"confidence="
            f"{boundary['confidence']:.3f}"
        )

    print()


# ============================================================
# Main test runner
# ============================================================

def main():

    files = sorted(
        file
        for file in TEST_IMAGE_FOLDER.iterdir()
        if file.is_file()
    )

    PAPER_BOUNDARY_RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("ReceiptSplitter Test Suite")
    print("=" * 60)
    print()

    print(
        f"Found {len(files)} test file(s)."
    )

    print()

    for file in files:

        print(
            f"[TEST] {file.name}"
        )

        try:

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = load_image(
                file
            )

            # ------------------------------------------------
            # Existing contour detection
            # ------------------------------------------------

            (
                rectangles,
                mask,
                contours,
                stats
            ) = detect(
                image
            )

            print(
                f"       Contours  : "
                f"{stats['contours_found']}"
            )

            print(
                f"       Accepted  : "
                f"{stats['accepted_contours']}"
            )

            print(
                f"       Candidates: "
                f"{len(rectangles)}"
            )

            print()

            candidate_index = 0

            for measurements in stats[
                "contour_measurements"
            ]:

                if (
                    measurements["area"]
                    < MIN_CONTOUR_AREA
                ):
                    continue

                candidate_index += 1

                print_measurements(
                    candidate_index,
                    measurements
                )

            # =================================================
            # PAPER-BOUNDARY EXPERIMENT
            # =================================================

            print(
                "       [PAPER] Running "
                "paper-boundary analysis..."
            )

            # ------------------------------------------------
            # Find paper regions
            # ------------------------------------------------

            (
                paper_regions,
                paper_mask,
                cleaned_paper_mask,
                paper_diagnostics
            ) = find_paper_regions(
                image
            )

            print(
                f"       [PAPER] Regions found: "
                f"{paper_diagnostics['regions_found']}"
            )

            print(
                f"       [PAPER] Paper pixels: "
                f"{paper_diagnostics['paper_pixels']}"
            )

            print()

            print_paper_regions(
                paper_regions
            )

            # ------------------------------------------------
            # Find internal boundaries
            # ------------------------------------------------

            all_boundaries = []

            for region in paper_regions:

                (
                    boundaries,
                    boundary_diagnostics
                ) = find_internal_boundaries(
                    image,
                    region
                )

                all_boundaries.extend(
                    boundaries
                )

            print_boundaries(
                all_boundaries
            )

            # ------------------------------------------------
            # Create diagnostic images
            # ------------------------------------------------

            base_name = file.stem

            paper_mask_file = (
                PAPER_BOUNDARY_RESULTS_FOLDER
                / f"{base_name}_paper_mask.png"
            )

            cleaned_mask_file = (
                PAPER_BOUNDARY_RESULTS_FOLDER
                / f"{base_name}_paper_cleaned.png"
            )

            regions_file = (
                PAPER_BOUNDARY_RESULTS_FOLDER
                / f"{base_name}_paper_regions.png"
            )

            boundaries_file = (
                PAPER_BOUNDARY_RESULTS_FOLDER
                / f"{base_name}_paper_boundaries.png"
            )

            # ------------------------------------------------
            # Paper mask
            # ------------------------------------------------

            cv2.imwrite(
                str(paper_mask_file),
                paper_mask
            )

            # ------------------------------------------------
            # Cleaned paper mask
            # ------------------------------------------------

            cv2.imwrite(
                str(cleaned_mask_file),
                cleaned_paper_mask
            )

            # ------------------------------------------------
            # Paper regions
            # ------------------------------------------------

            regions_preview = (
                draw_paper_regions(
                    image,
                    paper_regions
                )
            )

            cv2.imwrite(
                str(regions_file),
                regions_preview
            )

            # ------------------------------------------------
            # Paper boundaries
            # ------------------------------------------------

            boundaries_preview = (
                draw_internal_boundaries(
                    image,
                    all_boundaries
                )
            )

            cv2.imwrite(
                str(boundaries_file),
                boundaries_preview
            )

            print(
                "[PAPER] Saved diagnostic images:"
            )

            print(
                f"        {paper_mask_file}"
            )

            print(
                f"        {cleaned_mask_file}"
            )

            print(
                f"        {regions_file}"
            )

            print(
                f"        {boundaries_file}"
            )

            print()

        except Exception as error:

            print(
                f"       ERROR: {error}"
            )

            print()


if __name__ == "__main__":
    main()