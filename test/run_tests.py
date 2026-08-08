from pathlib import Path
import sys


# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from image.image_loader import load_image
from detectors.colour_detector import detect
from utils.scoring import score_candidate


TEST_IMAGE_FOLDER = Path(__file__).parent / "images"

MIN_CONTOUR_AREA = 10000


def print_measurements(index, measurements):

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
        f"           Perimeter      : "
        f"{measurements['perimeter']:,.0f} px"
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


def main():

    files = sorted(
        file
        for file in TEST_IMAGE_FOLDER.iterdir()
        if file.is_file()
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

            image = load_image(file)

            rectangles, mask, contours, stats = detect(
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

                if measurements["area"] < MIN_CONTOUR_AREA:
                    continue

                candidate_index += 1

                print_measurements(
                    candidate_index,
                    measurements
                )

        except Exception as error:

            print(
                f"       ERROR: {error}"
            )

            print()


if __name__ == "__main__":
    main()