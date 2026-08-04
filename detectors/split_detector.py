import numpy as np


def find_vertical_gaps(mask, min_gap_width=20):
    """
    Find vertical gaps in a binary mask.
    """

    white_pixels = np.sum(mask == 255, axis=0)

    threshold = mask.shape[0] * 0.05

    gaps = []

    in_gap = False
    gap_start = 0

    for x, count in enumerate(white_pixels):

        if count < threshold:

            if not in_gap:
                gap_start = x
                in_gap = True

        else:

            if in_gap:

                if x - gap_start >= min_gap_width:
                    gaps.append((gap_start, x))

                in_gap = False

    if in_gap:

        if mask.shape[1] - gap_start >= min_gap_width:
            gaps.append((gap_start, mask.shape[1]))

    return gaps