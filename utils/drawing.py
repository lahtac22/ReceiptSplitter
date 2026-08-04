import cv2


def draw_rectangles(image, rectangles):
    """
    Draw red rectangles around detected receipts.
    """

    preview = image.copy()

    for x, y, w, h in rectangles:

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            5
        )

    return preview


def draw_vertical_gaps(image, gaps):
    """
    Draw blue vertical lines showing detected gaps.
    """

    preview = image.copy()

    for start, end in gaps:

        x = (start + end) // 2

        cv2.line(
            preview,
            (x, 0),
            (x, preview.shape[0]),
            (255, 0, 0),
            3
        )

    return preview


def draw_contours(image, mask):
    """
    Draw every contour found in the mask.
    """

    preview = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        preview,
        contours,
        -1,
        (0, 255, 0),
        3
    )

    return preview