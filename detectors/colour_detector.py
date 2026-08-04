import cv2


def detect(image):
    """
    Detect likely receipt regions.

    Returns:
        rectangles: List of (x, y, w, h)
        mask: Thresholded image used for detection
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Automatically determine the best threshold
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Fill small holes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny blobs
        if area < 10000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        rectangles.append((x, y, w, h))

    return rectangles, mask