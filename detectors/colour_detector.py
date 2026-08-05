import cv2


MIN_CONTOUR_AREA = 10000


def detect(image):
    """
    Detect likely receipt regions.

    Returns:
        rectangles
        mask
        contours
        stats
    """

    # ----------------------------
    # Convert to grayscale
    # ----------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------
    # Otsu threshold
    # ----------------------------

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ----------------------------
    # Close small holes
    # ----------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ----------------------------
    # Find contours
    # ----------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    rectangles = []

    contour_areas = []

    accepted = 0
    rejected = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        contour_areas.append(area)

        if area < MIN_CONTOUR_AREA:

            rejected += 1
            continue

        x, y, w, h = cv2.boundingRect(contour)

        rectangles.append(
            (x, y, w, h)
        )

        accepted += 1

    # ----------------------------
    # Statistics
    # ----------------------------

    if contour_areas:

        largest = max(contour_areas)
        smallest = min(contour_areas)
        average = sum(contour_areas) / len(contour_areas)

    else:

        largest = 0
        smallest = 0
        average = 0

    stats = {

        "threshold": "Otsu",

        "contours_found": len(contours),

        "accepted_contours": accepted,

        "rejected_contours": rejected,

        "largest_contour": int(largest),

        "smallest_contour": int(smallest),

        "average_contour": int(average)

    }

    return (
        rectangles,
        mask,
        contours,
        stats
    )