def score_candidate(measurements):
    """
    Calculate a diagnostic document-likeness score.

    This score does not determine whether a candidate
    is accepted by the detector.
    """

    fill_ratio = measurements["fill_ratio"]
    rectangularity = measurements["rectangularity"]
    area = measurements["area"]

    score = 0

    # ----------------------------
    # Fill ratio
    # ----------------------------

    if fill_ratio >= 0.90:
        score += 40

    elif fill_ratio >= 0.75:
        score += 30

    elif fill_ratio >= 0.60:
        score += 20

    elif fill_ratio >= 0.40:
        score += 10

    # ----------------------------
    # Rectangularity
    # ----------------------------

    if rectangularity >= 0.90:
        score += 40

    elif rectangularity >= 0.75:
        score += 30

    elif rectangularity >= 0.60:
        score += 20

    elif rectangularity >= 0.40:
        score += 10

    # ----------------------------
    # Area
    # ----------------------------

    if area >= 1_000_000:
        score += 20

    elif area >= 250_000:
        score += 15

    elif area >= 100_000:
        score += 10

    elif area >= 25_000:
        score += 5

    return score