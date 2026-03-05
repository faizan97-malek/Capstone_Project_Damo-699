def risk_tier(prob, high_threshold, medium_threshold=None):
    """
    Assign a risk tier based on predicted failure probability.

    high_threshold  : the tuned decision threshold (from threshold.json)
    medium_threshold: defaults to half of high_threshold

    With Soft Voting threshold ~0.18:
      Low    : prob < 0.09  (< 9% on gauge)
      Medium : 0.09 <= prob < 0.18  (9-18% on gauge, approaching threshold)
      High   : prob >= 0.18  (at or above decision boundary)
    """
    if medium_threshold is None:
        medium_threshold = high_threshold * 0.5

    if prob >= high_threshold:
        return "High"
    elif prob >= medium_threshold:
        return "Medium"
    return "Low"