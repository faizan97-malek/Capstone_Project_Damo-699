def risk_tier(prob, high_threshold, medium_threshold=0.10):
    if prob >= high_threshold:
        return "High"
    elif prob >= medium_threshold:
        return "Medium"
    return "Low"
