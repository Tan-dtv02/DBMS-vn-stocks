import json
from pathlib import Path


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def add_prediction_signal(predicted_return):
    if predicted_return >= 0.03:
        return "STRONG_BUY"
    elif predicted_return >= 0.01:
        return "BUY"
    elif predicted_return > -0.01:
        return "HOLD"
    elif predicted_return > -0.03:
        return "SELL"
    else:
        return "STRONG_SELL"