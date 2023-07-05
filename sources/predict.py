import numpy as np


def predict(features: np.ndarray) -> np.ndarray:
    criterion = (features[:, 1] == features[:, 2])
    predictions = np.ones(len(features), dtype=int) * criterion

    return predictions
