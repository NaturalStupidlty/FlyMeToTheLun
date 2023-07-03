import numpy as np


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray):
    squared_difference = np.square(embedding1 - embedding2)
    sum_squared_difference = np.sum(squared_difference)
    distance = np.sqrt(sum_squared_difference)

    return distance


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray):
    dot_product = np.dot(embedding1, embedding2.T)
    normalised_embedding1 = np.linalg.norm(embedding1)
    normalised_embedding2 = np.linalg.norm(embedding2)
    similarity = dot_product / (normalised_embedding1 * normalised_embedding2)

    return similarity[0][0]
