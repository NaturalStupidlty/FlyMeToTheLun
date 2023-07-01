import os
import re
import numpy as np
import pandas as pd

from PIL import Image
from sources.classifier import Classifier


FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray):
    squared_difference = np.square(embedding1 - embedding2)
    sum_squared_difference = np.sum(squared_difference)
    distance = np.sqrt(sum_squared_difference)

    return distance


def process_dataframe(dataframe: pd.DataFrame):
    classifier = Classifier()
    class_indexes = []
    similarities = []

    for index, row in dataframe.iterrows():
        indexes = []
        embeddings = []
        for url in ["image_url1", "image_url2"]:
            image_url = row[url]
            image_path = FILENAME_PATTERN.search(image_url).group(1)
            image_path = os.path.join("..", "..", "data", "images", image_path)
            image = Image.open(image_path)

            logits = classifier(image).detach().numpy()
            score = logits.argmax(-1).item()

            indexes.append(score)
            embeddings.append(logits)

        euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])

        class_indexes.append(indexes)
        similarities.append(euclidean_similarity)

    processed_dataframe = dataframe.copy()
    processed_dataframe.loc[:, "class_index1"] = [i[0] for i in class_indexes]
    processed_dataframe.loc[:, "class_index2"] = [i[1] for i in class_indexes]
    processed_dataframe.loc[:, "euclidean_similarity"] = similarities

    return processed_dataframe


def test():
    df = pd.read_csv("../../data/train.csv")
    df = process_dataframe(df.head(10))
    print(df[["class_index1", "class_index1", "is_same", "euclidean_similarity"]])


if __name__ == "__main__":
    test()
