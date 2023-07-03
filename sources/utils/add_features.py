import os
import pandas as pd

from PIL import Image
from sources.utils.url_utils import get_image_path_from_url
from sources.classifier import Classifier
from sources.utils.distance import euclidean_distance, cosine_distance
from sources.utils.benchmarking import measure_time


@measure_time
def add_features(dataframe: pd.DataFrame):
    classifier = Classifier()
    processed_dataframe = dataframe.copy()

    for index, row in dataframe.iterrows():
        indexes = []
        embeddings = []

        for url in ["image_url1", "image_url2"]:
            image_path = get_image_path_from_url(row[url])
            if not os.path.exists(image_path):
                embeddings = []
                break
            image = Image.open(image_path)

            logits = classifier(image).detach().numpy()
            score = logits.argmax(-1).item()

            indexes.append(score)
            embeddings.append(logits)

        if not embeddings:
            indexes = [None, None]
            euclidean_similarity = None
            cosine_similarity = None
        else:
            euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])
            cosine_similarity = cosine_distance(embeddings[0], embeddings[1])

        processed_dataframe.loc[index, 'class_index1'] = indexes[0]
        processed_dataframe.loc[index, 'class_index2'] = indexes[1]
        processed_dataframe.loc[index, 'euclidean_similarity'] = euclidean_similarity
        processed_dataframe.loc[index, 'cosine_similarity'] = cosine_similarity

    return processed_dataframe


def main():
    df = pd.read_csv("../../data/test.csv")
    df = df.head(10)
    df = add_features(df)
    print(df)


if __name__ == "__main__":
    main()
