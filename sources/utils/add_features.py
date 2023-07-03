import os
import pandas as pd

from PIL import Image
from sources.utils.url_utils import get_image_path_from_url
from sources.classifier import Classifier
from sources.utils.sift import SIFT, SIFTPoint
from sources.utils.distance import euclidean_distance, cosine_distance
from sources.utils.benchmarking import measure_time


@measure_time
def add_features(dataframe: pd.DataFrame):
    classifier = Classifier()
    sift = SIFT()
    processed_dataframe = dataframe.copy()

    for index, row in dataframe.iterrows():
        class_indexes = []
        embeddings = []
        sift_points = []

        for url in ["image_url1", "image_url2"]:
            image_path = get_image_path_from_url(row[url])

            if not os.path.exists(image_path):
                embeddings.clear()
                break

            image = Image.open(image_path)

            classifier_logits = classifier(image).detach().numpy()
            classifier_score = classifier_logits.argmax(-1).item()
            class_indexes.append(classifier_score)
            embeddings.append(classifier_logits)

            sift_point = sift(image)
            sift_points.append(sift_point)

        if not embeddings:
            class_indexes = [None, None]
            euclidean_similarity = None
            cosine_similarity = None
            sift_score = None
        else:
            euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])
            cosine_similarity = cosine_distance(embeddings[0], embeddings[1])
            sift_score = sift.get_score(sift_points[0], sift_points[1], approximate=False)

        processed_dataframe.loc[index, 'class_index1'] = class_indexes[0]
        processed_dataframe.loc[index, 'class_index2'] = class_indexes[1]
        processed_dataframe.loc[index, 'euclidean_similarity'] = euclidean_similarity
        processed_dataframe.loc[index, 'cosine_similarity'] = cosine_similarity
        processed_dataframe.loc[index, 'sift_score'] = sift_score

    return processed_dataframe


def main():
    df = pd.read_csv("../../data/test.csv")
    df = df.head(10)
    df = add_features(df)
    print(df)


if __name__ == "__main__":
    main()
