import os
import pandas as pd

from PIL import Image
from sources.sift import SIFT
from sources.classifier import Classifier
from sources.utils.url_utils import get_image_path_from_url
from sources.utils.distance import euclidean_distance, cosine_distance
from sources.utils.benchmarking import measure_time


TEXT_FILE_PATH = '../../data/results.txt'
CSV_FILE_PATH = "../../data/results.csv"


@measure_time
def add_features(dataframe: pd.DataFrame):
    sift = SIFT()
    classifier = Classifier()
    processed_dataframe = dataframe.copy()

    start_index = 0
    if os.path.exists(TEXT_FILE_PATH):
        with open(TEXT_FILE_PATH, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                start_index = int(last_line.split(',')[0]) + 1

    with open(TEXT_FILE_PATH, 'a') as file:
        for index, row in dataframe.iterrows():
            if index < start_index:
                continue

            print(f"Processing entry number {index}...")
            success = True
            class_indexes = []
            embeddings = []
            sift_points = []

            for url in ["image_url1", "image_url2"]:
                image_path = get_image_path_from_url(row[url])

                if not os.path.exists(image_path):
                    success = False
                    break

                image = Image.open(image_path)
                if 'A' in image.mode or image.mode in ('L', 'LA', 'P'):
                    image = image.convert('RGB')
                classifier_logits = classifier(image).detach().numpy()
                classifier_score = classifier_logits.argmax(-1).item()
                class_indexes.append(classifier_score)
                embeddings.append(classifier_logits)

                sift_point = sift(image)
                sift_points.append(sift_point)

            if success:
                euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])
                cosine_similarity = cosine_distance(embeddings[0], embeddings[1])
                sift_score = sift.get_score(sift_points[0], sift_points[1], approximate=False)
            else:
                class_indexes = [None, None]
                euclidean_similarity = None
                cosine_similarity = None
                sift_score = None

            processed_dataframe.loc[index, 'class_index1'] = class_indexes[0]
            processed_dataframe.loc[index, 'class_index2'] = class_indexes[1]
            processed_dataframe.loc[index, 'euclidean_similarity'] = euclidean_similarity
            processed_dataframe.loc[index, 'cosine_similarity'] = cosine_similarity
            processed_dataframe.loc[index, 'sift_score'] = sift_score

            file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{euclidean_similarity},{cosine_similarity},{sift_score}\n")

    processed_dataframe.to_csv(CSV_FILE_PATH, index=False)
    return processed_dataframe


def main():
    df = pd.read_csv("../../data/test.csv")
    df = add_features(df)
    df.to_csv(CSV_FILE_PATH)


if __name__ == "__main__":
    main()
