import os
import pandas as pd

from PIL import Image
from sources.sift import SIFT
from sources.sam import SAM
from sources.transformerclassifier import TransformerClassifier
from sources.utils.url_utils import get_image_path_from_url
from sources.utils.distance import euclidean_distance, cosine_distance
from sources.utils.benchmarking import measure_time


TEXT_FILE_PATH = '../data/train_features.txt'
CSV_FILE_PATH = "../data/train_features.csv"


@measure_time
def add_features(dataframe: pd.DataFrame):
    # sift = SIFT()
    sam = SAM()
    classifier = TransformerClassifier()
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
            classifier_embeddings = []
            sam_embeddings = []
            # sift_points = []

            for url in ["image_url1", "image_url2"]:
                image_path = get_image_path_from_url(row[url])

                if not os.path.exists(image_path):
                    success = False
                    break

                image = Image.open(image_path)
                if 'A' in image.mode or image.mode in ('L', 'LA', 'P'):
                    image = image.convert('RGB')

                classifier_logits = classifier(image)
                sam_logits = sam(image)

                classifier_score = int(classifier_logits.argmax(-1).item())
                class_indexes.append(classifier_score)

                classifier_embeddings.append(classifier_logits)
                sam_embeddings.append(sam_logits)
                # sift_point = sift(image)
                # sift_points.append(sift_point)

            if success:
                classifier_euclidean_similarity = euclidean_distance(classifier_embeddings[0], classifier_embeddings[1])
                classifier_cosine_similarity = cosine_distance(classifier_embeddings[0], classifier_embeddings[1])

                sam_euclidean_similarity = euclidean_distance(sam_embeddings[0], sam_embeddings[1])
                sam_cosine_similarity = cosine_distance(sam_embeddings[0], sam_embeddings[1])
                # sift_score = sift.get_score(sift_points[0], sift_points[1], approximate=False)
            else:
                class_indexes = [None, None]
                classifier_euclidean_similarity = None
                classifier_cosine_similarity = None

                sam_euclidean_similarity = None
                sam_cosine_similarity = None
                # sift_score = None

            processed_dataframe.loc[index, 'class_index1'] = class_indexes[0]
            processed_dataframe.loc[index, 'class_index2'] = class_indexes[1]

            processed_dataframe.loc[index, 'classifier_euclidean_similarity'] = classifier_euclidean_similarity
            processed_dataframe.loc[index, 'classifier_cosine_similarity'] = classifier_cosine_similarity

            processed_dataframe.loc[index, 'sam_euclidean_similarity'] = sam_euclidean_similarity
            processed_dataframe.loc[index, 'sam_cosine_similarity'] = sam_cosine_similarity

            # processed_dataframe.loc[index, 'sift_score'] = sift_score

            is_same = processed_dataframe.loc[index, 'is_same']

            # file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{classifier_euclidean_similarity},{classifier_cosine_similarity},{sam_euclidean_similarity},{sam_cosine_similarity},{sift_score}\n")
            file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{classifier_euclidean_similarity},{classifier_cosine_similarity},{sam_euclidean_similarity},{sam_cosine_similarity},{is_same}\n")
            #file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{classifier_euclidean_similarity},{classifier_cosine_similarity},{sam_euclidean_similarity},{sam_cosine_similarity}\n")

    processed_dataframe.to_csv(CSV_FILE_PATH, index=False)
    return processed_dataframe


def dataframe_from_txt():
    with open(TEXT_FILE_PATH, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        values = line.strip().split(',')
        data.append(values)

    headers = ['ID', 'class_index1', 'class_index2', 'classifier_euclidean_similarity', 'classifier_cosine_similarity',
               'sam_euclidean_similarity', 'sam_cosine_similarity', 'is_same']
    # headers = ['ID', 'class_index1', 'class_index2', 'classifier_euclidean_similarity',
    # 'classifier_cosine_similarity', 'sam_euclidean_similarity', 'sam_cosine_similarity',]

    df = pd.DataFrame(data, columns=headers)

    df.to_csv(CSV_FILE_PATH, index=False)


def main():
    df = pd.read_csv("../../data/train.csv")
    df = add_features(df.head(1))
    df.to_csv(CSV_FILE_PATH)


if __name__ == "__main__":
    TEXT_FILE_PATH = '../../data/train_features.txt'
    CSV_FILE_PATH = "../../data/train_features.csv"
    main()
    # dataframe_from_txt()
