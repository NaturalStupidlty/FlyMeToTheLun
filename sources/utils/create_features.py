import os
import pandas as pd

from PIL import Image
from sources.sift import SIFT
from sources.classifier import Classifier
from sources.utils.url_utils import get_image_path_from_url
from sources.utils.distance import euclidean_distance, cosine_distance
from sources.utils.benchmarking import measure_time


TEXT_FILE_PATH = '../data/train_features.txt'
CSV_FILE_PATH = "../data/train_features.csv"


@measure_time
def add_features(dataframe: pd.DataFrame):
    #sift = SIFT()
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
                if classifier.device != 'cpu':
                    classifier_logits = classifier_logits.cpu()
                classifier_logits = classifier_logits.detach().numpy()
                classifier_score = classifier_logits.argmax(-1).item()
                class_indexes.append(classifier_score)
                embeddings.append(classifier_logits)
                # sift_point = sift(image)
                # sift_points.append(sift_point)

            if success:
                euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])
                cosine_similarity = cosine_distance(embeddings[0], embeddings[1])
                # sift_score = sift.get_score(sift_points[0], sift_points[1], approximate=False)
            else:
                class_indexes = [None, None]
                euclidean_similarity = None
                cosine_similarity = None
                # sift_score = None

            processed_dataframe.loc[index, 'class_index1'] = class_indexes[0]
            processed_dataframe.loc[index, 'class_index2'] = class_indexes[1]
            processed_dataframe.loc[index, 'euclidean_similarity'] = euclidean_similarity
            processed_dataframe.loc[index, 'cosine_similarity'] = cosine_similarity
            # processed_dataframe.loc[index, 'sift_score'] = sift_score
            is_same = processed_dataframe.loc[index, 'is_same']

            # file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{euclidean_similarity},{cosine_similarity},{sift_score}\n")
            file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{euclidean_similarity},{cosine_similarity},{is_same}\n")
            #file.write(f"{index},{class_indexes[0]},{class_indexes[1]},{euclidean_similarity},{cosine_similarity}\n")

    processed_dataframe.to_csv(CSV_FILE_PATH, index=False)
    return processed_dataframe


def dataframe_from_txt():
    with open(TEXT_FILE_PATH, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        values = line.strip().split(',')
        data.append(values)

    headers = ['ID', 'class_index1', 'class_index2', 'euclidean_similarity', 'cosine_similarity', 'is_same']
    # headers = ['ID', 'class_index1', 'class_index2', 'euclidean_similarity', 'cosine_similarity']

    df = pd.DataFrame(data, columns=headers)

    df.to_csv(CSV_FILE_PATH, index=False)


def main():
    df = pd.read_csv("../../data/train.csv")
    df = add_features(df)
    df.to_csv(CSV_FILE_PATH)


if __name__ == "__main__":
    TEXT_FILE_PATH = '../../data/train_features.txt'
    CSV_FILE_PATH = "../../data/train_features.csv"
    main()
    # dataframe_from_txt()
