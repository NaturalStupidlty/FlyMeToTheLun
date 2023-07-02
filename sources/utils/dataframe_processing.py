import os
import re
import cv2
import numpy as np
import pandas as pd
import albumentations as a

from PIL import Image
from sources.classifier import Classifier
from sources.utils.augmentation import Augmentation
from sources.utils.albumentation_transform import AlbumentationTransform
from sources.utils.benchmarking import measure_time


FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")


def get_image_path_from_url(url: str):
    filename = FILENAME_PATTERN.search(url).group(1)
    image_path = os.path.join("..", "..", "data", "images", filename)

    return image_path


def find_missing_images(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_images = []

    for index, row in dataframe.iterrows():
        image_path1 = get_image_path_from_url(row["image_url1"])
        image_path2 = get_image_path_from_url(row["image_url2"])

        if not os.path.exists(image_path1) or not os.path.exists(image_path2):
            missing_images.append(row)

    missing_images_dataframe = pd.DataFrame(missing_images)
    return missing_images_dataframe


def fill_images(dataframe: pd.DataFrame, augmentation: Augmentation, save_path: str) -> pd.DataFrame:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    missing_images_dataframe = find_missing_images(dataframe)
    print(missing_images_dataframe[["image_url1", "image_url2"]])

    filled_dataframe = dataframe.copy()
    filled_dataframe["missing"] = False

    for index, row in filled_dataframe.iterrows():
        if index in missing_images_dataframe.index:
            image_path1 = get_image_path_from_url(row["image_url1"])
            image_path2 = get_image_path_from_url(row["image_url2"])

            if not os.path.exists(image_path1):
                if os.path.exists(image_path2):
                    image1 = Image.open(image_path2)
                    image1 = augmentation(image1)

                    filename = FILENAME_PATTERN.search(row["image_url1"]).group(1)
                    image1.save(os.path.join(save_path, filename))

                    filled_dataframe.loc[index, "is_same"] = True
                else:
                    filled_dataframe.loc[index, "missing"] = True
                    continue

            if not os.path.exists(image_path2):
                image2 = Image.open(image_path1)
                image2 = augmentation(image2)

                filename = FILENAME_PATTERN.search(row["image_url2"]).group(1)
                image2.save(os.path.join(save_path, filename))

                filled_dataframe.loc[index, "is_same"] = True

    return filled_dataframe


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


@measure_time
def add_features(dataframe: pd.DataFrame):
    classifier = Classifier()
    processed_dataframe = dataframe.copy()

    processed_dataframe['class_index1'] = []
    processed_dataframe['class_index2'] = []
    processed_dataframe['euclidean_similarity'] = []
    processed_dataframe['cosine_similarity'] = []

    for _, row in dataframe.iterrows():
        indexes = []
        embeddings = []

        for url in ["image_url1", "image_url2"]:
            image_path = get_image_path_from_url(row[url])
            image = Image.open(image_path)

            logits = classifier(image).detach().numpy()
            score = logits.argmax(-1).item()

            indexes.append(score)
            embeddings.append(logits)

        euclidean_similarity = euclidean_distance(embeddings[0], embeddings[1])
        cosine_similarity = cosine_distance(embeddings[0], embeddings[1])

        processed_dataframe['class_index1'].append(indexes[0])
        processed_dataframe['class_index2'].append(indexes[1])
        processed_dataframe['euclidean_similarity'].append(euclidean_similarity)
        processed_dataframe['cosine_similarity'].append(cosine_similarity)

    return processed_dataframe


def main():
    df = pd.read_csv("../../data/train.csv")
    # df = add_features(df.head(10))
    # print(df[["class_index1", "class_index1", "is_same", "euclidean_similarity", "cosine_similarity"]])
    composition = a.Compose([
        a.Rotate(limit=10, p=0.9),
        a.Blur(p=0.8, blur_limit=10),
        a.GaussNoise(p=0.9, var_limit=(10.0, 100.0), mean=0, per_channel=True),
        a.ISONoise(p=0.9, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        a.Downscale(p=0.9, scale_min=0.1, scale_max=0.2, interpolation=cv2.INTER_AREA),
        a.ImageCompression(p=0.9, quality_lower=99, quality_upper=100),
    ])
    albumentations = AlbumentationTransform(composition)
    df = fill_images(df, albumentations, "../../data/lmao")


if __name__ == "__main__":
    main()
