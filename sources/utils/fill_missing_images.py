import os
import cv2
import pandas as pd
import albumentations as a

from PIL import Image
from sources.utils.augmentation import Augmentation
from sources.utils.albumentation_transform import AlbumentationTransform
from sources.utils.url_utils import get_image_path_from_url, FILENAME_PATTERN


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

    filled_dataframe = dataframe.copy()
    filled_dataframe["missing"] = 0

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

                    filled_dataframe.loc[index, "is_same"] = 1
                else:
                    filled_dataframe.loc[index, "missing"] = 1
                    continue

            if not os.path.exists(image_path2):
                image2 = Image.open(image_path1)
                image2 = augmentation(image2)

                filename = FILENAME_PATTERN.search(row["image_url2"]).group(1)
                image2.save(os.path.join(save_path, filename))

                filled_dataframe.loc[index, "is_same"] = 1

    return filled_dataframe


def main():
    df = pd.read_csv("../../data/train.csv")
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
    print(df)


if __name__ == "__main__":
    main()
