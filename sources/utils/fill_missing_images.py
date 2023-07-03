import os
import pandas as pd
import albumentations as a

from PIL import Image
from pathlib import Path
from sources.utils.augmentation import Augmentation
from sources.utils.albumentation_transform import Albumentation
from sources.utils.url_utils import create_filenames_csv


def fill_missed_photos(dataframe: pd.DataFrame,
                       augmentation: Augmentation,
                       images_folder: str,
                       save_folder: str = None) -> pd.DataFrame:
    if save_folder is None:
        save_folder = images_folder

    dataframe = dataframe.copy()
    images_folder = Path(images_folder)
    save_folder = Path(save_folder)
    folder_files = set(os.listdir(images_folder))
    image_names = set(dataframe["image_url1"].tolist() + dataframe["image_url2"].tolist())
    missed = list(image_names - folder_files)

    dataframe["missed_image1"] = dataframe.apply(lambda r: 1 if r["image_url1"] in missed else 0, axis=1)
    dataframe["missed_image2"] = dataframe.apply(lambda r: 1 if r["image_url2"] in missed else 0, axis=1)

    for index, row in dataframe.iterrows():
        missed1 = dataframe.loc[index, "missed_image1"]
        missed2 = dataframe.loc[index, "missed_image2"]
        image_url1 = row["image_url1"]
        image_url2 = row["image_url2"]

        if missed1 and not missed2:
            new_path = save_folder / image_url1
            old_path = images_folder / image_url2
        elif not missed1 and missed2:
            new_path = save_folder / image_url2
            old_path = images_folder / image_url1
        else:
            continue

        image = Image.open(old_path)
        augmentation(image).save(new_path)
        dataframe.loc[index, "is_same"] = 1

    return dataframe


def main():
    composition = a.Compose([
        a.Rotate(limit=10, p=0.9),
        a.Blur(p=0.8, blur_limit=10),
        a.GaussNoise(p=0.9, var_limit=(10.0, 100.0), mean=0, per_channel=True),
        a.ISONoise(p=0.9, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        a.ImageCompression(p=0.9, quality_lower=99, quality_upper=100),
    ])

    albumentations = Albumentation(composition)
    test = pd.read_csv("../../data/test.csv")
    test = create_filenames_csv(test)
    test = fill_missed_photos(test, albumentations, "../../data/images", "../../data/huina")


if __name__ == "__main__":
    main()
