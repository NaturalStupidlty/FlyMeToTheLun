import re
import os
import pandas as pd


FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")
URL_PATTERN = "https://storage.googleapis.com/lun-ua/images/{}".format


def get_image_path_from_url(url: str):
    filename = FILENAME_PATTERN.search(url).group(1)
    image_path = os.path.join("..", "..", "data", "images", filename)

    return image_path


def create_filenames_csv(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["image_url1"] = dataframe["image_url1"].apply(lambda x: FILENAME_PATTERN.search(x).group(1))
    dataframe["image_url2"] = dataframe["image_url2"].apply(lambda x: FILENAME_PATTERN.search(x).group(1))

    return dataframe
