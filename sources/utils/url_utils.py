import re
import os
import pandas as pd


FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")
URL_PATTERN = "https://storage.googleapis.com/lun-ua/images/{}".format


def get_image_path_from_url(url: str):
    filename = FILENAME_PATTERN.search(url).group(1)
    train_path = os.path.join("..", "..", "data", "train", filename)
    test_path = os.path.join("..", "..", "data", "test", filename)

    return train_path if os.path.exists(train_path) else test_path


def create_filenames_csv(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["image_url1"] = dataframe["image_url1"].apply(lambda x: FILENAME_PATTERN.search(x).group(1))
    dataframe["image_url2"] = dataframe["image_url2"].apply(lambda x: FILENAME_PATTERN.search(x).group(1))

    return dataframe
