import re
import os


FILENAME_PATTERN = re.compile(r"(\d{9}.\w+)")
URL_PATTERN = "https://storage.googleapis.com/lun-ua/images/{}".format


def get_image_path_from_url(url: str):
    filename = FILENAME_PATTERN.search(url).group(1)
    image_path = os.path.join("..", "..", "data", "images", filename)

    return image_path
