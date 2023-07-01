import os
import re
import pandas as pd

from PIL import Image
from sources.classifier import Classifier


FILENAME_PATTERN = re.compile(r"\/([\d]{9}.[\w]+)")


def classify_dataframe(dataframe: pd.DataFrame):
    classifier = Classifier()
    class_indexes = []

    for index, row in dataframe.iterrows():
        indexes = []
        for url in ["image_url1", "image_url2"]:
            image_url = row[url]
            image_path = FILENAME_PATTERN.search(image_url).group(1)
            image_path = os.path.join("..", "data", "images", image_path)
            image = Image.open(image_path)
            score = classifier(image).argmax(-1).item()
            indexes.append(score)

        class_indexes.append(indexes)

    processed_dataframe = dataframe.copy()
    processed_dataframe.loc[:, "class_index1"] = [i[0] for i in class_indexes]
    processed_dataframe.loc[:, "class_index2"] = [i[1] for i in class_indexes]

    return processed_dataframe


def test():
    df = pd.read_csv("../data/train.csv")
    df = classify_dataframe(df.head(10))
    print(df[["class_idx1", "class_idx2", "is_same"]])


if __name__ == "__main__":
    test()
