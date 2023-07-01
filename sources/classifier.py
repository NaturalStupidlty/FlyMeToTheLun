import os.path
import re
import pandas as pd

from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


FILENAME_PATTERN = re.compile(r"\/([\d]{9}.[\w]+)")


class Classifier:
    def __init__(self, model_name: str = 'google/vit-large-patch32-384') -> None:
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def __call__(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        scores = outputs.logits

        return scores


def apply2dataframe(dataframe: pd.DataFrame):
    classifier = Classifier()

    class_indexes = []

    for index, row in dataframe.iterrows():
        scores = []
        for url in ["image_url1", "image_url2"]:
            image_url = row[url]
            image_path = FILENAME_PATTERN.search(image_url).group(1)
            image_path = os.path.join("..", "data", "images", image_path)
            image = Image.open(image_path)
            score = classifier(image).argmax(-1).item()
            scores.append(score)

        class_indexes.append(scores)

    processed_dataframe = dataframe.copy()
    processed_dataframe.loc[:, "class_idx1"] = [i[0] for i in class_indexes]
    processed_dataframe.loc[:, "class_idx2"] = [i[1] for i in class_indexes]

    return processed_dataframe


def main():
    df = pd.read_csv("../data/train.csv")

    df = apply2dataframe(df.head(10))
    print(df[["class_idx1", "class_idx2", "is_same"]])
    # print("Predicted class:", self.model.config.id2label[predicted_class_idx])


if __name__ == "__main__":
    main()
