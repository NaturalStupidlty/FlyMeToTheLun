from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


class Classifier:
    def __init__(self, model_name: str = 'google/vit-large-patch32-384') -> None:
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def __call__(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        scores = outputs.logits

        return scores
