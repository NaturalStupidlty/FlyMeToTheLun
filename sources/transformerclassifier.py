import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, AutoImageProcessor, \
    ConvNextV2ForImageClassification


class TransformerClassifier:
    def __init__(self, model_name: str = 'google/vit-large-patch32-384') -> None:
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{model_name} model at device: {self.device}")
        self.model = self.model.to(self.device)

    def __call__(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        if self.device != 'cpu':
            logits = logits.cpu()
        logits = logits.detach().numpy()

        return logits


class ConvNeXT:
    def __init__(self, model_name: str = "facebook/convnextv2-huge-22k-512") -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ConvNextV2ForImageClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{model_name} model at device: {self.device}")
        self.model = self.model.to(self.device)

    def __call__(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        if self.device != 'cpu':
            logits = logits.cpu()
        logits = logits.detach().numpy()

        return logits
