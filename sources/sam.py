import torch

from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class SAM:
    def __init__(self, checkpoint_path: str = "sam_vit_h_4b8939.pth", model_type: str = "vit_h"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Sam model at device: {self.device}")
        self.model = self.model.to(self.device)
        self.predictor = SamPredictor(self.model)

    def __call__(self, image: Image):
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict_torch(
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )

        if self.device != 'cpu':
            logits = logits.cpu()
        logits = logits.detach().numpy()

        return logits


def main():


if __name__ == "__main__":
    main()
