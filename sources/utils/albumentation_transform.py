import numpy as np

from PIL import Image
from sources.utils.augmentation import Augmentation


class AlbumentationTransform(Augmentation):
    """
    Adapter (wrapper) for albumentations augmentations.
    """
    def __init__(self, transform: callable):
        """
        Can be a separate augmentation or Albumentations.Compose
        :param transform:
        """
        self.transform = transform

    def __call__(self, image: Image) -> Image:
        transformed = self.transform(image=np.asarray(image))
        transformed_image = Image.fromarray(transformed['image'])

        return transformed_image
