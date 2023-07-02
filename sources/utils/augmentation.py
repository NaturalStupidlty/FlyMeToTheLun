import abc

from PIL import Image


class Augmentation:
    """
    Interface for augmentations
    """
    @abc.abstractmethod
    def __call__(self, image: Image) -> Image:
        pass
