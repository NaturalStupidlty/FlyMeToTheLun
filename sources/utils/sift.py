import cv2
import numpy as np

from PIL import Image


def image_resize_test(image, max_dimension: int = 1024):
    height, width, channel = image.shape
    aspect_ratio = width / height

    if aspect_ratio < 1:
        new_size = (int(max_dimension * aspect_ratio), max_dimension)
    else:
        new_size = (max_dimension, int(max_dimension / aspect_ratio))

    resized_image = cv2.resize(image, new_size)
    return resized_image


class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def __call__(self, image: Image):
        image = np.asarray(image)
        keypoint, descriptor = self.sift.detectAndCompute(image, None)

        return keypoint, descriptor

    def get_score(self, k1, k2):
        matches = self._calculateMatches(k1[1], k2[1])
        score = 100 * (len(matches) / min(len(k1[0]), len(k2[0])))

        return score

    def _calculateMatches(self, descriptor1, descriptor2):
        matches = self.matcher.knnMatch(descriptor1, descriptor2, k=2)
        results1 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                results1.append([m])

        matches = self.matcher.knnMatch(descriptor2, descriptor1, k=2)
        results2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                results2.append([m])

        top_results = []
        for match1 in results1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx

            for match2 in results2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx

                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    top_results.append(match1)

        return top_results
