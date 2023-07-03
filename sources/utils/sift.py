import cv2
import numpy as np

from PIL import Image


class SIFTPoint:
    def __init__(self, keypoint, descriptor):
        self.keypoint = keypoint
        self.descriptor = descriptor


class SIFT:
    def __init__(self, image_resolution: int = 1920):
        self.image_resolution = image_resolution
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def __call__(self, image: Image):
        image = np.asarray(image)
        image = self._resize_image(image)
        keypoint, descriptor = self.sift.detectAndCompute(image, None)
        sift_point = SIFTPoint(keypoint, descriptor)

        return sift_point

    def _resize_image(self, image: np.ndarray):
        height, width, channel = image.shape
        aspect_ratio = width / height

        if aspect_ratio < 1:
            new_size = (int(self.image_resolution * aspect_ratio), self.image_resolution)
        else:
            new_size = (self.image_resolution, int(self.image_resolution / aspect_ratio))

        resized_image = cv2.resize(image, new_size)

        return resized_image

    def get_score(self, point1, point2, approximate: bool = False):
        if approximate:
            matches = self._approximateMatches(point1.descriptor, point2.descriptor)
        else:
            matches = self._calculateMatches(point1.descriptor, point2.descriptor)

        score = 100 * (len(matches) / min(len(point1.keypoint), len(point2.keypoint)))

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

    def _approximateMatches(self, descriptor1, descriptor2):
        matches = self.matcher.knnMatch(descriptor1, descriptor2, k=2)
        topResults = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                topResults.append([m])

        return topResults
