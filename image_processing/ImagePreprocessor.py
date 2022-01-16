import cv2 as cv
import numpy as np


class ImagePreprocessor:
    def __init__(self, resize=True, dim=(600, 400)):
        self.resize = resize
        self.dim = dim

    def preprocess_opencv(self, image: np.array):
        processed = image
        # casting to gray scale to detect circles
        processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
        if self.resize:
            processed = cv.resize(processed, self.dim)

        return processed
