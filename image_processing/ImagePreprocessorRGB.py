import PIL.Image
import cv2 as cv
import numpy as np

from image_processing.Histogram import Histogram
from image_processing.PixelScaler import PixelScaler


class ImagePreprocessorRGB:
    def __init__(self, resize=True, dim=(600, 400), stretching=(41, 255)):
        self.resize = resize
        self.dim = dim
        self.lower = stretching[0] if stretching else None
        self.upper = stretching[1] if stretching else None

    def preprocess(self, image: np.array) -> PIL.Image.Image:
        processed = image
        if self.resize:
            processed = cv.resize(processed, self.dim)

        # casting to hsv
        processed = cv.cvtColor(processed, cv.COLOR_RGB2HSV)
        #stretch histogram
        scaler = PixelScaler(dirs=None, lower=self.lower, upper=self.upper)
        processed = scaler.scale(processed, type='hsv')

        #return to rgb
        processed = cv.cvtColor(processed, cv.COLOR_HSV2RGB)

        return PIL.Image.fromarray(processed)
