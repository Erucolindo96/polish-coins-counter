import cv2
import cv2 as cv
import numpy as np

from image_processing.Histogram import Histogram
from image_processing.PixelScaler import PixelScaler


class ImagePreprocessor:
    def __init__(self, resize=True, dim=(600, 400)):  # , stretching=(41, 255)):
        self.resize = resize
        self.dim = dim

    def preprocess_opencv(self, image: np.array):
        processed = image

        # casting to gray scale to detect circles
        processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)

        if self.resize:
            processed = cv.resize(processed, self.dim)

        # stretch histogram
        histogram = Histogram(dirs=None)
        histogram.process_image(processed, 'grayscale')
        histogram.count_lower_upper()
        scaler = PixelScaler(dirs=None, lower=histogram.lower, upper=histogram.upper)
        processed = scaler.scale(processed, 'grayscale')

        # processed = cv.Sobel(processed,cv.CV_8U,1,1,ksize=5)

        # def sharpen(image):
        #     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #     return cv2.filter2D(image, -1, kernel)
        # processed = cv.Canny(processed, 30, 300)
        # processed = cv.medianBlur(processed, 3)
        # processed = sharpen(processed)
        return processed
