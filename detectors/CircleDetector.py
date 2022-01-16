import cv2 as cv
import numpy as np


class CircleDetector:
    def __init__(self, min_dist=50, min_radius=10, max_radius=60):
        self.min_dist = min_dist
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image):
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=self.min_dist, param1=150, param2=30,
                                  minRadius=self.min_radius, maxRadius=self.max_radius)

        circles = np.uint16(np.around(circles))
        return circles
