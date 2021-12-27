import cv2 as cv


class CircleDetector:
    def __init__(self, min_dist=50, min_radius=10, max_radius=100):
        self.min_dist = min_dist
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image):
        circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=self.min_dist, param1=200, param2=30,
                                  minRadius=self.min_radius, maxRadius=self.max_radius)
        return circles
