import numpy as np
import cv2 as cv
import PIL.Image

class CircleDrawer:
    def __init__(self, color=(0, 255, 0)):
        self.color = color
        self.subimage_margin = 2

    def draw(self, image: np.array, circles):
        for i in circles[0, :]:
            cv.circle(image, (i[0], i[1]), i[2], self.color, 2)
            cv.circle(image, (i[0], i[1]), 2, self.color, 3)
        return image

    def get_circles_bboxex(self, circles):
        bboxes = []
        for c in circles[0, :]:
            x, y, r = c
            top_left_x = x - r - self.subimage_margin if x - r - self.subimage_margin >= 0 else 0
            top_left_y = y - r - self.subimage_margin if y - r - self.subimage_margin >= 0 else 0
            bottom_right_x = x + r + self.subimage_margin
            bottom_right_y = y + r + self.subimage_margin
            bboxes.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        return bboxes

    def get_subimages(self, image: PIL.Image, bboxes):
        subimages = []
        for bbox in bboxes:
            subimages.append(image.crop(bbox))

        return subimages
