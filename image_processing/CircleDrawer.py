from typing import Dict

import PIL.Image
import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw

class CircleDrawer:
    def __init__(self, color=(0, 255, 0)):
        self.color = color
        self.subimage_margin = 2

    def draw(self, image: np.array, circles):
        for i in circles[0, :]:
            cv.circle(image, (i[0], i[1]), i[2], self.color, 2)
            cv.circle(image, (i[0], i[1]), 2, self.color, 3)
        return image

    def draw_classification_result(self, image: PIL.Image, bbox, result_label):
        # cv.rectangle(image, bbox[0:1], bbox[2:3], self.color)
        # cv.putText(image, result_label, bbox[0:1], cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.color)
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=self.color)
        draw.text((bbox[0], bbox[1]), result_label, font=ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 20), fill=self.color)
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

    def get_subimages(self, image: PIL.Image, bboxes) -> Dict:
        subimages = {}
        for bbox in bboxes:
            subimages[bbox] = image.crop(bbox)

        return subimages
