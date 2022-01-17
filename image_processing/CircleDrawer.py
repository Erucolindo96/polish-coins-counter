from typing import Dict

import PIL.Image
import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw


class CircleDrawer:
    def __init__(self, color=(0, 255, 0), subimage_margin=2):
        self.color = color
        self.subimage_margin = subimage_margin

    def draw(self, image: np.array, circles):
        for i in circles[0, :]:
            cv.circle(image, (i[0], i[1]), i[2], self.color, 2)
            cv.circle(image, (i[0], i[1]), 2, self.color, 3)
        return image

    def draw_classification_result(self, image: PIL.Image, bbox, result_label):
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=self.color)
        draw.text((bbox[0], bbox[1]), result_label,
                  font=ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 20), fill=self.color)
        return image

    def get_circles_bboxex(self, circles):
        bboxes = []
        for c in circles[0, :]:
            x, y, r = c
            x, y, r = (np.int64(x), np.int64(y), np.int64(r))
            top_left_x = x - r - self.subimage_margin if x - r - self.subimage_margin >= 0 else 0
            top_left_y = y - r - self.subimage_margin if y - r - self.subimage_margin >= 0 else 0
            bottom_right_x = x + r + self.subimage_margin
            bottom_right_y = y + r + self.subimage_margin
            bboxes.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
        return bboxes

    def get_subimages(self, image: PIL.Image, bboxes) -> Dict:
        subimages = {}
        for bbox in bboxes:
            subimage = image.crop(bbox)
            # subimages[bbox] = subimage
            subimages[bbox] = self.__cut_circle(subimage)
            # FIXME co wtedy gdy okrąg na krawedzi obrazu -> wycinek nie jest kwadratowy?
            # wtedy wycinamy koło w złym miejscu
        return subimages

    def __cut_circle(self, subimage: PIL.Image) -> PIL.Image:
        h, w = subimage.size
        lum_img = PIL.Image.new('L', (h, w), 0)

        drawer = ImageDraw.Draw(lum_img)
        drawer.pieslice(((0, 0), (h, w)), 0, 360, fill="white", outline="white")

        img_arr = np.array(subimage)
        lum_img_arr = np.array(lum_img)
        final_img_arr = np.dstack((img_arr, lum_img_arr))
        rgba_image = PIL.Image.fromarray(final_img_arr)

        background = PIL.Image.new('RGB', rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image.split()[3])
        return background
