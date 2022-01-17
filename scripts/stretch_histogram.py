import os
from glob import glob
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
from conf.Config import Config


class Scaler:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.scale_pixel_vect = np.vectorize(self.scale_pixel)

    def scale_pixel(self, pix: int):
        if pix < self.a:
            return 0
        if pix > self.b:
            return 255
        return int(255 * (pix - self.a) / (self.b - self.a))

    def scale(self, image: Image) -> Image:
        image_as_array = np.array(image)
        scaled = self.scale_pixel_vect(image_as_array)
        return Image.fromarray(np.uint8(scaled))


dirs = ['../dataset/single-coins/train', '../dataset/single-coins/validation', '../dataset/single-coins/test']

scaler = Scaler(50, 200)

for dyr in dirs:
    labels_dirs = glob(os.path.join(dyr, '*'))
    labels_cnt = len(labels_dirs)

    for label_dir in labels_dirs:
        image_files = [os.path.join(label_dir, f) for f in listdir(label_dir) if isfile(join(label_dir, f))]
        for file in image_files:
            image = Image.open(file)
            scaled = scaler.scale(image)
            scaled.save(file)
