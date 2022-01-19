import os
from glob import glob
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from PIL import Image


class PixelScaler:
    def __init__(self, dirs, lower, upper):
        self.lower = lower
        self.upper = upper
        self.dirs = dirs
        self.scale_pixel_vect = np.vectorize(self.scale_pixel)

    def scale_pixel(self, pix: int):
        if pix < self.lower:
            return 0
        if pix > self.upper:
            return 255
        return int(255 * (pix - self.lower) / (self.upper - self.lower))

    def scale(self, image, type='hsv'):
        if type == 'hsv':
            channel = image[:, :, 2]
        else:
            channel = image
        channel_scaled = self.scale_pixel_vect(channel)
        channel_scaled = channel_scaled.astype(np.uint8)

        if type == 'hsv':
            image[:, :, 2] = channel_scaled
            return image

        return channel_scaled

    def scale_images_in_dirs(self):
        for dyr in self.dirs:
            labels_dirs = glob(os.path.join(dyr, '*'))

            for label_dir in labels_dirs:
                image_files = [os.path.join(label_dir, f) for f in listdir(label_dir) if isfile(join(label_dir, f))]
                for file in image_files:
                    image = cv2.imread(file)
                    hsvim = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    scaled = self.scale(hsvim)
                    scaled_rgb = cv2.cvtColor(scaled, cv2.COLOR_HSV2RGB)
                    Image.fromarray(scaled_rgb).save(file)
                    print('Image {} has been scaled'.format(file))
