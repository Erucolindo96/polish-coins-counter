import os
from glob import glob
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2


class Histogram:
    def __init__(self, dirs, lower_threshold=0.05, upper_threshold=0.95):
        self.dirs = dirs
        self.lower = 0
        self.upper = 255

        self.cumulative_lower_threshold = lower_threshold
        self.cumulative_upper_threshold = upper_threshold
        self.histogram = {}
        self.pix_cnt = 0
        self.__process_pixel_vectorized = np.vectorize(self.__process_pixel)

    def count_histogram(self):
        for dyr in self.dirs:
            labels_dirs = glob(os.path.join(dyr, '*'))

            for label_dir in labels_dirs:
                image_files = [os.path.join(label_dir, f) for f in listdir(label_dir) if isfile(join(label_dir, f))]
                for file in image_files:
                    image = cv2.imread(file)
                    hsvim = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    self.process_image(hsvim)
        self.count_lower_upper()

    def save_results(self):
        with open('stretching-results', 'w') as f:
            f.write('lower: {}'.format(self.lower))
            f.write('upper: {}'.format(self.upper))

    def process_image(self, image, type='hsv'):
        if type == 'hsv':
            channel = image[:, :, 2]
        else:
            channel = image
        self.__process_pixel_vectorized(channel)

    def __process_pixel(self, value: int):
        if value not in self.histogram:
            self.histogram[value] = 0
        self.histogram[value] += 1
        self.pix_cnt += 1
        return value

    def count_lower_upper(self):
        cumulative = 0.0
        for pix_val, count in sorted(self.histogram.items(), key=lambda item: item[0]):
            delta = float(count) / self.pix_cnt
            cumulative += delta
            if cumulative <= self.cumulative_lower_threshold:
                self.lower = pix_val
            if cumulative <= self.cumulative_upper_threshold:
                self.upper = pix_val

