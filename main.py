# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

from image_processing.ImagePreprocessor import ImagePreprocessor
from detectors.CircleDetector import CircleDetector
import cv2 as cv
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detector = CircleDetector()
    preprocessor = ImagePreprocessor()

    img = cv.imread('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg',
                    cv2.IMREAD_COLOR)
    img = preprocessor.preprocess(img)
    circles = detector.detect(img)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow('detected circles', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
