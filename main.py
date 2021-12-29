# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

from image_processing.ImagePreprocessor import ImagePreprocessor
from detectors.CircleDetector import CircleDetector
from detectors.PolishCoinClassifier import PolishCoinClassifier
import cv2 as cv
import numpy as np


# Press the green button in the gutter to run the script.

def circle_detection_test():
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


def classifier_test():
    dirs = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'output': 'dataset/output/vgg19',
        'train_network_output': 'dataset/output',
        'test_network_output': 'dataset/output',
    }
    classes = ['5zl', '2zl', 'non-polish-coin']
    classifier = PolishCoinClassifier(classes=classes, dirs=dirs,
                                      # model_weights_path='dataset/logs/weights-epoch_01-val_loss_0.00.hdf5',
                                      )
    classifier.train()
    img = cv.imread('dataset/single-coins/test/5zl/033__5 Zlotych_poland.jpg',
                    cv2.IMREAD_COLOR)
    label, vect = classifier.classify(img)
    print(label)
    print(vect)

if __name__ == '__main__':
    classifier_test()
# circle_detection_test()
