# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from PIL import Image

from image_processing.ImagePreprocessor import ImagePreprocessor
from detectors.CircleDetector import CircleDetector
from detectors.PolishCoinClassifier import PolishCoinClassifier
from image_processing.CircleDrawer import CircleDrawer
import cv2 as cv
import numpy as np


def circle_detection_test():
    detector = CircleDetector()
    preprocessor = ImagePreprocessor()
    circle_drawer = CircleDrawer()

    # TODO ogarnij rozdźwięk między typami obrazków (dla detekcji okręgów - openCV, dla klasyfikacji - PIL.Image)
    img = cv.imread('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg',
                    cv2.IMREAD_COLOR)
    img = preprocessor.preprocess(img)
    circles = detector.detect(img)

    bboxes = circle_drawer.get_circles_bboxex(circles)
    print(bboxes)

    img = circle_drawer.draw(img, circles)
    cv.imshow('detected circles', img)
    cv.waitKey(0)

    img = Image.open('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg')
    img = img.resize(preprocessor.dim)
    subimages = circle_drawer.get_subimages(img, bboxes)
    for subimage in subimages:
        subimage.show()

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
    # classes = ['5zl', '2zl', 'non-polish-coin']
    classifier = PolishCoinClassifier(num_classes=3, dirs=dirs,
                                      model_weights_path='dataset/logs/weights-epoch_01-val_loss_0.00.hdf5',
                                      )
    # classifier.train()
    img = cv.imread('dataset/single-coins/test/5zl/033__5 Zlotych_poland.jpg',
                    cv2.IMREAD_COLOR)
    label, vect = classifier.classify(img)
    print(label)
    print(vect)


if __name__ == '__main__':
    # classifier_test()
    circle_detection_test()
