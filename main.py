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
    img = preprocessor.preprocess_opencv(img)
    circles = detector.detect(img)

    bboxes = circle_drawer.get_circles_bboxex(circles)
    print(bboxes)

    img = circle_drawer.draw(img, circles)
    cv.imshow('detected circles', img)
    cv.waitKey(0)

    img = Image.open('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg')
    img = img.resize(preprocessor.dim)
    subimages = circle_drawer.get_subimages(img, bboxes)
    for subimage in subimages.values():
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


def sample_full_detection():
    img_dim = (600, 400)
    detector = CircleDetector()
    preprocessor = ImagePreprocessor(dim=img_dim)
    circle_drawer = CircleDrawer()

    detected_img = cv.imread('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg',
                             cv.IMREAD_COLOR)
    # detect circles
    detected_img = preprocessor.preprocess_opencv(detected_img)
    circles = detector.detect(detected_img)
    bboxes = circle_drawer.get_circles_bboxex(circles)

    # load coin classifier
    dirs = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test'
    }
    classifier_weights_file = 'dataset/logs/weights-epoch_01-val_loss_0.00.hdf5'
    classifier = PolishCoinClassifier(num_classes=3, dirs=dirs,
                                      model_weights_path='dataset/logs/weights-epoch_01-val_loss_0.00.hdf5')

    #get subimages with circles
    detected_img_color = Image.open('dataset/multiple-coins/drive-download-20211227T133506Z-001/20211227_141629.jpg')
    detected_img_color = detected_img_color.resize(preprocessor.dim)
    subimages = circle_drawer.get_subimages(detected_img_color, bboxes)

    #classify
    for bbox, subimage in subimages.items():
        label, result_vect = classifier.classify(subimage)
        detected_img_color = circle_drawer.draw_classification_result(detected_img_color, bbox, label)

    #show detection results
    detected_img_color.show()




if __name__ == '__main__':
    # classifier_test()
    # circle_detection_test()
    sample_full_detection()
