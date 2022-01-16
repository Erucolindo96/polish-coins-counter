import cv2
from PIL import Image

from image_processing.ImagePreprocessor import ImagePreprocessor
from detectors.CircleDetector import CircleDetector
from detectors.PolishCoinClassifier import PolishCoinClassifier
from image_processing.CircleDrawer import CircleDrawer
import cv2 as cv


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


def classifier_train():
    dirs = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs',
        'output': 'output/vgg19',
    }
    classifier = PolishCoinClassifier(polish_coin_classes=9, dirs=dirs, epochs_training=40, input_shape=(150, 150, 3))
    classifier.train()


def sample_full_detection():
    img_dim = (600, 400)
    classifier_image_shape = (150, 150, 3)
    classifier_weights_file = 'output/vgg19/final-model-weights_3.013_150x150.hdf5'
    detected_image = 'dataset/multiple-coins/265814725_854205531939912_7533564023313811302_n.jpg'

    # load coin classifier
    dirs = {
        'training': 'dataset/single-coins/train',
        'validation': 'dataset/single-coins/validation',
        'test': 'dataset/single-coins/test',
        'logs': 'logs'
    }
    classifier = PolishCoinClassifier(polish_coin_classes=9, dirs=dirs, epochs_training=40, input_shape=classifier_image_shape,
                                      model_weights_path=classifier_weights_file)

    detector = CircleDetector()
    preprocessor = ImagePreprocessor(dim=img_dim)  # only for opencv images
    circle_drawer = CircleDrawer(non_polish_label=classifier.non_polish_class_label)

    detected_img = cv.imread(detected_image,
                             cv.IMREAD_COLOR)
    # detect circles
    detected_img = preprocessor.preprocess_opencv(detected_img)
    circles = detector.detect(detected_img)
    bboxes = circle_drawer.get_circles_bboxex(circles)

    # get subimages with circles
    detected_img_color = Image.open(detected_image)
    detected_img_color = detected_img_color.resize(preprocessor.dim)
    subimages = circle_drawer.get_subimages(detected_img_color, bboxes)

    # classify
    for bbox, subimage in subimages.items():
        label, result_vect = classifier.classify(subimage)
        detected_img_color = circle_drawer.draw_classification_result(detected_img_color, bbox, label, draw_non_polish=False)

    # show detection results
    detected_img_color.show()


if __name__ == '__main__':
    # classifier_train()
    # circle_detection_test()
    sample_full_detection()
