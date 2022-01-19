import cv2
from PIL import Image

from image_processing.ImagePreprocessorGrayscale import ImagePreprocessorGrayscale
from detectors.CircleDetector import CircleDetector
from detectors.PolishCoinClassifier import PolishCoinClassifier
from image_processing.CircleDrawer import CircleDrawer
import cv2 as cv
from conf.Config import Config
from image_processing.ImagePreprocessorRGB import ImagePreprocessorRGB


def circle_detection_test():
    detector = CircleDetector(min_dist=Config.circle_detection['min_dist'],
                              min_radius=Config.circle_detection['min_radius'],
                              max_radius=Config.circle_detection['max_radius'])
    preprocessor = ImagePreprocessorGrayscale(resize=True, dim=Config.main['img_dim'])
    circle_drawer = CircleDrawer(color=Config.circle_drawer['bbox_color'],
                                 subimage_margin=Config.circle_drawer['subimage_margin'])

    # TODO ogarnij rozdźwięk między typami obrazków (dla detekcji okręgów - openCV, dla klasyfikacji - PIL.Image)
    img = cv.imread(Config.main['detected_image'],
                    cv2.IMREAD_COLOR)
    img = preprocessor.preprocess_opencv(img)
    circles = detector.detect(img)

    bboxes = circle_drawer.get_circles_bboxex(circles)
    print(bboxes)

    img = circle_drawer.draw(img, circles)
    cv.imshow('detected circles', img)
    cv.waitKey(0)

    img = Image.open(Config.main['detected_image'])
    img = img.resize(preprocessor.dim)
    subimages = circle_drawer.get_subimages(img, bboxes)
    for subimage in subimages.values():
        subimage.show()

    cv.destroyAllWindows()


def classifier_train():
    classifier = PolishCoinClassifier(polish_coin_classes=Config.polish_coin_classifier['polish_coin_classes_num'],
                                      dirs=Config.polish_coin_classifier_dataset,
                                      epochs_training=Config.polish_coin_classifier['epochs_training'],
                                      input_shape=Config.main['classifier_image_shape'],
                                      image_datatype=Config.polish_coin_classifier['image_datatype'],
                                      non_polish_coin_threshold=Config.polish_coin_classifier[
                                          'non_polish_coin_threshold'],
                                      best_scores_multiplier=Config.polish_coin_classifier['best_scores_multiplier'],
                                      full_layer_size=Config.polish_coin_classifier['full_layer_size'],
                                      full_con_layers=Config.polish_coin_classifier['full_con_layers'])
    classifier.train()


def classifier_test():
    classifier = PolishCoinClassifier(polish_coin_classes=Config.polish_coin_classifier['polish_coin_classes_num'],
                                      dirs=Config.polish_coin_classifier_dataset,
                                      model_weights_path=Config.polish_coin_classifier['classifier_model_path'],
                                      epochs_training=Config.polish_coin_classifier['epochs_training'],
                                      input_shape=Config.main['classifier_image_shape'],
                                      image_datatype=Config.polish_coin_classifier['image_datatype'],
                                      non_polish_coin_threshold=Config.polish_coin_classifier[
                                          'non_polish_coin_threshold'],
                                      best_scores_multiplier=Config.polish_coin_classifier['best_scores_multiplier'],
                                      full_layer_size=Config.polish_coin_classifier['full_layer_size'],
                                      full_con_layers=Config.polish_coin_classifier['full_con_layers'])
    classifier.test()


def sample_full_detection():
    detected_image = Config.main['detected_image']

    classifier = PolishCoinClassifier(polish_coin_classes=Config.polish_coin_classifier['polish_coin_classes_num'],
                                      dirs=Config.polish_coin_classifier_dataset,
                                      model_weights_path=Config.polish_coin_classifier['classifier_model_path'],
                                      input_shape=Config.main['classifier_image_shape'],
                                      image_datatype=Config.polish_coin_classifier['image_datatype'],
                                      non_polish_coin_threshold=Config.polish_coin_classifier[
                                          'non_polish_coin_threshold'],
                                      best_scores_multiplier=Config.polish_coin_classifier['best_scores_multiplier'],
                                      full_layer_size=Config.polish_coin_classifier['full_layer_size'],
                                      full_con_layers=Config.polish_coin_classifier['full_con_layers'])

    detector = CircleDetector(min_dist=Config.circle_detection['min_dist'],
                              min_radius=Config.circle_detection['min_radius'],
                              max_radius=Config.circle_detection['max_radius'])
    preprocessor_grayscale = ImagePreprocessorGrayscale(dim=Config.main['img_dim'], resize=True)
    preprocessor_rgb = ImagePreprocessorRGB(dim=Config.main['img_dim'], resize=True,
                                            stretching=Config.image_scaler['stretching'])

    circle_drawer = CircleDrawer(color=Config.circle_drawer['bbox_color'],
                                 subimage_margin=Config.circle_drawer['subimage_margin'])

    detected_img = cv.imread(detected_image, cv.IMREAD_COLOR)
    # detect circles
    detected_img = preprocessor_grayscale.preprocess_opencv(detected_img)
    circles = detector.detect(detected_img)
    bboxes = circle_drawer.get_circles_bboxex(circles)

    detected_img_color_orig = Image.open(detected_image)
    detected_img_color_orig = detected_img_color_orig.resize(Config.main['img_dim'])

    # get subimages with circles
    detected_img_color_np = cv.imread(detected_image, cv.IMREAD_COLOR)
    detected_img_color_preprocessed_pil = preprocessor_rgb.preprocess(detected_img_color_np)
    subimages = circle_drawer.get_subimages(detected_img_color_preprocessed_pil, bboxes)

    # classify
    for bbox, subimage in subimages.items():
        label, result_vect = classifier.classify(subimage)
        label_non_polish = label == classifier.non_polish_class_label
        if label_non_polish:
            if Config.circle_drawer['draw_non_polish']:
                detected_img_color_orig = circle_drawer.draw_classification_result(detected_img_color_orig, bbox, label)
        else:
            detected_img_color_orig = circle_drawer.draw_classification_result(detected_img_color_orig, bbox, label)

    # show detection results
    detected_img_color_orig.show()


task_to_handler = {
    'full_detection': sample_full_detection,
    'circle_detection': circle_detection_test,
    'classifier_training': classifier_train,
    'classifier_test': classifier_test
}

if __name__ == '__main__':
    task = Config.main['task']
    handler = task_to_handler[task]
    handler()
