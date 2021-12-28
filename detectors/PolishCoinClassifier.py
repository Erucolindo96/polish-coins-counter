import tensorflow as tf
from glob import glob
import os

# Keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from neural_trainer.NeuralTrainer import NeuralTrainer


class PolishCoinClassifier:
    def __init__(self, classes, dirs, model=None, num_classes=3, input_shape=(80, 80, 3)):
        self.model = model
        self.num_classes = num_classes
        self.classes = classes
        self.input_shape = input_shape
        self.dirs = dirs

        self.gpu_memory_limit_mb = 3072
        self.tf_verbosity = False
        self.kernel_initializer = 'glorot_normal'
        self.bias_initializer = 'glorot_normal'
        self.log_dir = 'dataset/logs'
        self.__init_tf()

        if not self.model:
            self.__create()

        self.trainer = NeuralTrainer(model=self.model, dirs=self.dirs, log_dir=self.log_dir, epochs=40, test_model=True)
        self.trainer.initialize()

    def __init_tf(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=self.gpu_memory_limit_mb
                    )
                ])
            except RuntimeError as e:
                print(e)

        tf.debugging.set_log_device_placement(self.tf_verbosity)

    def __create(self):
        kernel_initializer = tf.keras.initializers.get(self.kernel_initializer)
        bias_initializer = tf.keras.initializers.get(self.bias_initializer)

        vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False,
                                                  input_shape=self.input_shape)
        vgg19 = Flatten()(vgg19)
        vgg19 = Dense(
            4096,
            activation='relu',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(vgg19)
        vgg19 = Dense(
            4096,
            activation='relu',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(vgg19)
        vgg19 = Dense(
            self.num_classes,
            activation='softmax',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(vgg19)

        self.model = vgg19

    def train(self):
        self.trainer.run()

    def classify(self, image):
        pass
