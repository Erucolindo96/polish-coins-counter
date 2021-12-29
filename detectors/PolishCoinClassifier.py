import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from neural_trainer.NeuralTrainer import NeuralTrainer


class PolishCoinClassifier:
    def __init__(self, num_classes, dirs, model_weights_path=None, input_shape=(80, 80, 3),
                 image_datatype='float32',
                 epochs_training=10):
        self.model_weights_path = model_weights_path
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dirs = dirs
        self.image_datatype = image_datatype
        self.epochs_training = epochs_training

        self.model = None

        self.gpu_memory_limit_mb = 2048
        self.tf_verbosity = False
        self.kernel_initializer = 'glorot_normal'
        self.bias_initializer = 'glorot_normal'
        self.log_dir = 'dataset/logs'
        self.learning_rate = 1e-4
        self.metrics = ['accuracy', 'mse']

        self.__init_tf()

        self.__create_model()
        if self.model_weights_path:
            self.__load_model()

        self.__compile_model()
        self.__show_model()

        self.trainer = NeuralTrainer(model=self.model, dirs=self.dirs, log_dir=self.log_dir,
                                     epochs=self.epochs_training, test_model=True)
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

    def __compile_model(self):
        optimizer = tf.keras.optimizers.get({
            "class_name": 'adam',
            "config": {"learning_rate": self.learning_rate}}
        )
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=self.metrics)

    def __show_model(self):
        """
        Prints model statistics
        """

        print('\n\n')
        self.model.summary()
        print('\n\n')

    def __load_model(self):
        self.model.load_weights(self.model_weights_path)

    def __create_model(self):
        kernel_initializer = tf.keras.initializers.get(self.kernel_initializer)
        bias_initializer = tf.keras.initializers.get(self.bias_initializer)

        model_input = tf.keras.layers.Input(self.input_shape, dtype=self.image_datatype)
        preprocessing = tf.keras.applications.vgg19.preprocess_input(model_input)

        vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False,
                                                  input_shape=self.input_shape)
        vgg19.trainable = False

        model = vgg19(preprocessing)
        model = Flatten()(model)
        model = Dense(
            4096,
            activation='relu',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(model)
        model = Dense(
            4096,
            activation='relu',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(model)
        model = Dense(
            self.num_classes,
            activation='softmax',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)(model)

        self.model = Model(inputs=[model_input], outputs=[model])

    def __find_label(self, result):
        max_likelihood_idx = tf.math.argmax(result, 0)
        for label, label_vect in self.trainer.labels_to_labels_vect.items():
            if tf.math.equal(tf.math.argmax(label_vect, 0), max_likelihood_idx):
                return label

        return None

    def train(self):
        self.trainer.run()

    def classify(self, image):
        img_array = keras.preprocessing.image.img_to_array(np.array(image))
        img_array = tf.image.resize(img_array, [self.input_shape[0], self.input_shape[1]])
        img_array = tf.expand_dims(img_array, 0)

        result = self.model.predict(img_array).flatten()

        result_label = self.__find_label(result)
        return result_label, result
