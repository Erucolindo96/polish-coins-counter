import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from neural_trainer.NeuralTrainer import NeuralTrainer


class PolishCoinClassifier:
    def __init__(self, classes, dirs, model=None, input_shape=(80, 80, 3), image_datatype='float32',
                 epochs_training=10):
        self.model = model
        self.num_classes = len(classes)
        self.classes = classes
        self.input_shape = input_shape
        self.dirs = dirs
        self.image_datatype = image_datatype
        self.epochs_training = epochs_training

        self.gpu_memory_limit_mb = 2048
        self.tf_verbosity = False
        self.kernel_initializer = 'glorot_normal'
        self.bias_initializer = 'glorot_normal'
        self.log_dir = 'dataset/logs'
        self.__init_tf()

        if not self.model:
            self.__create()

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

    def __create(self):
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

    def train(self):
        self.trainer.run()

    def classify(self, image):
        pass
