# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-09 18:16:36
# @ Modified time: 2020-12-20 16:55:11
# @ Description:
#
#     Implementation of the complete image-recognision neural network's training flow.
#
# ================================================================================================================

# Tensorflow
import tensorflow as tf
# Dedicated datapipe
from neural_trainer.tools.ImageAugmentation import ImageAugmentation
from neural_trainer.tools.DataPipe import DataPipe
from neural_trainer.tools.ConfusionMatrixCallback import ConfusionMatrixCallback
from neural_trainer.tools.LRTensorBoard import LRTensorBoard
# Images manipulation
# from PIL import Image
# Utilities
from glob import glob
import numpy as np
import pickle
import os
import imghdr


class NeuralTrainer:
    def __init__(self, model, dirs, log_dir, epochs=40, test_model=True, input_shape=(80, 80, 3),
                 image_datatype='float32'):
        self.model = model
        self.callbacks = []
        self.dirs = dirs
        self.log_dir = log_dir
        self.epochs = epochs
        self.test_model = test_model
        self.input_shape = input_shape
        self.image_datatype = image_datatype

        self.pipe = None
        self.__history = None

        self.batch_size = 64
        self.learning_rate = 1e-4
        self.metrics = ['accuracy', 'mse']
        self.histogram_freq = 1
        self.write_graph = True
        self.write_images = False
        self.update_freq = 'epoch'
        self.profile_batch = 0
        self.confusion_matrix_freq = 10
        self.confusion_matrix_size = [180, 180]
        self.confusion_matrix_raw_ext = 'png'
        self.confusion_matrix_to_save = 'both'

        # Create output folder
        # os.makedirs(self.dirs['output'], exist_ok=True)

    def initialize(self):
        """
        Constructs data pipeline and default training callbacks 
        """

        # Prepare the pipeline
        self.__prepare_pipeline()

        # Compile model
        self.__compile_model()

        # Print models' statistics
        self.__show_model()

        # Prepare training callbacks
        self.__prepare_callbacks()

        return self

    def run(self):
        """
        Runs the training flow and tests the result model
        """

        # Train the model
        self.__train()

        # Test the model
        self.__test()

        pass

    def read_image(self, filepath):
        image = tf.io.read_file(filepath)
        # print(tf.strings.regex_full_match(filepath, '(.*).jpg'))
        # TODO moze zle dzialac - trzeba bedzie
        # if tf.strings.regex_full_match(filepath, '(.*).jpg'):
        print('jpg image')
        image = tf.image.decode_jpeg(image, channels=3)
        # else:
        #     print('png image')
        #     image = tf.image.decode_png(image, channels=3)

        image = tf.cast(image, tf.dtypes.as_dtype(self.image_datatype))
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]])
        return image

    def __create_dataset(self, directory):
        labels_dirs = glob(os.path.join(directory, '*'))
        datas = {'files': [], 'labels': []}

        for label_dir in labels_dirs:
            label = os.path.basename(os.path.normpath(label_dir))
            files = glob(os.path.join(label_dir, '*'))

            datas['files'].extend(files)
            datas['labels'].extend([label for f in files])

        ds = tf.data.Dataset.from_tensor_slices((datas['files'], datas['labels']))
        ds = ds.map(
            lambda file, file_label: (self.read_image(file), file_label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds

    def __prepare_pipeline(self):

        # Create data pipe (contains training and validation sets)
        self.training_set = self.__create_dataset(self.dirs['training'])
        self.validation_set = self.__create_dataset(self.dirs['validation'])
        self.test_set = self.__create_dataset(self.dirs['test'])

        self.training_set = ImageAugmentation(
            dtype='float32'
        )(self.training_set)

        # Apply batching to the data sets
        self.training_set = self.training_set.batch(self.batch_size)
        self.validation_set = self.validation_set.batch(self.batch_size)
        self.test_set = self.test_set.batch(self.batch_size)
        # TODO zastabnowić się nad dodaniem prefetch batch

    def __compile_model(self):
        """
        Compiles the model setting required optimizer and loss function
        """

        # Initialize optimizer
        optimizer = tf.keras.optimizers.get({
            "class_name": 'adam',
            "config": {"learning_rate": self.learning_rate}}
        )

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=self.metrics)

    def __show_model(self):
        """
        Prints model statistics
        """

        print('\n\n')
        self.model.summary()
        print('\n\n')

    def __prepare_callbacks(self):
        # Create output folder for weights saves
        # modeldir = os.path.join(self.dirs['output'], 'weights')
        # os.makedirs(modeldir, exist_ok=True)

        # Create the logdir
        os.makedirs(self.log_dir, exist_ok=True)

        # List names of images' classes
        class_folders = glob(os.path.join(self.dirs['validation'], '*'))
        class_names = [os.path.basename(folder) for folder in class_folders]
        class_names.sort()

        # Create tensorboard callback
        tensorboard_callback = LRTensorBoard(
            log_dir=self.log_dir,
            histogram_freq=self.histogram_freq,
            write_graph=self.write_graph,
            write_images=self.write_images,
            update_freq=self.update_freq,
            profile_batch=self.profile_batch
        )
        self.callbacks.append(tensorboard_callback)

        # Create a confusion matrix callback
        os.makedirs(os.path.join(self.log_dir, 'validation/cm'), exist_ok=True)
        cm_callback = ConfusionMatrixCallback(
            logdir=os.path.join(self.log_dir, 'validation/cm'),
            validation_set=self.validation_set,
            class_names=class_names,
            freq=self.confusion_matrix_freq,
            fig_size=self.confusion_matrix_size,
            raw_fig_type=self.confusion_matrix_raw_ext,
            to_save=self.confusion_matrix_to_save
        )
        self.callbacks.append(cm_callback)

        # Create a checkpoint callback
        checkpoint_name = os.path.join(self.log_dir, 'weights-epoch_{epoch:02d}-val_loss_{val_loss:.2f}.hdf5')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            save_weights_only=True,
            verbose=True,
            save_freq='epoch',
            save_best_only=True
        )
        self.callbacks.append(checkpoint_callback)

        # Create learning rate scheduler callback
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=2e-1,
            patience=4,
            verbose=1,
            min_delta=5e-2,
            cooldown=0,
            min_lr=1e-7
        )
        self.callbacks.append(lr_callback)

    def __train(self):
        """
        Runs the training flow
        """
        for elem in self.training_set:
            print('step')
            print(elem[1])
        # print(np.shape(self.training_set.as_numpy_iterator()))
        # Start training

        #jest jakis problem z wielkością outputu, pewnie wynik powinien byc tablicą a nie pojedyncza wartoscia?
        #siec musi dostac jako y wektor liczb z jedynką tam, gdzoie jest klasa, i zerami w pozostałych przypadkach
        #na razie dostaje jedna labelke tekstowa
        self.__history = self.model.fit(
            x=self.training_set,
            validation_data=self.validation_set,
            epochs=self.epochs,
            initial_epoch=0,
            callbacks=self.callbacks,
            verbose=1,
            workers=4,
            use_multiprocessing=True,
            shuffle=False
        )

        # Create path to the output folder
        # historydir = os.path.join(self.log_dir, 'history')
        # os.makedirs(historydir, exist_ok=True)
        #
        #     # Compute index of the subrun
        # subrun = len(glob(os.path.join(historydir, '*.pickle'))) + 1
        #
        #     # Create path to the output file
        #     historyname = os.path.join(historydir, 'subrun_{:d}'.format(subrun))

        with open(os.path.join(self.log_dir, 'history.pickle'), 'wb') as history_file:
            pickle.dump(self.__history.history, history_file)

    def __test(self):
        """
        Tests the result model
        """

        if self.pipe.test_set is not None and self.test_model is True:
            # # If the best models hould be evaluated, load appropriate weights
            # if self.logging_params['test_model'] == 'best':
            #     # Find epoch's index of the best score
            #     best_score = np.nanmin(np.array(self.__history.history['val_loss']))
            #
            #     # Find the weights file
            #     modeldir = os.path.join(self.dirs['output'], 'weights')
            #     weights_file = glob(os.path.join(modeldir, '*val_loss_{:.2f}*'.format(best_score)))[0]
            #
            #     # Load weights
            #     self.model.load_weights(weights_file)

            # Create path to the output folder
            testdir = os.path.join(self.log_dir, 'test')
            os.makedirs(testdir, exist_ok=True)

            # Compute index of the subrun
            subrun = len(glob(os.path.join(testdir, '*.pickle'))) + 1

            # Create basename for CM raw files (include type of the model that is tested: lates or best)
            testbasename = 'test_results'

            # Create path to the output file
            testname = os.path.join(testdir, testbasename)

            # List names of images' classes
            class_folders = glob(os.path.join(self.dirs['validation'], '*'))
            class_names = [os.path.basename(folder) for folder in class_folders]
            class_names.sort()

            # Prepare a new Confusion Matrix callback for the test set
            cm_callback = ConfusionMatrixCallback(
                logdir=os.path.join(testdir, 'cm'),
                validation_set=self.test_set,
                class_names=class_names,
                freq=self.confusion_matrix_freq,
                fig_size=self.confusion_matrix_size,
                raw_fig_type=self.confusion_matrix_raw_ext,
                to_save=self.confusion_matrix_to_save,
                basename=testbasename
            )

            # Wrap Confusion Matrix callback to be usable with tf.keras.Model.evaluate() method
            cm_callback.set_model(self.model)
            cm_callback_test_decorator = \
                tf.keras.callbacks.LambdaCallback(on_test_end=lambda logs: cm_callback.on_epoch_end('', logs))

            # Evaluate test score
            test_dict = self.model.evaluate(
                x=self.test_set,
                verbose=1,
                workers=4,
                use_multiprocessing=True,
                return_dict=True,
                callbacks=[cm_callback_test_decorator]
            )

            # Save test score
            with open(testname + '.pickle', 'wb') as test_file:
                pickle.dump(test_dict, test_file)

