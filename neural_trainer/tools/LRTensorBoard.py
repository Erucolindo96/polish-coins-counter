# ================================================================================================================
 # @ Author: Krzysztof Pierczyk
 # @ Create Time: 2021-01-08 11:47:58
 # @ Modified time: 2021-01-09 13:40:51
 # @ Description:
 #     
 #     Wrapper around tf.keras.callbacks.TensoBoard that introduces learning rate's logging into the
 #     tesnorboard
 #
 # @ Source code: https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
 # ================================================================================================================

import tensorflow as tf

class LRTensorBoard(tf.keras.callbacks.TensorBoard):

    """
    Wrapper around tf.keras.callbacks.TensoBoard that introduces learning rate's logging into the
    tesnorboard.
    """

    def __init__(self, **kwargs):

        """
        Initialized underlying tf.keras.callbacks.TensorBoard

        Params
        ------
        kwargs : keyword arguments
            @see tf.keras.callbacks.TensorBoard
        """

        super().__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):

        """
        Adds learning rate to the logs and calls tf.keras.callbacks.TensorBoard.on_epoch_end()

        Params
        ------
        epoch: Int
            index of the current epoch
        logs : Dict
            Keras-internal logs
        """

        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
