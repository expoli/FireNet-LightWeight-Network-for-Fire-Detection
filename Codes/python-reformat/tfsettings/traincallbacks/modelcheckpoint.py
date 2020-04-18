import tensorflow as tf

from projectconfig import pathconfig


class ModelCheckpointCallBack:
    def __init__(self, period=1, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto'):
        self.checkpoint_path = pathconfig.pathconfig().get_checkpoint_path()
        self.period = period
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weight_only = save_weights_only
        self.mode = mode

    def build_cb(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                                 monitor=self.monitor,
                                                                 verbose=self.verbose,
                                                                 save_best_only=self.save_best_only,
                                                                 save_weights_only=self.save_weight_only,
                                                                 mode=self.mode,
                                                                 period=self.period)
        return checkpoint_callback
