import tensorflow as tf

from projectconfig import pathconfig


class TensorboardCallBack:
    def __init__(self, histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch'):
        self.tensorboard_log_path = pathconfig.pathconfig().get_tensorblard_path()
        self.histogram_freq = histogram_freq
        self.batch_size = batch_size
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata
        self.embeddings_data = embeddings_data
        self.update_freq = update_freq

    def build_cb(self):
        tonserboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log_path,
                                                              histogram_freq=self.histogram_freq,
                                                              batch_size=self.batch_size,
                                                              write_graph=self.write_graph,
                                                              write_grads=self.write_grads,
                                                              write_images=self.write_images,
                                                              embeddings_freq=self.embeddings_freq,
                                                              embeddings_layer_names=self.embeddings_layer_names,
                                                              embeddings_metadata=self.embeddings_metadata,
                                                              embeddings_data=self.embeddings_data,
                                                              update_freq=self.update_freq)

        return tonserboard_callback
