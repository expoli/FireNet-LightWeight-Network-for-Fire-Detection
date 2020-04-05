import tensorflow as tf


class InitGpu:
    def __init__(self):
        self.gpus = None

    def init(self):
        self.gpus = tf.config.experimental.list_physical_devices('GPU')
        if self.gpus:
            try:
                # Memory growth must be set before GPUs have been initialized
                # Currently, memory growth needs to be the same across GPUs
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(self.gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                return 0
            except RuntimeError as e:
                print(e)
                return -1
        else:
            print("[WARNING] Can't found GPU!")
            return 0
