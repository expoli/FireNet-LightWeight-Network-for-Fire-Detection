import os


class pathconfig:
    def __init__(self):
        self.RESULT_ROOT_PATH = 'results00/results07/'
        self.TRAINING_TIME = 'train'
        self.MODEL_SAVE_PATH = 'models/'
        self.TENSORBOARD_LOG_PATH = 'tensorboard_log/'
        self.CHECKPOINT_PATH = 'checkpoint_path/'
        self.CHECKPOINT_FORMAT = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

    def get_tensorblard_path(self):
        path = self.RESULT_ROOT_PATH + self.TENSORBOARD_LOG_PATH
        check_dir(path)
        return path

    def get_model_save_path(self):
        path = self.RESULT_ROOT_PATH + self.MODEL_SAVE_PATH
        check_dir(path)
        return path

    def get_checkpoint_path(self):
        path = self.RESULT_ROOT_PATH + self.CHECKPOINT_PATH + self.CHECKPOINT_FORMAT
        check_dir(self.RESULT_ROOT_PATH + self.CHECKPOINT_PATH)
        return path

    def set_root_path(self, new_root_path):
        self.RESULT_ROOT_PATH = new_root_path
        return self.RESULT_ROOT_PATH


def check_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        # directory already exists
        pass
