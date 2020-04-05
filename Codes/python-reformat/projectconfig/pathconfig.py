import os

RESULT_ROOT_PATH = 'results03/'
TRAINING_TIME = 'train'
MODEL_SAVE_PATH = 'models/'
TENSORBOARD_LOG_PATH = 'tensorboard_log/'
CHECKPOINT_PATH = 'checkpoint_path/weights.{epoch:02d}-{val_loss:.2f}.hdf5'


def get_tensorblard_path():
    return RESULT_ROOT_PATH + TENSORBOARD_LOG_PATH


def get_model_save_path():
    return RESULT_ROOT_PATH + MODEL_SAVE_PATH


def get_checkpoint_path():
    return RESULT_ROOT_PATH + CHECKPOINT_PATH


def check_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        # directory already exists
        pass
