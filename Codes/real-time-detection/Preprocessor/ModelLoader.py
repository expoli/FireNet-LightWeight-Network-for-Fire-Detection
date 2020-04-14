from tensorflow.keras.models import load_model


class LoadModel:
    def __init__(self, modelPath):
        self.modelPath = modelPath

    def load_saved_model(self):
        # loading the stored model from file
        model = load_model(self.modelPath)
        return model
