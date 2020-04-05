from projectconfig import pathconfig


class modelsaver:
    def __init__(self, ):
        self.model_save_path = pathconfig.pathconfig().get_model_save_path()

    def save(self, model):
        print('[INFO] saving model to folder.....')
        model.save(self.model_save_path)
