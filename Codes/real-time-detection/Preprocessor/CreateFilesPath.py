import os

from tqdm import tqdm


class CreateFilesPath:
    def __init__(self, rootPath):
        self.rootPath = rootPath

    def create_path_list(self):
        video_files_path = []
        root_path = self.rootPath

        try:
            for img in tqdm(os.listdir(root_path)):  # iterate over each image
                try:
                    video_files_path.append(os.path.join(root_path, img))  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    print(e)
                    exit(1)
            return video_files_path

        except Exception as e:
            print(e)
            exit(1)
