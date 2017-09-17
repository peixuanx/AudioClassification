import os
import os.path

AUDIO_EXT = "wav"
class TrainData:
    def __init__(self, train_inputs, train_targets):
        self.train_inputs = train_inputs
        self.train_targets = train_targets


class TestData:
    def __init__(self, test_inputs):
        self.test_inputs = test_inputs


def get_files(file_directory_path):
    files = os.listdir(file_directory_path)
    waves = []
    names = []
    for file in files:
        if AUDIO_EXT in file:
            names.append(file)
            waves.append(os.path.join(file_directory_path, file))

    return waves, names
