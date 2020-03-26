import numpy
import pathlib
from PIL import Image
from tensorflow.keras import preprocessing


class DangerIndexDataloader:
    def __init__(self, directory, size):
        self.path = pathlib.Path(directory)
        self.size = size

    def load_data(self):
        data_list = []
        label_list = []
        for file_path in self.path.glob("**/*.png"):
            scaled_image = Image.open(file_path).convert("L").resize(self.size)
            array_2d = numpy.array(scaled_image)
            array_3d = numpy.expand_dims(array_2d, 2)
            data_list.append(array_3d)
            #
            danger_index = 1 if int(file_path.stem) > 45 else 0
            label_list.append(data_list)
        #
        generator = preprocessing.image.ImageDataGenerator()
        return generator.flow(numpy.array(data_list), label_list, batch_size=5)
