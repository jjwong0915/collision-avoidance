import numpy
import pathlib
from PIL import Image
from tensorflow.keras import preprocessing


class DangerIndexDataloader:
    def __init__(self, size):
        self.size = size

    def read_depth_image(self, path):
        scaled_image = Image.open(path).convert("L").resize(self.size)
        array_2d = numpy.transpose(numpy.array(scaled_image))
        array_3d = numpy.expand_dims(array_2d, 2)
        return array_3d

    def load_data(self, directory):
        data_list = []
        label_list = []
        for file_path in pathlib.Path(directory).glob("**/*.png"):
            data_list.append(self.read_depth_image(file_path))
            #
            labeled_index = int(file_path.stem) / 50
            label_list.append(labeled_index)
        #
        generator = preprocessing.image.ImageDataGenerator()
        return generator.flow(numpy.array(data_list), label_list, batch_size=5)
