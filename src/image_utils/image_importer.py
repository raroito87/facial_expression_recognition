import os
from utils import Preprocessing

class ImageImporter:

    def __init__(self, name, data = 'fer2013_DatasetA', file = 'train.csv', ):
        self.name = name.lower()
        self.image = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        prep = Preprocessing(name = data)
        self.data = prep.load_data(name = 'img_arrays', filename = file)

    def load_data_as_img(self, index = 0, size = 48):
        emotion = self.data.loc[index, :]['emotion']
        img_array = self.data.drop(columns = ['emotion']).loc[index, :].values
        img = img_array.reshape(size, size)

        return img, emotion
