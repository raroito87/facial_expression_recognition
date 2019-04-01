import os

import torch

import imageio
from skimage.transform import rescale, resize
from skimage import color

class Image_Importer:

    def __init__(self, name):
        self.name = name.lower()
        self.image = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/images/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)


    def load_image_as_grey(self, number):
        filename = 'written' + str(number) +'.png'
        filepath = f'{self.directory}/{filename}'

        im = imageio.imread(filepath)
        im = color.rgb2gray(im)
        self.image[number] = im
        return im

    def get_image_as_256px_array(self, number):
        #this method prepares the images to the same type of the zip.train
        #256 long images and values of pixels from -1 to 1
        # training images are inverted, background black foreground white

        dtype = torch.float
        device = torch.device("cpu")

        image_resized = resize(self.image[number], (16, 16), anti_aliasing=True)
        print('len image resized: ', len(image_resized))
        flatten = []
        for row in image_resized:
            for v in row:
                flatten.append(-v * 2 + 1)

        x = torch.tensor(flatten, device=device, dtype=dtype)
        return x.view(1, len(x))


