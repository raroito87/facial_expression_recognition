import unittest
from image_utils import ImageConverter
import cv2
import numpy as np
import os

root_dir = os.path.dirname(__file__)
data_path = '{root_dir}/data/'
data_directory = data_path.format(root_dir=root_dir, name='data')

def _load_image(filename):
    file = f'{data_directory}{filename}'
    print(file)
    img = cv2.imread(file, 0)
    return img

def _save_image(img, filename):
    file = f'{data_directory}{filename}'
    cv2.imwrite(file, img)

class TestImageConverter(unittest.TestCase):

    def test_rotate_image(self):
        file_name = 'sad_48.png'
        img = _load_image(file_name)
        self.assertEqual(img.size == 0, False)

        degrees = 90
        file_name_rot = f'rot{degrees}_sad_48.png'
        img_rot_ = _load_image(file_name_rot)
        self.assertEqual(img_rot_.size == 0, False)

        img_converter = ImageConverter()
        img_rot = img_converter.rotate_image(img, degrees)

        self.assertEqual(np.array_equal(img_rot, img_rot_), True)

    def test_rotate_image_save(self):
        file_name = 'sad_48.png'
        img = _load_image(file_name)
        self.assertEqual(img.size == 0, False)
        degrees = 15
        img_converter = ImageConverter()
        img_rot = img_converter.rotate_image(img, degrees)
        _save_image(img_rot, f'rot{degrees}_{file_name}')







