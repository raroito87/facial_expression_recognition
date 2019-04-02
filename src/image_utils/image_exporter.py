import os
import datetime
from .image_converter import ImageConverter
import cv2

class ImageExporter:

    def __init__(self):
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        root_dir = os.path.dirname(__file__)
        directory_template = f'{root_dir}/../../data/captured_images/'
        self.directory = directory_template.format(root_dir=root_dir, name=self.date)

        if not os.path.exists(self.directory):
            print(f'Creating captured data directory')
            os.makedirs(self.directory)


    def save_capture(self, frame):
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        path = self.directory + self.date + '/'

        if not os.path.exists(path):
            print(f'Creating {self.date} directory')
            os.makedirs(path)

        self._save_original(frame, path)
        self._save_converted(frame, path)

    def _save_original(self, frame, path, name = 'original'):
        img_name = path + name + '.png'
        cv2.imwrite(img_name, frame)

    def _save_converted(self, frame, path):
        im_conv = ImageConverter()

        temp_img = im_conv.crop_frame_to_square(im_conv.convert_frame_to_grey_scale(frame))

        img_48 = im_conv.rescale(temp_img, size = 48)
        img_name = path + 'img_48.png'
        cv2.imwrite(img_name, img_48)

        img_96 = im_conv.rescale(temp_img, size = 96)
        img_name = path + 'img_96.png'
        cv2.imwrite(img_name, img_96)
