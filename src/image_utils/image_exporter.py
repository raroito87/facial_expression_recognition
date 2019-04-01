import os
import datetime
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

    def _save_original(self, frame, path):
        img_name = path + 'original.png'
        print(img_name)
        cv2.imwrite(img_name, frame)
