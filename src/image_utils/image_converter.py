import cv2
import numpy as np

class ImageConverter:

    def __init__(self):
        pass

    def crop_frame_to_square(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]

        y1, y2, x1, x2 = self._get_crop_coordinates(rows, cols)

        return frame[y1:y2, x1:x2].copy()

    def convert_frame_to_grey_scale(self, frame):
        if frame.shape[2] == 1:  # frame is already greyscale, only one channel
            return frame

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def rescale(self, frame, size = 48):
        return cv2.resize(frame,(size,size), interpolation = cv2.INTER_AREA)

    def reshape_frame_to_array(self, frame):
        return frame.reshape(1, 48*48)

    def _get_crop_coordinates(self, rows, cols):
        if cols > rows:
            #rows remain same, crop columns
            y1 = 0
            y2 = rows - 1

            x1 = (cols - rows)*0.5
            x2 = x1 + rows
            return int(y1), int(y2), int(x1), int(x2)
        else:
            #rows crop, cols remain same
            y1 = (rows - cols)*0.5
            y2 = y1 + cols

            x1 = 0
            x2 = cols - 1
            return int(y1), int(y2), int(x1), int(x2)
