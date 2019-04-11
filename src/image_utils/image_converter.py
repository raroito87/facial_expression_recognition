import cv2
import numpy as np
import math

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

    def reshape_array_to_frame(self, array):
        return array.reshape(48, 48)

    def flip_frame_horitzontally(self, frame):
        return cv2.flip(frame, 1)

    def rotate_image(self, img, degrees):
        #add padding so after rotation there wont be 'blank spaces'
        img_padding, org_rows, org_cols = self._add_padding_to_img(img)

        rows, cols = img_padding.shape
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degrees, 1)
        dst = cv2.warpAffine(img_padding, M, (cols, rows), borderValue=255)

        #crop image to the original size
        y1 = rows//2 - org_rows//2
        y2 = y1 + org_rows
        x1 = cols//2 - org_cols//2
        x2 = x1 + org_cols

        rot_img = dst[y1:y2, x1:x2].copy()

        return rot_img

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

    def _compute_mean_bg_value(self, img):
        rows, cols = img.shape
        rows_2 = rows//2
        cols_2 =  cols//2

        v = np.array([img[0, cols_2], img[rows-1, cols_2], img[rows_2, 0], img[rows_2, cols - 1]])

        return np.mean(v)

    def _add_padding_to_img(self, img):
        #extend the image by half
        rows, cols = img.shape
        rows_2 = rows//2
        cols_2 =  cols//2
        reflect101 = cv2.copyMakeBorder(img, rows_2, rows_2, cols_2, cols_2, cv2.BORDER_REPLICATE)# top , bottom, left, right

        return reflect101, rows, cols
