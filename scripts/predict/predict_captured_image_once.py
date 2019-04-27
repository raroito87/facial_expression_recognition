import cv2
import torch
import os
from image_utils import ImageCapture, ImageConverter
from utils import ModelImporter
import numpy as np

#https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

if not __name__ == '__main_':

    cap_img = ImageCapture()
    frame = cap_img.capture_image()

    if frame is not None:
        im_conv = ImageConverter()
        temp_img = im_conv.crop_frame_to_square(im_conv.convert_frame_to_grey_scale(frame))
        img_48 = im_conv.rescale(temp_img, size=48)
        img_48 = im_conv.normalize_frame(img_48)
        img_array = im_conv.reshape_frame_to_array(img_48)

        dtype = torch.float
        device = torch.device("cpu")

        model_name = f'best'
        m_importer = ModelImporter('best_model')
        model = m_importer.load_nn_model(model_name,)
        model.eval()

        x = model.reshape_data(torch.tensor([img_48], device=device, dtype=dtype))
        predicted_emotion = None
        predicted_emotion2 = None
        with torch.no_grad():
            results = model(x).squeeze().detach().numpy()
            print(results)
            sort = np.sort(results, axis = 0)
            idx1 = np.where(results == sort[6])
            idx2 = np.where(results == sort[5])
            print(idx1[0][0])
            print(idx2[0][0])
            predicted_emotion = emotion_dict[idx1[0][0]]
            predicted_emotion2 = emotion_dict[idx2[0][0]]

        prediction = f'today your face radiates {predicted_emotion}....and a bit of {predicted_emotion2}'
        print(prediction)
        cv2.imshow(prediction, cv2.resize(img_48,(int(288),int(288)), interpolation=cv2.INTER_CUBIC))
        #https://stackoverflow.com/questions/31350240/python-opencv-open-window-on-top-of-other-applications/44852940
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
        cv2.waitKey(0)

