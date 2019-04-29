import cv2
import torch
import os
from image_utils import ImageConverter
from utils import ModelImporter
import numpy as np
import argparse

#https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

root_dir = os.path.dirname(__file__)
data_path = '{root_dir}/../../data/images_to_predict/'
data_directory = data_path.format(root_dir=root_dir, name='data')

emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--img_name', default='0.png', help='save trained model')
    args = parser.parse_args()

    file = f'{data_directory}{args.img_name}'
    print(file)
    frame = cv2.imread(file, 0)

    if frame is not None:
        im_conv = ImageConverter()
        temp_img = im_conv.crop_frame_to_square(im_conv.convert_frame_to_grey_scale(frame))
        img_48 = im_conv.rescale(temp_img, size=48)
        img_48 = im_conv.normalize_frame(img_48)
        img_array = im_conv.reshape_frame_to_array(img_48)

        dtype = torch.float
        device = torch.device("cpu")

        model_name = f'best_model'
        m_importer = ModelImporter('best')
        model = m_importer.load_nn_model(model_name)
        model.eval()

        x = model.reshape_data(torch.tensor([img_48], device=device, dtype=dtype))
        predicted_emotion = None
        predicted_emotion2 = None
        with torch.no_grad():
            results = model(x).squeeze().detach().numpy()
            sort = np.sort(results, axis = 0)
            idx1 = np.where(results == sort[6])
            idx2 = np.where(results == sort[5])
            print(results)
            predicted_emotion = emotion_dict[idx1[0][0]]
            predicted_emotion2 = emotion_dict[idx2[0][0]]

        frame = im_conv.upscale(temp_img, size=400)
        cv2.putText(frame, 'your face gives me', (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        cv2.putText(frame, predicted_emotion, (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 6)
        cv2.putText(frame, '...and a bit of', (5, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        cv2.putText(frame, predicted_emotion2, (130, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)
        cv2.imshow('result', frame)
        #https://stackoverflow.com/questions/31350240/python-opencv-open-window-on-top-of-other-applications/44852940
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
        cv2.waitKey(0)
        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

