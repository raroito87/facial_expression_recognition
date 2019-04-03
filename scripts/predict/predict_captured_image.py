import sys
#print(sys.path)
sys.path.append("/Users/raroito/PycharmProjects/facial_expression_recognition/src/")

import torch
from image_utils import ImageCapture, ImageConverter
from utils import ModelImporter

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
        img_array = im_conv.reshape_frame_to_array(img_48)

        dtype = torch.float
        device = torch.device("cpu")

        model_name = 'cnn_simple'
        m_importer = ModelImporter('fer2013_reduced')
        n_classes = 7
        n_epochs = 100
        model = m_importer.load_nn_model(model_name, 0, n_classes, n_epochs)

        x = model.reshape_data(torch.tensor([img_48], device=device, dtype=dtype))

        predicted_emotion = emotion_dict[model(x).argmax(1).item()]
        print(f'today you are {predicted_emotion}')

        #detected_patterns = model.get_detected_patterns()
        #if detected_patterns is not None:
        #    plt.figure(1, figsize=(20, 10))
        #    for p in range(model.n_patterns):
        #        pattern = detected_patterns[0][p].reshape(detected_patterns.shape[2], detected_patterns.shape[3])
        #        patern_np = pattern.detach().numpy().reshape(24, 24)
        #        plt.subplot(3, 5, 1 + p)
        #        plt.imshow(patern_np, cmap='gray', interpolation='none')
        #    plt.show()
