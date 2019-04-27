from utils import Preprocessing, ModelExporter
from train import TrainClassifier2
from models import CnnVGG16Pretrained
from image_utils import ImageConverter

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
import torchvision

import os
import time
import argparse

if not __name__ == '__main_':
    print('train learning transfer VGG16 DatasetA')

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')
    parser.add_argument('--s_patterns', default=False, help='save patterns images')

    args=parser.parse_args()

    current_working_dir = os.getcwd()
    print('current_working_dir: ', current_working_dir)
    pre = Preprocessing('fer2013', root_dir=current_working_dir)

    pre.load_data('train_reduced_norm.csv.gz', name='train')
    pre.load_data('test_public_norm.csv.gz', name='val')

    X = pre.get('val').drop(columns=['emotion'])
    y = pre.get('val')['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    val = pd.DataFrame(X_test)
    val['emotion'] = y_test
    pre.set(name='val', value=val)

    print(pre.get(name='val').head())

    train_pixels = pre.get(name='train').drop(columns=['emotion'])
    val_pixels = pre.get(name='val').drop(columns=['emotion'])

    print('data loaded')

    img_conv = ImageConverter()

    train_pixel_np = train_pixels.values
    train_pixel_224_np = np.zeros(shape=[train_pixel_np.shape[0], 224 * 224])
    for i in range(train_pixel_np.shape[0]):
        image = train_pixel_np[i].reshape(48, 48)
        newimg = img_conv.upscale(image)
        train_pixel_224_np[i] = newimg.reshape(1, 224 * 224)

    print('train data scaled to 224x224')

    val_pixel_np = val_pixels.values
    val_pixel_224_np = np.zeros(shape=[val_pixels.shape[0], 224 * 224])
    for i in range(val_pixel_np.shape[0]):
        image = val_pixel_np[i].reshape(48, 48)
        newimg = img_conv.upscale(image)
        val_pixel_224_np[i] = newimg.reshape(1, 224 * 224)


    print('val data scaled to 224x224')

    #train_pixel_224_np = np.round_(train_pixel_224_np, decimals=4)
    #val_pixel_224_np = np.round_(train_pixel_224_np, decimals=4)


    ###Train
    y_train_df = pre.get(name='train')['emotion']
    y_val_df = pre.get(name='val')['emotion']

    n_classes = 7
    n_epochs = 10
    learning_rate = 0.001
    batch_size = 64
    dtype = torch.float

    model_conv = torchvision.models.vgg16_bn(pretrained=True)  # _bn batch normalization
    model_name = f'cnn_VGG16_pretrained_A_bs_{learning_rate}_{batch_size}_{n_epochs}_{n_classes}'
    model = CnnVGG16Pretrained(model_name, features = model_conv.features)
    model.load_state_dict(state_dict=model_conv.state_dict())
    model.prepare_model()
    model.train()

    train_classifier = TrainClassifier2(model, train_pixel_224_np, y_train_df, val_pixel_224_np, y_val_df, to_rgb=True,
                                        root_dir=current_working_dir)

    t = time.time()
    trained_model, optimizer, criterion, \
    train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,\
    val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist = train_classifier.run_train(n_epochs=n_epochs,
                                                                          lr=learning_rate,
                                                                          batch_size=batch_size)
    print(f'trained in {time.time() - t} sec')

    if args.s_model:
        m_exporter = ModelExporter('fer2013_datasetA', root_dir=current_working_dir)
        m_exporter.save_nn_model(trained_model, optimizer,trained_model.get_args())
        m_exporter.save_results(f'{model_name}',
                     train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,
                     val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist)
        print('model saved')

    print('try to save train and val')


    train_pixel_224 = pd.DataFrame(train_pixel_224_np)
    val_pixel_224 = pd.DataFrame(val_pixel_224_np)
    train_pixel_224.to_csv('train_224_gray.csv')
    val_pixel_224.to_csv('val_224_gray.csv')
