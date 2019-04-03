import sys
#print(sys.path)
sys.path.append("/Users/raroito/PycharmProjects/facial_expression_recognition/src/")

from train import TrainClassifierEncoder
from utils import Preprocessing, ModelExporter
import pandas as pd
import numpy as np
import math
import torch
import argparse
from models import AnnAutoencoder

import matplotlib.pyplot as plt

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')

    args=parser.parse_args()

    n_epochs = 100

    pre = Preprocessing('fer2013')
    pre.load_data(filename='train_norm.csv', name='train')

    X_train_df = pre.get(name='train').drop(columns=['emotion'])
    y_train_df = pre.get(name='train')['emotion']

    dtype = torch.float
    device = torch.device("cpu")

    H1 = 1764
    n_features = len(X_train_df.columns)
    n_features_encoded = 1296
    print(f'features {n_features}')
    print(f'H1 {H1}')
    print(f'n_features_encoded {n_features_encoded}')

    model_name = 'ann_encoder'
    model = AnnAutoencoder(model_name, n_features, H1, n_features_encoded)

    learning_rate = 0.001
    batch_size = 128

    train_classifier = TrainClassifierEncoder(model, X_train_df, X_train_df)
    trained_model , opt, c, loss_hist, loss_val_hist, best_model_param = train_classifier.run_train(n_epochs = n_epochs, lr=learning_rate, batch_size=batch_size)
    pre.save_results(loss_hist, loss_val_hist, f'{model_name}')

    #evaluate last trained model:
    pre.load_data(filename='test_public_norm.csv', name='test')

    X_test_df = pre.get(name='test').drop(columns=['emotion'])
    y_test_df = pre.get(name='test')['emotion']

    X_test = model.reshape_data(torch.tensor(X_test_df.values, device=device, dtype=dtype))
    y_test = torch.tensor(y_test_df.values, device=device, dtype=torch.long)

    X_train = model.reshape_data(torch.tensor(X_train_df.values, device=device, dtype=dtype))
    y_train = torch.tensor(y_train_df.values, device=device, dtype=torch.long)

    X_pred = trained_model(X_test)
    print(f'test accuracy last trained model {c(X_pred, X_test)}')

    trained_model.load_state_dict(state_dict=best_model_param)
    trained_model.eval()

    X_pred = trained_model(X_test)
    print(f'test accuracy best model {c(X_pred, X_test)}')

    if args.s_model:
        m_exporter = ModelExporter('fer2013')
        m_exporter.save_nn_model(trained_model, opt, 0, n_features_encoded, n_epochs, trained_model.get_args())

    X_train_encoded = trained_model.encoder(X_train)
    X_test_encoded = trained_model.encoder(X_test)
    X_test_decoded = trained_model.decoder(X_test_encoded)

    X_train_encoded_df = pd.DataFrame(X_train_encoded.detach().numpy())
    X_test_encoded_df = pd.DataFrame(X_test_encoded.detach().numpy())

    cols = list(range(1, n_features_encoded + 1))

    X_train_encoded_df.columns = cols
    X_test_encoded_df.columns = cols

    train_encoded_data = X_train_encoded_df.join(y_train_df)
    test_encoded_data = X_test_encoded_df.join(y_test_df)

    pre.set('train_encoded', train_encoded_data)
    pre.set('test_encoded', test_encoded_data)

    pre.save('train_encoded')
    pre.save('test_encoded')

    plt.figure(1, figsize=(30, 20))
    for idx in range(30):
        image = X_test[idx].detach().numpy().reshape(48, 48)
        image2 = X_test_encoded[idx].detach().numpy().reshape(int(math.sqrt(n_features_encoded)), int(math.sqrt(n_features_encoded)))
        image3 = X_test_decoded[idx].detach().numpy().reshape(48, 48)
        # Call signature: subplot(nrows, ncols, index, **kwargs)
        plt.subplot(10, 9, 1 + idx * 3)
        plt.imshow(image, cmap='hot', interpolation='none')
        plt.subplot(10, 9, 2 + idx * 3)
        plt.imshow(image2, cmap='winter', interpolation='none')
        plt.subplot(10, 9, 3 + idx * 3)
        plt.imshow(image3, cmap='winter', interpolation='none')

    pre.save_plt_as_image(plt, f'encoded_{n_features_encoded}')

    #for idx in range(30):
    #    plt.figure(1, figsize=(10, 5))
    #    plt.subplot(1, 3, 1)
    #    X_test_np = X_test[idx].detach().numpy().reshape(16, 16)
    #    plt.imshow(X_test_np, cmap='hot', interpolation='none')
#
    #    plt.subplot(1, 3, 2)
    #    test_num_np = X_test_encoded[idx].detach().numpy().reshape(int(math.sqrt(n_features_encoded)), int(math.sqrt(n_features_encoded)))
    #    plt.imshow(test_num_np, cmap='hot', interpolation='none')
#
    #    plt.subplot(1, 3, 3)
    #    X_test_decoded_np = X_test_decoded[idx].detach().numpy().reshape(16, 16)
    #    plt.imshow(X_test_decoded_np, cmap='hot', interpolation='none')
#
    #    pre.save_plt_as_image(plt, f'encoded_{y_test[idx].item()}')

    if n_features_encoded == 1:
        decoder = trained_model.decoder
        values = np.arange(-15, 15, 0.5, dtype=float)

        plt.clf()
        plt.figure(1, figsize=(20, 10))
        i = 1
        for v in values:
            data = torch.tensor([v], device=device, dtype=torch.float)
            print(v)
            data_decoded = trained_model.decoder(data)
            image = data_decoded.detach().numpy().reshape(16, 16)
            plt.subplot(10, 9, i)
            plt.imshow(image, cmap='hot', interpolation='none')
            i = i + 1

        pre.save_plt_as_image(plt, f'swap_{n_features_encoded}')








