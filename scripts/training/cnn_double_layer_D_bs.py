from train import TrainClassifier2
from utils import Preprocessing, ModelExporter
import torch
import argparse
from models import CnnDoubleLayer
import time
import matplotlib.pyplot as plt

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')
    parser.add_argument('--s_patterns', default=False, help='save patterns images')

    args=parser.parse_args()

    n_classes = 7
    n_epochs = 300
    learning_rate = 0.0001
    batch_size = 32

    pre = Preprocessing('fer2013')
    pre.load_data(filename='DatasetD.csv', name='train')

    X_df = pre.get(name='train').drop(columns=['emotion'])
    y_df = pre.get(name='train')['emotion']

    dtype = torch.float
    device = torch.device("cpu")

    model_name = f'cnn_double_layer_D_bs_{learning_rate}_{batch_size}_{n_epochs}_{n_classes}'
    model = CnnDoubleLayer(model_name, d_out=n_classes)
    model.train()

    train_classifier = TrainClassifier2(model, X_df, y_df)
    t = time.time()
    trained_model , optimizer, criterion, loss_hist, loss_val_hist, f1_val_hist = train_classifier.run_train(n_epochs = n_epochs, lr=learning_rate, batch_size=batch_size)
    print(f'trained in {time.time() - t} sec')
    pre.save_results(loss_hist, loss_val_hist, f1_val_hist, f'{model_name}')

    if args.s_model:
        m_exporter = ModelExporter('fer2013_DatasetD')
        m_exporter.save_nn_model(trained_model, optimizer,trained_model.get_args())

    if args.s_patterns:
        detected_patterns1 = trained_model.get_detected_patterns1()
        for idx in range(10):
            plt.figure(1, figsize=(20, 10))
            for p in range(trained_model.n_patterns1):
                pattern = detected_patterns1[idx][p].reshape(detected_patterns1.shape[2],
                                                             detected_patterns1.shape[3])
                patern_np = pattern.detach().numpy().reshape(24, 24)
                plt.subplot(2, 5, 1 + p)
                plt.imshow(patern_np, cmap='gray', interpolation='none')
            pre.save_plt_as_image(plt, f'patterns_1_{idx}')

        detected_patterns2 = trained_model.get_detected_patterns2()
        for idx in range(10):
            plt.figure(1, figsize=(20, 10))
            for p in range(trained_model.n_patterns2):
                pattern = detected_patterns2[idx][p].reshape(detected_patterns2.shape[2],
                                                             detected_patterns2.shape[3])
                patern_np = pattern.detach().numpy().reshape(12, 12)
                plt.subplot(3, 5, 1 + p)
                plt.imshow(patern_np, cmap='gray', interpolation='none')
            pre.save_plt_as_image(plt, f'patterns_2_{idx}')
