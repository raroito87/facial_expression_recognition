from train import TrainClassifier2
from utils import Preprocessing, ModelExporter
import torch
import argparse
from models import CnnMulti8Layer
import time

if not __name__ == '__main_':
    print('train cnn_multi8_layer_balanced_sampling_D')

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')
    parser.add_argument('--s_patterns', default=False, help='save patterns images')

    args=parser.parse_args()

    n_classes = 7
    n_epochs = 200
    learning_rate = 0.0001
    batch_size = 32

    pre = Preprocessing('fer2013')
    pre.load_data(filename='DatasetD.csv', name='train')

    X_df = pre.get(name='train').drop(columns=['emotion'])
    y_df = pre.get(name='train')['emotion']

    dtype = torch.float

    model_name = f'cnn_multi8_layer_D_bs_{learning_rate}_{batch_size}_{n_epochs}_{n_classes}'
    model = CnnMulti8Layer(model_name, d_out=n_classes)
    model.train()

    train_classifier = TrainClassifier2(model, X_df, y_df)
    t = time.time()
    trained_model , optimizer, criterion, loss_hist, loss_val_hist, f1_val_hist = train_classifier.run_train(n_epochs = n_epochs, lr=learning_rate, batch_size=batch_size)
    print(f'trained in {time.time() - t} sec')
    pre.save_results(loss_hist, loss_val_hist, f1_val_hist, f'{model_name}')

    if args.s_model:
        m_exporter = ModelExporter('fer2013_datasetD')
        m_exporter.save_nn_model(trained_model, optimizer,trained_model.get_args())

