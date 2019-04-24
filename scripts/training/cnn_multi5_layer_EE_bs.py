from train import TrainClassifier2
from utils import Preprocessing, ModelExporter
import torch
import argparse
from models import CnnMulti5Layer
import time
import os

if not __name__ == '__main_':
    print('train cnn_multi5_layer_balanced_sampling_EE')

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')
    parser.add_argument('--s_patterns', default=False, help='save patterns images')

    args=parser.parse_args()

    n_classes = 7
    n_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    current_working_dir = os.getcwd()
    print('current_working_dir: ', current_working_dir)
    pre = Preprocessing('fer2013', root_dir=current_working_dir)
    pre.load_data(filename='DatasetEE.csv.gz', name='train')
    pre.load_data(filename='test_public_norm_centered.csv.gz', name='validate')

    X_train_df = pre.get(name='train').drop(columns=['emotion'])
    y_train_df = pre.get(name='train')['emotion']
    X_val_df = pre.get(name='validate').drop(columns=['emotion'])
    y_val_df = pre.get(name='validate')['emotion']

    dtype = torch.float

    model_name = f'cnn_multi5_layer_EE_bs_{learning_rate}_{batch_size}_{n_epochs}_{n_classes}'
    model = CnnMulti5Layer(model_name, d_out=n_classes)
    model.train()

    train_classifier = TrainClassifier2(model, X_train_df, y_train_df, X_val_df, y_val_df, root_dir=current_working_dir)
    t = time.time()
    trained_model, optimizer, criterion, \
    train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,\
    val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist = train_classifier.run_train(n_epochs=n_epochs,
                                                                          lr=learning_rate,
                                                                          batch_size=batch_size)
    print(f'trained in {time.time() - t} sec')

    if args.s_model:
        m_exporter = ModelExporter('fer2013_datasetEE', root_dir=current_working_dir)
        m_exporter.save_nn_model(trained_model, optimizer,trained_model.get_args())
        m_exporter.save_results(f'{model_name}',
                     train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,
                     val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist)

