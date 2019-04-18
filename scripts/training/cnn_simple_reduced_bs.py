import os

from train import TrainClassifier2
from utils import Preprocessing, ModelExporter
import argparse
from models import CnnSimple
import time
import matplotlib.pyplot as plt

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='fer2013')
    parser.add_argument('--s_model', default=True, help='save trained model')
    parser.add_argument('--s_patterns', default=False, help='save patterns images')

    args=parser.parse_args()

    script_root_dir = os.path.dirname(__file__)
    pre = Preprocessing('fer2013', root_dir=script_root_dir)
    pre.load_data(filename='train_reduced_norm_centered.csv', name='train')
    pre.load_data(filename='test_public_norm_centered.csv', name='validate')

    X_train_df = pre.get(name='train').drop(columns=['emotion'])
    y_train_df = pre.get(name='train')['emotion']
    X_val_df = pre.get(name='validate').drop(columns=['emotion'])
    y_val_df = pre.get(name='validate')['emotion']

    n_classes = 7
    n_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    model_name = f'cnn_simple_reduced_bs_{learning_rate}_{batch_size}_{n_epochs}_{n_classes}'
    model = CnnSimple(model_name, d_out=n_classes)

    train_classifier = TrainClassifier2(model, X_train_df, y_train_df, X_val_df, y_val_df, root_dir=script_root_dir)
    t = time.time()
    trained_model, optimizer, criterion, \
    train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,\
    val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist = train_classifier.run_train(n_epochs=n_epochs,
                                                                          lr=learning_rate,
                                                                          batch_size=batch_size)

    print(f'trained in {time.time() - t} sec')

    if args.s_model:
        m_exporter = ModelExporter('fer2013_reduced',root_dir=script_root_dir)
        m_exporter.save_nn_model(trained_model, optimizer,trained_model.get_args())
        m_exporter.save_results(f'{model_name}',
                     train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,
                     val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist)

    if args.s_patterns:
        detected_patterns = trained_model.get_detected_patterns()
        for idx in range(10):
            plt.figure(1, figsize=(20, 10))
            for p in range(trained_model.n_patterns):
                pattern = detected_patterns[idx][p].reshape(detected_patterns.shape[2], detected_patterns.shape[3])
                patern_np = pattern.detach().numpy().reshape(24, 24)
                plt.subplot(3, 5, 1 + p)
                plt.imshow(patern_np, cmap='gray', interpolation='none')
            pre.save_plt_as_image(plt, f'patterns_{idx}')

