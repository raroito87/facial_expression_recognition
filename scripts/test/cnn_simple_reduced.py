#import sys
#print(sys.path)
#sys.path.append("/Users/raroito/PycharmProjects/facial_expression_recognition/src/")

from utils import ModelImporter, Preprocessing
import torch

if not __name__ == '__main_':

    pre = Preprocessing('fer2013')
    pre.load_data(filename='test_public_norm.csv', name='test')

    X_df = pre.get(name='test').drop(columns=['emotion'])
    y_df = pre.get(name='test')['emotion']


    dtype = torch.float
    device = torch.device("cpu")

    model_name = 'cnn_simple'

    m_importer = ModelImporter('fer2013_reduced')

    n_classes = 7
    n_epochs = 100
    model = m_importer.load_nn_model(model_name, 0, n_classes, n_epochs)

    X_test = model.reshape_data(torch.tensor(X_df.values, device=device, dtype=dtype))
    y_test = torch.tensor(y_df.values, device=device, dtype=torch.long)

    y_pred = model(X_test).argmax(1)
    print(y_pred)
    print(y_test)

    accuracy_soft = (y_pred == y_test).float().mean()

    print(f'test accuracy {accuracy_soft.item()}')
