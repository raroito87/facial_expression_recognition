from utils import ModelImporter, Preprocessing
import torch

if not __name__ == '__main_':

    pre = Preprocessing('fer2013_DatasetA')
    pre.load_data(filename='test_public_norm.csv', name='test')

    X_df = pre.get(name='test').drop(columns=['emotion'])
    y_df = pre.get(name='test')['emotion']

    n_classes = 7
    n_totalepochs = 200
    learning_rate = 0.0001
    batch_size = 32
    epoch_n = 40#for the temp folder

    dtype = torch.float
    device = torch.device("cpu")

    model_name = f'cnn_double_layer_balanced_sampling_exp_c1_rot_{learning_rate}_{batch_size}_{n_totalepochs}_{n_classes}'
    model_name_bestf1 = f'{model_name}_epoch150'
    model_name_bestvalacc = f'{model_name}_epoch150'

    m_importer = ModelImporter('fer2013_DatasetD')
    model = m_importer.load_nn_model(model_name)
    model.eval()

    X_test = model.reshape_data(torch.tensor(X_df.values, device=device, dtype=dtype))
    y_test = torch.tensor(y_df.values, device=device, dtype=torch.long)

    y_pred = model(X_test).argmax(1)
    print(y_pred)
    print(y_test)

    accuracy_soft = (y_pred == y_test).float().mean()

    print(f'test accuracy {accuracy_soft.item()}')
