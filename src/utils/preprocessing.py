import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self, name, root_dir=os.path.dirname(__file__)):
        self.name = name.lower()
        self.data = {}

        directory_template = f'{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'

        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, **kwargs)
        self.data[name] = df
        return df

    def load_data_np(self, filename, filetype='csv', sep = ',', type = np.float32,  **kwargs):
        filepath = f'{self.directory}/{filename}'

        file = f'{filepath}.{filetype}'
        data_np = np.loadtxt(file, delimiter = sep, dtype = type)
        return data_np

    def save(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)

    def cleanup(self, name, *, drop=None, drop_duplicates=False, dropna=None):
        data = self.data[name]

        if drop is not None:
            data = data.drop(columns=drop)

        if drop_duplicates is True:
            data = data.drop_duplicates()

        if dropna is not None:
            if 'axis' not in dropna:
                dropna['axis'] = 1

            data = data.dropna(**dropna)

        self.data['clean'] = data

    def label_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        encoder = preprocessing.LabelEncoder()
        labels = pd.DataFrame()

        label_index = 0
        for column in columns:
            encoder.fit(data[column])
            label = encoder.transform(data[column])
            labels.insert(label_index, column=column, value=label)
            label_index += 1

        data = data.drop(columns, axis=1)
        data = pd.concat([data, labels], axis=1)
        self.data['clean'] = data

        return data

    def one_hot_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        categorical = pd.get_dummies(data[columns], dtype='int')
        data = pd.concat([data, categorical], axis=1, sort=False)
        self.data['clean'] = data

        return data

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value

    def save_results(self, name,
                     train_loss_hist, train_acc_hist, train_f1_hist, train_b_hist,
                     val_loss_hist, val_acc_hist, val_f1_hist, val_b_hist):
        title = f'{name}_loss'
        plt.figure(1)
        plt.plot(train_loss_hist)
        plt.plot(val_loss_hist)
        plt.title = title
        self.save_plt_as_image(plt, title)
        plt.close()

        title = f'{name}_acc'
        plt.figure(1)
        plt.plot(train_acc_hist)
        plt.plot(val_acc_hist)
        plt.title = title
        self.save_plt_as_image(plt, title)
        plt.close()

        title = f'{name}_f1'
        plt.figure(1)
        plt.plot(train_f1_hist)
        plt.plot(val_f1_hist)
        plt.title = title
        self.save_plt_as_image(plt, title)
        plt.close()

        title = f'{name}_balanced_score'
        plt.figure(1)
        plt.plot(train_b_hist)
        plt.plot(val_b_hist)
        plt.title = title
        self.save_plt_as_image(plt, title)
        plt.close()

    def save_plt_as_image(self, plt, name, format='.png'):
        #!!!!!! call this before plt.show()
        path = f'{self.directory}/plots/'
        if not os.path.exists(path):
            print(f'created directory {path}')
            os.makedirs(path)

        filename = path + name + format
        print(f'save {filename}')
        plt.savefig(filename)

