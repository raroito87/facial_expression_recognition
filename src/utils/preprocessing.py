import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re
import torch
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'

        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, **kwargs)
        self.data[name] = df
        return df

    def load_all_texts_from_directory(self, path, *, name, clean_text=True):
        directory = f'{self.directory}/{path}'
        all_files = os.listdir(directory)
        assert len(all_files) > 0, 'directory is empty!'

        n_files = 0
        texts = []
        for file in all_files:
            rel_file_name = f'{path}/{file}'
            text = None
            if clean_text:
                text = self.load_clean_text(rel_file_name)
            else:
                text = self.load_text(rel_file_name)

            texts += text
            n_files = n_files + 1

        self.data[name] = pd.DataFrame(texts, columns=['texts'])
        print('loaded {} {} files to {}'.format(name, n_files, name))
        return self.data[name]

    def load_text(self, filename):
        filepath = f'{self.directory}/{filename}'
        f = open(filepath, "r")
        return [f.read()]

    def load_clean_text(self, filename):
        filepath = f'{self.directory}/{filename}'
        f = open(filepath, "r")
        return [self._clean_up_word_list_np(np.str.split(f.read(), sep=' '))]

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

    #def save_model_(self, model, name):
    #    path = f'{self.directory}/model/'
    #    if not os.path.exists(path):
    #        print(f'created directory {path}')
    #        os.makedirs(path)
#
    #    filename = path + name
    #    pickle.dump(model, open(filename, 'wb'))
    #    print(f'saved model in {filename}')

    def save_model(self, model, name):
        path = f'{self.directory}/model/'
        if not os.path.exists(path):
            print(f'created directory {path}')
            os.makedirs(path)

        filename = path + name
        torch.save(model, filename)

    def load_model(self, name):
        filename = f'{self.directory}/model/{name}'
        return torch.load(filename)

    def save_results(self, result_train, result_val, name):
        title = name
        plt.figure(1)
        plt.plot(result_train)
        plt.plot(result_val)
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

    def _clean_up_word_list_np(self, words_np):

        def functions(z):
            #clean characters
            z = re.sub(r'[^\w\s]', '', str(z))
            #clean new line
            z = re.sub(r'[\n]', '', str(z))
            #all to lower case
            z = str(z).lower()
            #remove numbers
            z = re.sub(r'[0-9]', '', str(z))
            return str(z)

        return np.apply_along_axis(functions, 0, words_np)
