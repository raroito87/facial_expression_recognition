import os
import torch
from models import  CnnSimple

#todo
#domehow the modul should also be saved and loaded so I dont have to ass here all model classes as import
class ModelImporter:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/models/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

    def load_nn_model(self, model_name, n_features = 0, n_classes = 0, n_iter=0):
        file_name = f'{model_name}_{n_features}_{n_classes}_{n_iter}.pt'
        the_dict = torch.load(self.directory + file_name)

        #model = eval(the_dict['model_class'])(*the_dict['args'])
        model = eval(the_dict['model_class'])(*the_dict['args'])
        optimizer = eval(the_dict['optimizer_class'])

        #clean the dictionry
        del the_dict['args']
        del the_dict['model_class']
        del the_dict['optimizer_class']

        print(f'load model {model}')

        model.load_state_dict(state_dict=the_dict)
        model.eval()
        return model
