import os
import torch
from models import CnnSimple, CnnDoubleLayer, CnnTripleLayer, CnnMulti5Layer, CnnMulti8Layer, CnnVGG16Pretrained
import torch.nn as nn
from torch.optim import Adam

#todo
#somehow the modul should also be saved and loaded so I dont have to ass here all model classes as import
class ModelImporter:
    def __init__(self, name, root_dir = os.path.dirname(__file__)):
        self.name = name.lower()
        self.data = {}

        directory_template = f'{root_dir}/../../models/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

    def load_nn_model(self, model_name):
        file_name = f'{model_name}.pt'
        the_dict = torch.load(self.directory + file_name, map_location='cpu')

        print(the_dict['args'])
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
