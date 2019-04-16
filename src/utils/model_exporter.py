import os
import torch
import matplotlib.pyplot as plt

class ModelExporter:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../models/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print (f'created directory {self.directory}')
            os.makedirs(self.directory)

    def save_nn_model(self, model, optimizer, args = [], debug=True):
        the_dict = model.state_dict()
        the_dict['model_class'] = type(model).__name__
        the_dict['optimizer_class']= type(optimizer).__name__
        the_dict['args'] = args

        file_name = f'{model.name}.pt'
        torch.save(the_dict, self.directory + file_name)

        if debug:
            print(f'model saved {model}')



    def save_results(self, result_train, result_val, f1_val, name):
        title = f'{name}_loss'
        plt.figure(1)
        plt.plot(result_train)
        plt.plot(result_val)
        plt.title = title
        self.save_plt_as_image(plt, title)
        plt.close()

        title = f'{name}_f1'
        plt.figure(1)
        plt.plot(f1_val)
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
