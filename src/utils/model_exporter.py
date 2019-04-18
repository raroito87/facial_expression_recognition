import os
import torch
import matplotlib.pyplot as plt

class ModelExporter:
    def __init__(self, name, root_dir=os.path.dirname(__file__)):
        self.name = name.lower()
        self.data = {}

        directory_template = f'{root_dir}/../../models/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print (f'created directory {self.directory}')
            os.makedirs(self.directory)

    def save_nn_model(self, model, optimizer, args = [], debug=True):
        if model.device is not 'cpu':
            model.to('cpu')

        the_dict = model.state_dict()
        the_dict['model_class'] = type(model).__name__
        the_dict['optimizer_class']= type(optimizer).__name__
        the_dict['args'] = args

        file_name = f'{model.name}.pt'
        torch.save(the_dict, self.directory + file_name)

        if debug:
            print(f'model saved {model}')

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
