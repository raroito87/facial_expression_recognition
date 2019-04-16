import torch
import copy
from utils import ModelExporter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from data import  Fer2013Dataset
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
#import resource

class TrainClassifier2():
    def __init__(self, model, inputs, targets, test_size = 0.1):
        #inputs and target are DF
        self.model = model

        #model to evaluate
        self.model_eval = copy.deepcopy(self.model)
        self.model_eval.to('cpu')

        inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=test_size)

        self.sampler = self._create_sampler(targets_train.values.astype(int))

        # Generators
        self.training_set = Fer2013Dataset(inputs=inputs_train, targets=targets_train, device='cpu')
        self.validation_set = Fer2013Dataset(inputs=inputs_val, targets=targets_val, device='cpu')

        #https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        #torch.backends.cudnn.benchmark = True#if inputs sizes remain the same, should go faster


        self.model.to(self.device)

        print(f'use cuda: {self.use_cuda}')

    def run_train(self, n_epochs, lr=0.001, batch_size=256):
        print(f'training model: {self.model.name}')

        training_generator = DataLoader(self.training_set, sampler=self.sampler,  batch_size =batch_size, num_workers=2)

        self.model.train()#set model to training mode
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Train
        train_loss_hist = []
        val_loss_hist = []

        train_acc_hist = []
        val_acc_hist = []

        train_f1_hist = []
        val_f1_hist = []

        model_versions = {}

        m_exporter = ModelExporter('temp')
        model_name = copy.deepcopy(self.model.name)

        f = 10
        for t in range(n_epochs):
            for i, (batch_x, batch_y) in enumerate(training_generator):

                # move to gpu if possible
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Berechne die Vorhersage (foward step)
                outputs = self.model(batch_x)

                # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
                loss = criterion(outputs, batch_y)

                # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del batch_x, batch_y

                if i % 10 == 0:
                    print('.', end='', flush=True)

            if t % f == 0:
                model_params = copy.deepcopy(self.model.state_dict())
                model_versions[t] = model_params

                train_loss, train_acc, train_f1, val_loss, val_acc, val_f1 = self._evaluate(model_params, criterion)

                train_loss_hist.append(train_loss)
                train_acc_hist.append(train_acc)
                train_f1_hist.append(train_f1)

                val_loss_hist.append(val_loss)
                val_acc_hist.append(val_acc)
                val_f1_hist.append(val_f1)

                print('\n{} loss t:{:0.3f} v: {:0.3f} | acc t: {:0.4f} v: {:0.3f} | f1 t: {:0.3f} v: {:0.3f}'.format(t,
                      train_loss, val_loss, train_acc, val_acc, train_f1, val_f1))

                self.model.name = f'{model_name}_epoch{t}'
                m_exporter.save_nn_model(self.model, optimizer, self.model.get_args(), debug=False)

        print(f'\n ####training finished####')
        best_iteration = f*val_loss_hist.index(min(val_loss_hist))
        print(f'optimal iteration val_loss: {best_iteration}')
        best_iteration_acc = f * val_acc_hist.index(max(val_acc_hist))
        print(f'optimal iteration val_acc: {best_iteration_acc}')
        best_iteration_f1 = f * val_f1_hist.index(max(val_f1_hist))
        print(f'optimal iteration val_f1: {best_iteration_f1}')

        # use the best trained model
        self.model.load_state_dict(state_dict=model_versions[best_iteration])
        self.model.eval()# set model to test model
        self.model.name = f'{model_name}'

        return self.model, optimizer, criterion,\
               train_loss_hist, train_acc_hist, train_f1_hist,\
               val_loss_hist, val_acc_hist, val_f1_hist

    def _create_sampler(self, target_np):
        class_sample_count = np.array([len(np.where(target_np == t)[0]) for t in np.unique(target_np)])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in target_np])).double()
        return WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    def _evaluate(self, model_param, criterion):
        # evaluate in CPU
        # can't move all the training dataset to GPU, in my case and resources it is too much
        self.model_eval.load_state_dict(state_dict=model_param)
        self.model_eval.eval()

        train_prob = self.model_eval(self.training_set.x_data.to('cpu'))
        train_pred = train_prob.argmax(1)
        train_loss = criterion(train_prob, self.training_set.y_data.to('cpu'))
        train_acc = (train_pred == self.training_set.y_data.long()).float().mean()
        train_f1 = metrics.f1_score(self.training_set.y_data.long().numpy(), train_pred.numpy(), average='macro')

        val_prob = self.model_eval(self.validation_set.x_data.to('cpu'))
        val_pred = val_prob.argmax(1)
        val_loss = criterion(val_prob, self.validation_set.y_data.to('cpu'))
        val_acc = (val_pred == self.validation_set.y_data.long()).float().mean()
        val_f1 = metrics.f1_score(self.validation_set.y_data.long().numpy(), val_pred.numpy(), average='macro')

        return train_loss.item(), train_acc, train_f1, val_loss.item(), val_acc, val_f1



