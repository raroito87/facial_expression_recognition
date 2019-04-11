import torch
import copy
from utils import ModelExporter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from data import  Fer2013Dataset
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

class TrainClassifier2():
    def __init__(self, model, inputs, targets, test_size = 0.1):
        #inputs and target are DF
        self.model = model
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=test_size)

        # Generators
        self.training_set = Fer2013Dataset(inputs=inputs_train, targets=targets_train)

        self.validation_set = Fer2013Dataset(inputs=inputs_val, targets=targets_val)
        #validation_generator = data.DataLoader(validation_set, **params)

        #https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True#if inputs sizes remain the same, should go faster

        self.model.to(self.device)
        self.sampler = None


    def create_sampler(self, batch_size = 128):
        target_np = self.training_set.y_data.numpy().astype(int)

        #create the stratified sampler
        class_sample_count = np.array([len(np.where(target_np==t)[0]) for t in np.unique(target_np)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target_np])
        return WeightedRandomSampler(samples_weight, batch_size, replacement=True)


    def run_train(self, n_epochs, lr=0.001, batch_size=256):
        self.sampler = self.create_sampler(batch_size)

        training_generator = DataLoader(self.training_set, sampler=self.sampler)

        self.model.train()#set model to training mode
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Train
        loss_hist = []
        loss_val_hist = []
        f1_val_hist = []
        model_versions = {}

        m_exporter = ModelExporter('temp')
        model_name = copy.deepcopy(self.model.name)

        for t in range(n_epochs):
            for batch_x, batch_y in training_generator:
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

            # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)

            idx = 10
            if t % idx == 0:
                outputs = self.model(self.training_set.x_data)
                loss = criterion(outputs, self.training_set.y_data)
                loss_hist.append(loss.item())

                outputs_val = self.model(self.validation_set.x_data)
                loss_val = criterion(outputs_val, self.validation_set.y_data)
                loss_val_hist.append(loss_val.item())
                model_versions[t] = copy.deepcopy(self.model.state_dict())

                accuracy_train = (outputs.argmax(1) == self.training_set.y_data.long()).float().mean()
                accuracy_val= (outputs_val.argmax(1) == self.validation_set.y_data.long()).float().mean()
                f1_score = metrics.f1_score(self.validation_set.y_data.long().numpy(), outputs_val.argmax(1).numpy(), average='macro')
                f1_val_hist.append(f1_score)

                print(t, ' train_loss: ',loss.item(), 'val_loss: ', loss_val.item(), ' - train_acc: ',
                    accuracy_train, ', val_acc: ', accuracy_val, ', val_f1: ', f1_score)

                self.model.name = f'{model_name}_epoch{t}'
                m_exporter.save_nn_model(self.model, optimizer, self.model.get_args(), debug=False)

        best_iteration = idx*loss_val_hist.index(min(loss_val_hist))
        print(f'optimal iteration val_loss: {best_iteration}')
        best_iteration_f1 = idx * f1_val_hist.index(max(f1_val_hist))
        print(f'optimal iteration val_f1: {best_iteration_f1}')

        #use the best trained model
        self.model.load_state_dict(state_dict=model_versions[best_iteration])
        self.model.eval()#set model to test mode
        self.model.name = f'{model_name}'

        y_pred = self.model(self.training_set.x_data).argmax(1)
        accuracy_soft = (y_pred == self.training_set.y_data.long()).float().mean()
        print(f'training accuracy: {accuracy_soft}')

        return self.model, optimizer, criterion, loss_hist, loss_val_hist, f1_val_hist
