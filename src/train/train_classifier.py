import torch
from sklearn.model_selection import train_test_split
import copy
from utils import ModelExporter
from sklearn import metrics

class TrainClassifier():
    def __init__(self, model, inputs, targets):
        #inputs and target are DF
        self.model = model
        self.inputs = inputs
        self.targets = targets

        self.data_is_prepared = False

        self.lr = 0.01
        self.decay_rate = 0.99


    def prepare_data(self, test_size=0.1):
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(self.inputs, self.targets, test_size=test_size)

        self.N = inputs_train.shape[0]

        self.x = self.model.reshape_data(torch.tensor(inputs_train.values, device=self.model.device, dtype=self.model.dtype))
        self.y = torch.tensor(targets_train.values, device=self.model.device, dtype=torch.long).squeeze()

        self.x_val = self.model.reshape_data(torch.tensor(inputs_val.values, device=self.model.device, dtype=self.model.dtype))
        self.y_val = torch.tensor(targets_val.values, device=self.model.device, dtype=torch.long).squeeze()

        del inputs_train
        del inputs_val
        del targets_train
        del targets_val

        self.data_is_prepared = True
        return


    def run_train(self, n_epochs, lr=0.001, batch_size=256):
        self.lr = lr
        if(self.data_is_prepared == False):
            self.prepare_data()

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
            for batch in range(0, int(self.N / batch_size)):
                # Berechne den Batch
                batch_x, batch_y = self.model.get_batch(self.x, self.y, batch, batch_size)

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
                #current_lr = self._get_lr(optimizer)
                #self._set_lr(optimizer, self._update_lr(optimizer, t))
                #print(f'learning_rate: {current_lr}')

                outputs = self.model(self.x)
                loss = criterion(outputs, self.y)
                loss_hist.append(loss.item())

                outputs_val = self.model(self.x_val)
                loss_val = criterion(outputs_val, self.y_val)
                loss_val_hist.append(loss_val.item())
                model_versions[t] = copy.deepcopy(self.model.state_dict())

                accuracy_train = (outputs.argmax(1) == self.y.long()).float().mean()
                accuracy_val= (outputs_val.argmax(1) == self.y_val.long()).float().mean()
                f1_score = metrics.f1_score(self.y_val.long().numpy(), outputs_val.argmax(1).numpy(), average='macro')
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
        self.model.eval()
        self.model.name = f'{model_name}'

        y_pred = self.model(self.x).argmax(1)
        accuracy_soft = (y_pred == self.y.long()).float().mean()
        print(f'training accuracy: {accuracy_soft}')

        return self.model, optimizer, criterion, loss_hist, loss_val_hist, f1_val_hist

    #https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _set_lr(selfs, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _update_lr(self, optimizer, epoch):
        #return self.lr/(1 + self.decay_rate*epoch)
        if epoch == 0:
            return self.lr
        return self.lr/epoch


