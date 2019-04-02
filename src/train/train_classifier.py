import torch
from sklearn.model_selection import train_test_split
import copy

class TrainClassifier():
    def __init__(self, model, inputs, targets):
        #inputs and target are DF
        self.model = model
        self.inputs = inputs
        self.targets = targets

        self.data_is_prepared = False


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
        if(self.data_is_prepared == False):
            self.prepare_data()

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Train
        loss_hist = []
        loss_validate_hist = []
        model_versions = {}

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
                    outputs = self.model(self.x)
                    loss = criterion(outputs, self.y)
                    loss_hist.append(loss.item())

                    outputs_val = self.model(self.x_val)
                    loss_val = criterion(outputs_val, self.y_val)
                    loss_validate_hist.append(loss_val.item())
                    model_versions[loss_val.item()] = copy.copy(self.model.state_dict())
                    print(t, ' train_loss: ',loss.item(), 'validate_loss: ', loss_val.item())

        print(f'optimal iteration: {idx*loss_validate_hist.index(min(loss_validate_hist))}')

        y_pred = self.model(self.x).argmax(1)
        accuracy_soft = (y_pred == self.y.long()).float().mean()
        print(f'training accuracy: {accuracy_soft}')

        return self.model, optimizer, criterion, loss_hist, loss_validate_hist
