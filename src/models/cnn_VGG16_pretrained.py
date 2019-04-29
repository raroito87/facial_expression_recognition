import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


# Neural Network
class CnnVGG16Pretrained(M.VGG):
    def __init__(self, name, features, d_out = 7, prepare_model = False,  dtype=torch.float, device='cpu'):
        super().__init__(features)

        self.name = name
        self.d_out = d_out#number of classes

        self.dtype = dtype
        self.device = device

        if prepare_model:
            self.prepare_model()


    def prepare_model(self):
        # Freeze training for all layers
        for param in self.features.parameters():
            param.requires_grad = False

        #change last layer, it won't be freezed
        num_features = self.classifier[6].in_features
        features = list(self.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, self.d_out)])  # Add our layer with 7 outputs
        self.classifier = nn.Sequential(*features) # Replace the model classifier


    def get_batch(self, x, y, batch_idx, batch_size):
        # we need the structure (#n_samples, #channels_per_sample, size_im_x, size_im_y)
        batch_x = x[batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
        batch_x = batch_x.reshape(batch_size, self.ch_in, self.size_im[0], self.size_im[1])

        batch_y = y[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        return batch_x, batch_y

    def reshape_data(self, x):
        return x.reshape(x.shape[0], self.ch_in, self.size_im[0], self.size_im[1])

    def get_args(self):
        return [self.name, self.d_out]# self.dtype , self.device] I cant save stype because is a torch specific type and I get THE ERROR
