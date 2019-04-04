import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network
class CnnDoubleLayer(nn.Module):
    def __init__(self, name, ch_in = 1, d_out = 7, size_im = [48, 48],
                 n_patterns1 = 10, n_patterns2 = 15,
                 kernel_pool = 2, dtype=torch.float, device='cpu'):
        super(CnnDoubleLayer, self).__init__()

        self.name = name

        self.ch_in = ch_in#number of chaels for input
        self.d_out = d_out#number of classes
        self.size_im = size_im#size in px of the input images
        self.n_patterns = n_patterns#num of pattern to look for int he convolution
        self.detected_patterns = None
        self.kernel_pool = kernel_pool
        self.drop_layer = nn.Dropout(p=0.5)

        self.dtype = dtype
        self.device = device

        #Input channels = 1, output channels = n_patterns
        self.conv1 = torch.nn.Conv2d(ch_in, n_patterns1, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(n_patterns1)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_pool, stride=2, padding=0)


        self.conv2 = torch.nn.Conv2d(n_patterns1, n_patterns2, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(n_patterns2)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_pool, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(int(n_patterns2 * size_im[0]/(kernel_pool**2) * size_im[1]/(kernel_pool**2)), 32)

        #  7 output features for our 7 defined classes
        self.fc2 = torch.nn.Linear(32, d_out)

    def forward(self, x):

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        self.detected_patterns1 = x

        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        self.detected_patterns2 = x

        x = x.view(-1, int(self.n_patterns2 * self.size_im[0]/(self.kernel_pool**2) * self.size_im[1]/(self.kernel_pool**2)))

        x = F.relu(self.batchnorm_l1(self.fc1(x)))

        x = self.fc2(x)
        return (x)

    def get_batch(self, x, y, batch_idx, batch_size):
        # we need the structure (#n_samples, #channels_per_sample, size_im_x, size_im_y)
        batch_x = x[batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
        batch_x = batch_x.reshape(batch_size, self.ch_in, self.size_im[0], self.size_im[1])

        batch_y = y[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        return batch_x, batch_y

    def reshape_data(self, x):
        return x.reshape(x.shape[0], self.ch_in, self.size_im[0], self.size_im[1])

    def get_args(self):
        return [self.name, self.ch_in, self.d_out, self.size_im , self.n_patterns,
                self.kernel_pool]# self.dtype , self.device] I cant save stype because is a torch specific type and I get THE ERROR

    def get_detected_patterns1(self):
        return self.detected_patterns1

    def get_detected_patterns2(self):
        return self.detected_patterns2

