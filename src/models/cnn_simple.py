import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network
class CnnSimple(nn.Module):
    def __init__(self, name, ch_in = 1, d_out = 7, size_im = [48, 48], n_patterns = 15,
                 kernel_pool = 2, dtype=torch.float, device='cpu'):
        super(CnnSimple, self).__init__()

        self.name = name

        self.ch_in = ch_in#number of chaels for input
        self.d_out = d_out#number of classes
        self.size_im = size_im#size in px of the input images
        self.n_patterns = n_patterns#num of pattern to look for int he convolution
        self.detected_patterns = None
        self.kernel_pool = kernel_pool

        self.dtype = dtype
        self.device = device


        # Input channels = 1, output channels = 6
        self.conv1 = torch.nn.Conv2d(ch_in, n_patterns, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_pool, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(int(n_patterns * size_im[0]/kernel_pool * size_im[1]/kernel_pool), 32)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(32, d_out)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 16, 16) to (6, 16, 16)
        x = F.relu(self.conv1(x))

        # Size changes from (6, 16, 16) to (6, 8, 8)
        x = self.pool(x)
        self.detected_patterns = x
        # Reshape data to input to the input layer of the neural net
        # Size changes from (6, 8, 8) to (1, 384)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, int(self.n_patterns * self.size_im[0]/self.kernel_pool * self.size_im[1]/self.kernel_pool))

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 384) to (1, 32)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 32) to (1, 10)
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

    def get_detected_patterns(self):
        return self.detected_patterns
