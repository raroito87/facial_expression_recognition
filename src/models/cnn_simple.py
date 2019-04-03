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


        #Input channels = 1, output channels = n_patterns
        self.conv1 = torch.nn.Conv2d(ch_in, n_patterns, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_pool, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(int(n_patterns * size_im[0]/kernel_pool * size_im[1]/kernel_pool), 32)

        #  7 output features for our 7 defined classes
        self.fc2 = torch.nn.Linear(32, d_out)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, size_im[0], size_im[1]) to (num_patterns, size_im[0], size_im[1])
        x = F.relu(self.conv1(x))

        # Size changes from (num_patterns, size_im[0], size_im[1]) to (num_patterns, size_im[0]/poolsize, size_im[1]/poolsize)
        x = self.pool(x)
        self.detected_patterns = x

        # Reshape data to input to the input layer of the neural net
        # Size changes from a structured dimensional torch to a 1D feature vector
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, int(self.n_patterns * self.size_im[0]/self.kernel_pool * self.size_im[1]/self.kernel_pool))

        # Computes the activation of the first fully connected layer
        # Stil uses a Relu as activation Function
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # use Sa Sigmoid because is the last layer
        x = torch.sigmoid(self.fc2(x))
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
