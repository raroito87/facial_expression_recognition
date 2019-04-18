import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network
class CnnMulti8Layer(nn.Module):
    def __init__(self, name, ch_in = 1, d_out = 7, size_im = [48, 48],
                 n_patterns1 = 64, n_patterns2 = 64,
                 n_patterns3 =128, n_patterns4 = 128,
                 n_patterns5 = 256, n_patterns6 = 256,
                 n_patterns7 = 512, n_patterns8 = 512,
                 kernel_pool = 2, dtype=torch.float, device='cpu'):
        super(CnnMulti8Layer, self).__init__()

        self.name = name

        self.ch_in = ch_in#number of chanels for input
        self.d_out = d_out#number of classes
        self.size_im = size_im#size in px of the input images
        self.n_patterns1 = n_patterns1#num of pattern to look for int he convolution
        self.n_patterns2 = n_patterns2#num of pattern to look for int he convolution
        self.n_patterns3 = n_patterns3#num of pattern to look for int he convolution
        self.n_patterns4 = n_patterns4#num of pattern to look for int he convolution
        self.n_patterns5 = n_patterns5#num of pattern to look for int he convolution
        self.n_patterns6 = n_patterns6#num of pattern to look for int he convolution
        self.n_patterns7 = n_patterns7#num of pattern to look for int he convolution
        self.n_patterns8 = n_patterns8#num of pattern to look for int he convolution

        self.kernel_pool = kernel_pool
        self.drop_hidden = nn.Dropout(p=0.5)
        self.drop_visible = nn.Dropout(p=0.4)

        self.dtype = dtype
        self.device = device

        #Input channels = 1, output channels = n_patterns
        self.conv1 = torch.nn.Conv2d(ch_in, n_patterns1, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(n_patterns1)
        self.conv2 = torch.nn.Conv2d(n_patterns1, n_patterns2, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(n_patterns2)

        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_pool, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(n_patterns2, n_patterns3, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = torch.nn.BatchNorm2d(n_patterns3)
        self.conv4 = torch.nn.Conv2d(n_patterns3, n_patterns4, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = torch.nn.BatchNorm2d(n_patterns4)

        self.conv5 = torch.nn.Conv2d(n_patterns4, n_patterns5, kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = torch.nn.BatchNorm2d(n_patterns5)
        self.conv6 = torch.nn.Conv2d(n_patterns5, n_patterns6, kernel_size=3, stride=1, padding=1)
        self.batchnorm6 = torch.nn.BatchNorm2d(n_patterns6)

        self.conv7 = torch.nn.Conv2d(n_patterns6, n_patterns7, kernel_size=3, stride=1, padding=1)
        self.batchnorm7 = torch.nn.BatchNorm2d(n_patterns7)
        self.conv8 = torch.nn.Conv2d(n_patterns7, n_patterns8, kernel_size=3, stride=1, padding=1)
        self.batchnorm8 = torch.nn.BatchNorm2d(n_patterns8)

        self.fc1 = torch.nn.Linear(int(n_patterns8 * size_im[0]/(kernel_pool**4) * size_im[1]/(kernel_pool**4)), 512)

        self.fc2 = torch.nn.Linear(512, 256)

        self.fc3 = torch.nn.Linear(256, 128)

        #  7 output features for our 7 defined classes
        self.fc4 = torch.nn.Linear(128, d_out)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = F.relu(self.batchnorm6(self.conv6(x)))
        x = self.pool(x)

        x = F.relu(self.batchnorm7(self.conv7(x)))
        x = F.relu(self.batchnorm8(self.conv8(x)))
        x = self.pool(x)

        x = x.view(-1, int(self.n_patterns8 * self.size_im[0]/(self.kernel_pool**4) * self.size_im[1]/(self.kernel_pool**4)))

        x = F.relu(self.fc1(self.drop_hidden(x)))

        x = F.relu(self.fc2(self.drop_hidden(x)))

        x = F.relu(self.fc3(self.drop_hidden(x)))

        # Computes the second fully connected layer (activation applied later)
        x = self.fc4(self.drop_visible(x))
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
        return [self.name, self.ch_in, self.d_out, self.size_im , self.n_patterns1,
                self.n_patterns2, self.n_patterns3, self.n_patterns4, self.n_patterns5,
                self.n_patterns6, self.n_patterns7, self.n_patterns8,
                self.kernel_pool]# self.dtype , self.device] I cant save stype because is a torch specific type and I get THE ERROR