import torch
import torch.nn as nn

# Neural Network
class AnnAutoencoder(nn.Module):
    def __init__(self, name, d_in, H1, d_out, dtype=torch.float, device='cpu'):
        super(AnnAutoencoder, self).__init__()

        self.dtype = dtype
        self.device = device

        self.name = name
        self.d_in = d_in
        self.H1 = H1
        self.d_out = d_out

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, d_out)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d_out, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, d_in),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def backward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_batch(self, x, y, batch_idx, batch_size):
        batch_x = x[batch_idx * batch_size: (batch_idx + 1) * batch_size, :]
        batch_y = y[batch_idx * batch_size: (batch_idx + 1) * batch_size, :]

        return batch_x, batch_y

    def get_args(self):
        return [self.name, self.d_in, self.H1, self.d_out]#, self.dtype, self.device]

    def reshape_data(self, x):
        return x