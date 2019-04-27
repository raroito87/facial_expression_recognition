import torch
from torch.utils.data import Dataset

class Fer2013Dataset(Dataset):
    """ fer2013_DatasetA dataset."""
    def __init__(self, inputs, targets, to_rgb = False, size_im = [48, 48], dtype=torch.float, device='cpu'):

        self.size_im = size_im
        self.dtype = dtype
        self.device = device

        self.x_data = None
        #turn all the images into rgb
        if to_rgb:
            data_grey =  torch.tensor(inputs, device=device, dtype=dtype)
            self.x_data = torch.zeros([data_grey.shape[0], 3, data_grey.shape[1]])
            for i in range(data_grey.shape[0]):
                self.x_data [i] = data_grey[i].unsqueeze(0).repeat(3, 1)
            self.x_data  = self.x_data .reshape(data_grey.shape[0], 3, 224, 224)
        else:
            self.x_data =  torch.tensor(inputs, device=device, dtype=dtype)
            self.x_data = self.x_data.reshape(self.x_data.shape[0], 1, self.size_im[0], self.size_im[1])

        self.y_data = torch.tensor(targets,device=device, dtype=torch.long).squeeze()

        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
