import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch



class NetworkDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, num_files=10, transformer=None):

        self.data_path_state = data_path
        self.num_files = num_files
        self.transformer = transformer

        self.state_IDs = [i for i in range(self.num_files)]

        if self.transformer is not None:
            self.transform = transformer

    def transform_state(self, data):
        return self.transform.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transform.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        data = np.load(f"{self.data_path_state}{idx}")
        if self.transformer is not None:
            data = self.transform_state(data)
        pars = 1.
        return data, pars

def get_dataloader(data_path,
                    num_files=100000,
                    transformer=None,
                    batch_size=512,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                    ):

    dataset = NetworkDataset(data_path=data_path,
                             num_files=num_files,
                             transformer=transformer)
    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               drop_last=drop_last)

    return dataloader