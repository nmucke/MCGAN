import pdb
import numpy as np
import torch



class PipeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_files=10,
                 transformer_state=None, transformer_pars=None):

        self.data_path_state = data_path + 'state/pipe_flow_state_data_'
        self.data_path_pars = data_path + 'parameters/pipe_flow_parameter_data_'
        self.num_files = num_files
        self.transformer_state = transformer_state
        self.transformer_pars = transformer_pars

        self.state_IDs = [i for i in range(self.num_files)]

        if self.transformer_state is not None:
            self.transformer_state = transformer_state

        if self.transformer_pars is not None:
            self.transformer_pars = transformer_pars

    def transform_state(self, data):
        return self.transformer_state.min_max_transform(data)

    def inverse_transform_state(self, data):
        return self.transformer_state.min_max_inverse_transform(data)

    def transform_pars(self, data):
        return self.transformer_pars.min_max_transform(data)

    def inverse_transform_pars(self, data):
        return self.transformer_pars.min_max_inverse_transform(data)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        state = np.load(f"{self.data_path_state}{idx}.npy", allow_pickle=True)
        state = torch.tensor(state).float()
        if self.transformer_state is not None:
            state = self.transform_state(state)

        pars = np.load(f"{self.data_path_pars}{idx}.npy", allow_pickle=True)
        pars = np.asarray([pars[1:2][0][0]])
        if self.transformer_pars is not None:
            pars = self.transform_pars(pars)
        pars = torch.tensor(pars).float()
        return state, pars

def get_dataloader(data_path,
                    num_files=100000,
                    transformer_state=None,
                    transformer_pars=None,
                    batch_size=512,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                    ):

    dataset = PipeDataset(data_path=data_path,
                          num_files=num_files,
                          transformer_state=transformer_state,
                          transformer_pars=transformer_pars)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             drop_last=drop_last)

    return dataloader