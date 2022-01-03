import pdb
import numpy as np
import torch

class transform_state():
    def __init__(self, a=-1, b=1):
        super(transform_state, self).__init__()
        self.a = a
        self.b = b

        max_min_vec = np.load('transforms/max_min_data.npy', allow_pickle=True)
        self.max_vec = max_min_vec.item()['max_vec']
        self.min_vec = max_min_vec.item()['min_vec']

    def min_max_transform(self, data):

        transformed_state = np.zeros(data.shape)
        if len(data.shape) == 2:
            transformed_state = self.a + (data-self.min_vec[0])\
                                *(self.b-self.a)/(self.max_vec[0]-self.min_vec[0])

        elif len(data.shape) == 3:
            transformed_state[0] = self.a + (data[0] - self.min_vec[0]) * (self.b - self.a) \
                                   /(self.max_vec[0] - self.min_vec[0])
            transformed_state[1] = self.a + (data[1]-self.min_vec[1])*(self.b-self.a) \
                                     /(self.max_vec[1]-self.min_vec[1])

        elif len(data.shape) == 4:
            transformed_state[:, 0] = self.a + (data[:, 0] - self.min_vec[0]) * (self.b - self.a) \
                                   / (self.max_vec[0] - self.min_vec[0])
            transformed_state[:, 1] = self.a + (data[:, 1] - self.min_vec[1]) * (self.b - self.a) \
                                   / (self.max_vec[1] - self.min_vec[1])

        return transformed_state

    def min_max_inverse_transform(self, data):

        transformed_state = torch.zeros(data.shape)
        if len(data.shape) == 2:
            transformed_state = (data-self.a)*(self.max_vec[0]-self.min_vec[0]) \
                  /(self.b-self.a) + self.min_vec[0]

        elif len(data.shape) == 3:
            transformed_state[0] = (data[0]-self.a)*(self.max_vec[0]-self.min_vec[0]) \
                                    /(self.b-self.a) + self.min_vec[0]
            transformed_state[1] = (data[1]-self.a)*(self.max_vec[1]-self.min_vec[1]) \
                                    /(self.b-self.a) + self.min_vec[1]

        elif len(data.shape) == 4:
            transformed_state[:, 0] = (data[:, 0]-self.a)*(self.max_vec[0]-self.min_vec[0]) \
                                        /(self.b-self.a) + self.min_vec[0]
            transformed_state[:, 1] = (data[:, 1]-self.a)*(self.max_vec[1]-self.min_vec[1]) \
                                        /(self.b-self.a) + self.min_vec[1]

        return transformed_state

class transform_pars():
    def __init__(self, a=-1, b=1):
        super(transform_pars, self).__init__()
        self.a = a
        self.b = b

        self.max_vec = 9e-4
        self.min_vec = 1e-4

    def min_max_transform(self, data):
        return self.a + (data-self.min_vec)*(self.b-self.a) \
               /(self.max_vec-self.min_vec)

    def min_max_inverse_transform(self, data):
        return (data-self.a)*(self.max_vec-self.min_vec) \
               /(self.b-self.a) + self.min_vec

if __name__ == "__main__":
    data_path = '../data/pipe_flow_data_state/pipe_flow_state_data_'

    data = []
    max_vec = []
    min_vec = []
    for i in range(100000):
        data_state = np.load(f"{data_path}{i}.npy")
        max_vec.append(np.max(data_state, axis=(1, 2)))
        min_vec.append(np.min(data_state, axis=(1, 2)))
        #data.append(data_state)
        if i % 1000 == 0:
            print(i)
    max_vec = np.transpose(max_vec)
    min_vec = np.transpose(min_vec)

    max_vec = np.max(max_vec, axis=1)
    min_vec = np.min(min_vec, axis=1)

    #data = np.asarray(data)
    #max_vec = np.max(data, axis=(0,2,3))
    #min_vec = np.min(data, axis=(0,2,3))

    np.save('max_min_data', {'max_vec': max_vec,
                             'min_vec': min_vec})