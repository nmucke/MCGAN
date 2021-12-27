import pdb
import numpy as np


class transform_data():
    def __init__(self, a=-1, b=1):

        super(transform_data, self).__init__()
        self.a = a
        self.b = b

        max_min_vec = np.load('transforms/max_min_data.npy', allow_pickle=True)
        self.max_vec = max_min_vec.item()['max_vec']
        self.min_vec = max_min_vec.item()['min_vec']

    def min_max_transform(self, data):
        return self.a + (data-self.min_vec)*(self.b-self.a) \
               /(self.max_vec-self.min_vec)

    def min_max_inverse_transform(self, data):
        return (data-self.a)*(self.max_vec-self.min_vec) \
               /(self.b-self.a) + self.min_vec

if __name__ == "__main__":
    data_path = 'test_data/pipe_flow_state_data_'

    data = []
    for i in range(10):
        data_state = np.load(f"{data_path}{i}_npy") 
        
        data.append(data_state)

    data = np.asarray(data)
    max_vec = np.max(data, axis=0)
    min_vec = np.min(data, axis=0)

    np.save('max_min_data_with_leak_small', {'max_vec': max_vec,
                                             'min_vec': min_vec})