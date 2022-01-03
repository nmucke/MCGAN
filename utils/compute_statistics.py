import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from scipy.ndimage.filters import convolve

def compute_edge_location(state, domain):
    x_vec = np.linspace(domain['xmin'], domain['xmax'], domain['numx'])

    k3 = np.array([[0, 0, 0], [-5, 0, 5], [0, 0, 0]])
    res = convolve(state, k3)
    res = np.sum(res, axis=0)

    return x_vec[res.argmax()]

def get_statistics_from_latent_samples(z_samples,
                                       generator,
                                       transformer_state=None,
                                       transformer_pars=None,
                                       domain=None
                                       ):
    sample_batch_size = 64
    z_loader = torch.utils.data.DataLoader(z_samples,
                                           batch_size=sample_batch_size,
                                           shuffle=False,
                                           drop_last=False)

    gen_states = torch.zeros((len(z_samples), 2, 256, 256))
    gen_pars = torch.zeros((len(z_samples), 2))
    for idx, z in enumerate(z_loader):
        edge_estimate = []

        generated_state, generated_pars = generator(z)
        generated_state = generated_state.cpu().detach()

        gen_states[idx * sample_batch_size:(idx * sample_batch_size + len(z))]\
                    = generated_state
        for i in range(len(z)):
            edge_estimate.append(compute_edge_location(generated_state[i, 0, :, 0:-10],
                                                       domain))

        gen_pars[idx * sample_batch_size:(idx * sample_batch_size + len(z)),1] \
                    = generated_pars.cpu().detach()[:,0]
        gen_pars[idx * sample_batch_size:(idx * sample_batch_size + len(z)), 0] \
                    = torch.tensor(edge_estimate)

    if transformer_state is not None:
        gen_states = torch.tensor(transformer_state(gen_states.detach()))

    if transformer_pars is not None:
        gen_pars[:, 1] = torch.tensor(transformer_pars(gen_pars.numpy()[:, 1]))

    return {'gen_states': gen_states,
            'gen_pars': gen_pars,
            'state_mean': torch.mean(gen_states, dim=0),
            'par_mean': torch.mean(gen_pars, dim=0),
            'state_std': torch.std(gen_states, dim=0),
            'par_std': torch.std(gen_pars, dim=0)}