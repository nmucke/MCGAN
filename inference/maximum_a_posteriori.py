import pdb
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import time

def compute_MAP(z, observations, generator, obs_operator, obs_std,
                inverse_transformer_state=None, num_iters=1000):

    optimizer = optim.Adam([z], lr=1e-2)

    #sloss = nn.MSELoss()

    if inverse_transformer_state is not None:

        scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=500,min_lr=0.001)
        with tqdm(range(num_iters), mininterval=3.,postfix=['Loss', dict(loss="0")]) as pbar:
            for epoch in pbar:

                optimizer.zero_grad()
                gen_state, _ = generator(z)
                gen_obs = obs_operator(gen_state)
                gen_obs = inverse_transformer_state(gen_obs,
                                                    velocity_or_pressure='pressure')
                error = 1 / obs_std / obs_std * torch.pow(
                        torch.linalg.norm(observations - gen_obs), 2) \
                        + torch.pow(torch.linalg.norm(z), 2)
                error.backward()
                optimizer.step()

                scheduler.step(error)
                pbar.postfix[1] = f"{error.item():.3f}"

    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=500,min_lr=0.001)
        with tqdm(range(num_iters), mininterval=3.,postfix=['Loss', dict(loss="0")]) as pbar:
            for epoch in pbar:
                optimizer.zero_grad()
                gen_state, _ = generator(z)
                gen_obs = obs_operator(gen_state)
                error = 1 / obs_std / obs_std * torch.pow(
                        torch.linalg.norm(observations - gen_obs), 2) \
                        + torch.pow(torch.linalg.norm(z), 2)
                error.backward()
                optimizer.step()

                scheduler.step(error)
                pbar.postfix[1] = f"{error.item():.3f}"

    return z.detach()