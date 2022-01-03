import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


def compute_MAP(z, observations, generator, obs_operator,
                inverse_transformer_state=None, num_iters=1000):

    optimizer = optim.Adam([z], lr=1e-2, weight_decay=1.)

    loss = nn.MSELoss()

    if inverse_transformer_state is not None:

        scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=500,min_lr=0.001)
        with tqdm(range(num_iters), mininterval=3.,postfix=['Loss', dict(loss="0")]) as pbar:
            for epoch in pbar:
                optimizer.zero_grad()
                gen_obs = inverse_transformer_state(generator(z)[0])
                gen_obs = obs_operator(gen_obs)
                error = loss(observations, gen_obs)
                error.backward()
                optimizer.step()

                scheduler.step(error)
                pbar.postfix[1] = f"{error.item():.3f}"
    else:

        scheduler = ReduceLROnPlateau(optimizer, factor=0.9, patience=500,min_lr=0.001)
        with tqdm(range(num_iters), mininterval=3.,postfix=['Loss', dict(loss="0")]) as pbar:
            for epoch in pbar:
                optimizer.zero_grad()
                error = loss(observations, obs_operator(generator(z)[0]))
                error.backward()
                optimizer.step()

                scheduler.step(error)
                pbar.postfix[1] = f"{error.item():.3f}"

    return z.detach()