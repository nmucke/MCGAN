import pdb
import numpy as np
import torch.nn as nn
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_state, transform_pars
from utils.seed_everything import seed_everything
from inference.maximum_a_posteriori import compute_MAP
from inference.MCMC import hamiltonian_MC
import hamiltorch
torch.set_default_dtype(torch.float64)
from utils.compute_statistics import get_statistics_from_latent_samples
from plotting import plot_results
import matplotlib.pyplot as plt

def observation_operator(data, obs_idx):
    if len(data.shape) == 3:
        obs = data[obs_idx]
    elif len(data.shape) == 4:
        obs = data[0]
        obs = obs[obs_idx]
    return obs.flatten()

def add_noise_to_data(obs, noise_mean, noise_std):
    obs_noise = torch.normal(mean=noise_mean,
                             std=noise_std)
    obs += obs_noise
    return obs

if __name__ == "__main__":

    seed_everything()

    cuda = False
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Running on {device}')

    load_string = 'model_weights/GAN'

    data_path = 'data/pipe_flow_data_'

    domain = {'xmin': 0,
              'xmax': 2000,
              'numx': 256}

    latent_dim = 50
    activation = nn.LeakyReLU()
    transformer_state = transform_state(a=-1, b=1)
    transformer_pars = transform_pars(a=-1, b=1)

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 1,
                        'output_dim': (2, 256, 256),
                        'activation': activation,
                        'gen_channels': [128, 64, 32, 16, 8, 4],
                        'par_neurons': [8, 16, 32, 64]}

    generator = GAN_models.ParameterGeneratorPipeFlow(**generator_params).to(device)
    load_checkpoint(load_string, generator)
    generator.eval()

    case = 0
    state_path = data_path + 'state/pipe_flow_state_data_' + str(case) + '.npy'
    pars_path = data_path + 'parameters/pipe_flow_parameter_data_' + str(case) + '.npy'

    true_state = torch.tensor(np.load(state_path, allow_pickle=True))
    true_pars = np.zeros((2))
    load_pars = np.load(pars_path, allow_pickle=True)
    true_pars[0], true_pars[1] = load_pars[0][0], load_pars[1][0]
    true_pars = torch.tensor(true_pars)


    obs_x, obs_y = range(0,256,10), range(0,256,10)
    obs_x, obs_y = np.meshgrid(obs_x, obs_y)
    obs_idx = (0, obs_x, obs_y)
    obs_std = 0.025

    obs_operator = lambda obs: observation_operator(obs, obs_idx)
    observations = obs_operator(true_state).to(device)

    noise_mean = torch.zeros(observations.shape, device=device)
    noise_std = obs_std*torch.ones(observations.shape, device=device)
    observations = add_noise_to_data(observations,
                                     noise_mean,
                                     noise_std)

    z_init = torch.randn(1, latent_dim, requires_grad=True, device=device)
    z_map = compute_MAP(z=z_init,
                        observations=observations,
                        generator=generator,
                        obs_operator=obs_operator,
                        inverse_transformer_state=transformer_state.min_max_inverse_transform,
                        num_iters=1000)

    obs_error = torch.linalg.norm(observations-\
                  obs_operator(generator(z_map)[0][0])) \
                / torch.linalg.norm(observations)
    full_error = torch.linalg.norm(true_state-generator(z_map)[0][0]) \
                / torch.linalg.norm(true_state)
    print(f'Observation error: {obs_error:0.4f}')
    print(f'Full error: {full_error:0.4f}')

    posterior_params = {'generator': generator,
                        'obs_operator': obs_operator,
                        'observations': observations,
                        'prior_mean': torch.zeros(latent_dim, device=device),
                        'prior_std': torch.ones(latent_dim, device=device),
                        'noise_mean': noise_mean,
                        'noise_std': noise_std,
                        'inverse_transformer_state': transformer_state.min_max_inverse_transform}
    HMC_params = {'num_samples': 50,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 25,
                  'integrator': hamiltorch.Integrator.IMPLICIT,
                  'sampler': hamiltorch.Sampler.HMC_NUTS,
                  'desired_accept_rate': 0.3}

    z_samples = hamiltonian_MC(z_init=torch.squeeze(z_map),
                               posterior_params=posterior_params,
                               HMC_params=HMC_params)

    MCGAN_results = \
        get_statistics_from_latent_samples(z_samples=z_samples,
                                           generator=generator,
                                           transformer_state=transformer_state.min_max_inverse_transform,
                                           transformer_pars=transformer_pars.min_max_inverse_transform,
                                           domain=domain)

