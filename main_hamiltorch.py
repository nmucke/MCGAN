import pdb
import numpy as np
import torch.nn as nn
import torch
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_state, transform_pars
from utils.seed_everything import seed_everything
from inference.maximum_a_posteriori import compute_MAP
from inference.MCMC import hamiltonian_MC
import hamiltorch
torch.set_default_dtype(torch.float32)
from utils.compute_statistics import get_statistics_from_latent_samples
from plotting import plot_results
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

def observation_operator(data, obs_idx):
    if len(data.shape) == 3:
        obs = data[obs_idx]
    elif len(data.shape) == 4:
        obs = data[0]
        obs = obs[obs_idx]
    return obs

def add_noise_to_data(obs, noise_mean, noise_std):
    obs_noise = torch.normal(mean=noise_mean,
                             std=noise_std)
    obs += obs_noise
    return obs

if __name__ == "__main__":

    seed_everything()

    cuda = True
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Running on {device}')

    load_string = 'model_weights/GAN'

    data_path = 'data/pipe_flow_data_'

    domain = {'xmin': 0,
              'xmax': 2000,
              'numx': 256,
              'tmin': 0,
              'tmax': 1.1*60,
              'numt': 256}

    latent_dim = 50
    activation = nn.LeakyReLU()
    transformer_state = transform_state(a=-1, b=1, device=device)
    transformer_pars = transform_pars(a=-1, b=1)

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 1,
                        'output_dim': (2, 256, 256),
                        'activation': activation,
                        'gen_channels': [256, 128, 64, 32, 16, 8],
                        'par_neurons': [64, 32, 16, 8]}

    generator = GAN_models.ParameterGeneratorPipeFlow(**generator_params)
    load_checkpoint(load_string, generator)
    generator = generator.to(device)
    generator.eval()

    case = 0
    state_path = data_path + 'state/pipe_flow_state_data_' + str(case) + '.npy'
    pars_path = data_path + 'parameters/pipe_flow_parameter_data_' + str(case) + '.npy'

    true_state = torch.tensor(np.load(state_path, allow_pickle=True), device=device)

    true_pars = np.zeros((2))
    load_pars = np.load(pars_path, allow_pickle=True)
    true_pars[0], true_pars[1] = load_pars[0][0], load_pars[1][0]
    true_pars = torch.tensor(true_pars)


    obs_t, obs_x =  range(0,256), [10, 246]
    obs_t, obs_x = np.meshgrid(obs_t, obs_x)
    obs_idx = [1, obs_t, obs_x]
    obs_std = 2.5e3

    obs_operator = lambda obs: observation_operator(obs, obs_idx)
    observations = obs_operator(true_state).to(device)

    noise_mean = torch.zeros(observations.shape, device=device)
    noise_std = obs_std*torch.ones(observations.shape, device=device)
    observations = add_noise_to_data(observations, noise_mean, noise_std)

    z_init = torch.randn(1, latent_dim, requires_grad=True, device=device)
    z_map = compute_MAP(z=z_init,
                        observations=observations,
                        generator=generator,
                        obs_operator=obs_operator,
                        obs_std=obs_std,
                        inverse_transformer_state=transformer_state.min_max_inverse_transform,
                        num_iters=5000)

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
    HMC_params = {'num_samples': 5000,
                  'step_size': 1.,
                  'num_steps_per_sample': 5,
                  'burn': 3500,
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

    plot_results.plot_contours(MCGAN_results, true_state, domain,
                               pressure_or_velocity='velocity')

    plot_results.plot_state(MCGAN_results,true_state, domain,
                           time_plot_ids=[100,200,253],
                           pressure_or_velocity='velocity',
                           save_string='MCGAN_state_plots.pdf')

    plot_results.plot_par_histograms(MCGAN_results, true_pars)
