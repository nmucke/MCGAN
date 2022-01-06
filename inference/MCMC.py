import pdb
import torch
import hamiltorch
import matplotlib.pyplot as plt

def latent_posterior(z, generator, obs_operator, observations,
                     prior_mean, prior_std, noise_mean, noise_std,
                     inverse_transformer_state):

    z_prior_score = torch.distributions.Normal(prior_mean,
                                               prior_std).log_prob(z).sum()

    gen_state, _ = generator(z.view(1, len(z)))
    gen_state = obs_operator(gen_state)
    gen_state = inverse_transformer_state(gen_state,
                                          velocity_or_pressure='pressure')
    error = observations - gen_state
    error = error

    reconstruction_score = torch.distributions.Normal(noise_mean,
                                      noise_std).log_prob(error).sum()

    return z_prior_score + reconstruction_score

def hamiltonian_MC(z_init,posterior_params, HMC_params):
    posterior = lambda z: latent_posterior(z, **posterior_params)
    z_samples = hamiltorch.sample(log_prob_func=posterior,
                                  params_init=z_init,
                                  **HMC_params)
    return torch.stack(z_samples)
