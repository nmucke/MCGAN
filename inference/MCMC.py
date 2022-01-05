import pdb
import torch
import hamiltorch

def latent_posterior(z, generator, obs_operator, observations,
                     prior_mean, prior_std, noise_mean, noise_std,
                     inverse_transformer_state):

    z_prior_score = torch.distributions.Normal(prior_mean,
                                               prior_std).log_prob(z).sum()

    gen_state, _ = generator(z.view(1, len(z)))
    gen_measurement = obs_operator(gen_state)
    gen_measurement = inverse_transformer_state(gen_measurement)
    error = observations - gen_measurement
    error = error.detach()
    reconstruction_score = torch.distributions.Normal(noise_mean,
                                      noise_std).log_prob(error).sum()

    return z_prior_score + reconstruction_score

def hamiltonian_MC(z_init,posterior_params, HMC_params):
    posterior = lambda z: latent_posterior(z, **posterior_params)
    z_samples = hamiltorch.sample(log_prob_func=posterior,
                                  params_init=z_init,
                                  **HMC_params)
    return torch.stack(z_samples)
