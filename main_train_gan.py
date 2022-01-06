import pdb
import torch.nn as nn
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_state, transform_pars
from utils.seed_everything import seed_everything
from training.training_GAN import TrainParGAN

torch.set_default_dtype(torch.float32)

if __name__ == "__main__":

    seed_everything()

    cuda = True
    if cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    print(f'Training GAN on {device}')

    data_path = 'data/pipe_flow_data_'

    train_WGAN = True
    continue_training = True
    load_string = 'model_weights/GAN'
    save_string = 'model_weights/GAN'

    latent_dim = 50
    activation = nn.LeakyReLU()
    transformer_state = transform_state(a=-1, b=1)
    transformer_pars = transform_pars(a=-1, b=1)

    training_params = {'latent_dim': latent_dim,
                       'n_critic': 2,
                       'gamma': 5,
                       'n_epochs': 1000,
                       'save_string': save_string,
                       'device': device}
    optimizer_params = {'learning_rate': 1e-4}
    dataloader_params = {'data_path': data_path,
                         'num_files': 100000,
                         'transformer_state': transformer_state,
                         'transformer_pars': transformer_pars,
                         'batch_size': 64,
                         'shuffle': True,
                         'num_workers': 8,
                         'drop_last': True}

    generator_params = {'latent_dim': latent_dim,
                        'par_dim': 1,
                        'output_dim': (2, 256, 256),
                        'activation': activation,
                        'gen_channels': [256, 128, 64, 32, 16, 8],
                        'par_neurons': [64, 32, 16, 8]}
    critic_params = {'activation': activation,
                     'critic_channels': [256, 128, 64, 32, 16, 8],
                     'par_dim': 1,
                     'combined_neurons': [64, 32, 16, 8, 4]}

    generator = GAN_models.ParameterGeneratorPipeFlow(**generator_params).to(device)
    critic = GAN_models.ParameterCriticPipeFlow(**critic_params).to(device)

    dataloader = get_dataloader(**dataloader_params)

    if train_WGAN:

        generator_optimizer = torch.optim.RMSprop(generator.parameters(),
                                                  lr=optimizer_params['learning_rate'])
        critic_optimizer = torch.optim.RMSprop(critic.parameters(),
                                               lr=optimizer_params['learning_rate'])

        if continue_training:
            load_checkpoint(load_string, generator, critic,
                            generator_optimizer, critic_optimizer)

        trainer = TrainParGAN(generator=generator,
                              critic=critic,
                              generator_optimizer=generator_optimizer,
                              critic_optimizer=critic_optimizer,
                              **training_params)

        generator_loss, critic_loss, gradient_penalty = trainer.train(
                data_loader=dataloader)