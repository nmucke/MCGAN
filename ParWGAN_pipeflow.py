import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
# import data_loading
import GAN_models
import os

from training_WGAN_GP import TrainParGAN
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

'''
pars = np.load('pipe_flow_parameter_data_simple.npy')
pars_new = np.load('pipe_flow_parameter_data.npy')
pdb.set_trace()

state = []
for i in range(10000):
    state.append(np.load('pipe_flow_data/pipe_flow_state_data_'+str(i)+'.npy'))
    if np.min(state[-1][:, :, 0]) < 5:
        print(np.min(state[-1][:, :, 0]))
state = np.asarray(state)
'''
'''
data_string = '../../../reduced_order_modeling/DG_module/' \
                               'pipe_flow_data_friction/' \
                               'pipe_flow_state_data_'

vel_min = []
vel_max = []
pres_min = []
pres_max = []
for i in range(100000):
    sol = np.load(data_string+str(i) + '.npy')
    vel_min.append(np.min(sol[0]))
    vel_max.append(np.max(sol[0]))

    pres_min.append(np.min(sol[1]))
    pres_max.append(np.max(sol[1]))

print(f'Velocity min: {np.min(vel_min)}')
print(f'Velocity max: {np.max(vel_max)}')

print(f'pressure min: {np.min(pres_min)}')
print(f'pressure max: {np.min(pres_max)}')
#sol = np.asarray(sol)
#print(np.max(sol[:,0]))
#print(np.min(sol[:,0]))
#print(np.max(sol[:,1]))
#print(np.min(sol[:,1]))
pdb.set_trace()
'''

class PipeFlowDataset(torch.utils.data.Dataset):

    def __init__(self,):


        #self.data_path_state = '../../../reduced_order_modeling/DG_module/old_data/' \
        #                       'pipe_flow_data_leak_time/' \
        #                       'pipe_flow_state_data_'

        #self.data_path_pars = '../../../reduced_order_modeling/DG_module/old_data/' \
        #                      'pipe_flow_parameters_leak_time/' \
        #                      'pipe_flow_parameter_data_'

        self.data_path_state = '../../../reduced_order_modeling/DG_module/' \
                               'pipe_flow_data_friction/' \
                               'pipe_flow_state_data_'

        self.data_path_pars = '../../../reduced_order_modeling/DG_module/' \
                              'pipe_flow_parameters_friction/' \
                              'pipe_flow_parameter_data_'

        self.state_IDs = [i for i in range(100000)]


        #self.transform = transforms.Normalize((0.5,0.5), (0.5,0.5))

    def transform_state(self,data):
        normalized_data = np.zeros(data.shape)

        upper = 1
        lower = -1

        #mins = np.array([105.26344713524894,2.0930417842491322, 4989224.769038255])
        #maxs = np.array([106.07579313302448,4.554016883622388,5066287.159771234])

        #normalized_data[0] = (data[0] - mins[1]) / \
        #                    (maxs[1] - mins[1])
        #normalized_data[1] = (data[1] - mins[2]) / \
        #                    (maxs[2] - mins[2])
        velocity_min = 2.232358432009038
        velocity_max = 4.665898915031297
        pressure_min = 4995123.731936767
        pressure_max = 5051183.281894633

        #mins = np.array([2.0930417842491322, 4989224.769038255])
        #maxs = np.array([4.554016883622388, 5066287.159771234])
        mins = np.array([velocity_min, pressure_min])
        maxs = np.array([velocity_max, pressure_max])

        normalized_data[0] = (data[0] - mins[0]) / \
                            (maxs[0] - mins[0]) * (upper-lower) + lower
        normalized_data[1] = (data[1] - mins[1]) / \
                            (maxs[1] - mins[1]) * (upper-lower) + lower

        return normalized_data

    def transform_pars(self,pars):
        normalized_pars = np.zeros(pars.shape)

        upper = 1
        lower = 0

        par_min_size = 1e-4
        par_max_size = 9e-4

        #par_min_loc = 100
        #par_max_loc = 1900

        #normalized_pars[0] = (pars[0][0] - par_min_loc) / \
        #                        (par_max_loc - par_min_loc)
        normalized_pars[0] = (pars[0] - par_min_size) / \
                                (par_max_size - par_min_size) \
                             * (upper-lower) + lower

        return normalized_pars

    def __len__(self):
        return len(self.state_IDs)

    def __getitem__(self, idx):
        parameters = np.load(self.data_path_pars + str(self.state_IDs[idx]) + '.npy'
                             ,allow_pickle=True)
        parameters = np.array([parameters[1:2][0][0]])
        parameters = self.transform_pars(parameters)
        parameters = torch.tensor(parameters).float()

        state = np.load(self.data_path_state + str(self.state_IDs[idx]) + '.npy'
                        ,allow_pickle=True)
        #state = state[1:3,0:256]
        state = self.transform_state(state)
        state = torch.tensor(state).float()

        return state, parameters

def weights_init_he(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

train_WGAN = True
continue_training = True
load_string = 'par_GAN_pipeflow_extraoplate_friction_latent_50'
save_string = 'par_GAN_pipeflow_extraoplate_friction_latent_50'

learning_rate = 1e-4
batch_size = 64
latent_dim = 50
gamma = 5
n_critic = 2
n_epochs = 500
parameter_dim = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

# Number of workers
num_workers = 8

train_dataset = PipeFlowDataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           drop_last=True)

gen_channels = [512, 256, 128, 64, 32, 16]
par_neurons = [128,64,32,16,8]
#par_neurons = [128,64,32,16]
generator = GAN_models.ParameterGeneratorPipeFlow(latent_dim=latent_dim,
                                                 parameter_dim=parameter_dim,
                                                 gen_channels=gen_channels,
                                                 par_neurons=par_neurons).to(device)
critic_channels = [512, 256, 128, 64, 32, 16]
state_neurons = [128]
par_neurons = [8]
combined_neurons = [128, 64, 32, 16]
#state_neurons = [128, 64, 32]
#par_neurons = [32, 32, 32]
#combined_neurons = [256, 128, 64]
critic = GAN_models.ParameterCriticPipeFlow(parameter_dim=parameter_dim,
                                            critic_channels=critic_channels,
                                            state_neurons=state_neurons,
                                            par_neurons=par_neurons,
                                            combined_neurons=combined_neurons
                                            ).to(device)
generator.apply(weights_init_he)
critic.apply(weights_init_he)

print(f'Generator weights: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}')
print(f'Critic weights: {sum(p.numel() for p in critic.parameters() if p.requires_grad)}')

if train_WGAN:

    generator_optimizer = torch.optim.RMSprop(generator.parameters(),
                                              lr=learning_rate)
    critic_optimizer = torch.optim.RMSprop(critic.parameters(),
                                           lr=learning_rate)

    if continue_training:
        checkpoint = torch.load('model_weights/' + load_string)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.to(device)

        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.to(device)

        generator_optimizer.load_state_dict(
                checkpoint['generator_optimizer_state_dict'])

        critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])

    WGAN = TrainParGAN(generator=generator,
                       critic=critic,
                       generator_optimizer=generator_optimizer,
                       critic_optimizer=critic_optimizer,
                       latent_dim=latent_dim,
                       n_critic=n_critic,
                       gamma=gamma,
                       device=device)

    generator_loss, critic_loss, gradient_penalty = WGAN.train(
                                                    data_loader=train_loader,
                                                    n_epochs=n_epochs,
                                                    save_string=save_string)

    plt.figure()
    plt.plot(generator_loss, linewidth=1.6, label='Generator Loss')
    plt.plot(critic_loss, linewidth=1.6, label='Critic Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

else:
    checkpoint = torch.load('model_weights/' + load_string)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()

    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.to(device)
    critic.eval()

