import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm
import time

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class ParameterGeneratorPipeFlow(nn.Module):
    def __init__(self, latent_dim, par_dim, gen_channels, par_neurons, 
                 output_dim=(2,256), activation=None):
        super().__init__()
        self.t_state = 0
        self.t_pars = 0

        self.latent_dim = latent_dim
        self.par_dim = par_dim
        self.par_neurons = par_neurons
        self.channels = gen_channels
        self.channels.append(2)
        self.init_dim = 3

        kernel_size = [3, 3, 3, 3, 3, 3]
        stride = [2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0]
        output_padding = [0, 0, 0, 0, 0, 1]
        bias = [True, True, True, True, True, False]

        self.activation = activation
        self.relu = nn.ReLU()

        self.linear_layer1 = nn.Linear(in_features=latent_dim,
                                      out_features=self.channels[0]\
                                                   *self.init_dim*self.init_dim)

        self.deconv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(len(self.channels)-1):
            self.batch_norm.append(nn.BatchNorm2d(self.channels[i]))

            self.deconv.append(nn.ConvTranspose2d(in_channels=self.channels[i],
                                        out_channels=self.channels[i+1],
                                        kernel_size=kernel_size[i],
                                        stride=stride[i],
                                        padding=padding[i],
                                        output_padding=output_padding[i],
                                        bias=bias[i]))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.flatten = nn.Flatten()

        self.conv_par_in = nn.Conv2d(in_channels=2,
                                     out_channels=par_neurons[-1],
                                     kernel_size=5,
                                     stride=3,
                                     padding=0,
                                     bias=True)

        self.conv_par = nn.ModuleList()
        self.batch_norm_pars = nn.ModuleList()
        for i in range(1,len(self.par_neurons)):
            self.batch_norm_pars.append(nn.BatchNorm2d(par_neurons[-i]))
            self.conv_par.append(nn.Conv2d(in_channels=par_neurons[-i],
                               out_channels=par_neurons[-i-1],
                               kernel_size=5,
                               stride=3,
                               padding=0,
                               bias=True))
        self.batch_norm_pars_out = nn.BatchNorm2d(par_neurons[-i-1])

        self.pars_linear_layer1 = nn.Linear(in_features=par_neurons[-i-1]*2*2,
                                       out_features=par_neurons[-1])
        self.pars_linear_layer2 = nn.Linear(in_features=par_neurons[-1],
                                            out_features=self.par_dim,
                                            bias=True)
        self.dx = torch.tensor(7.8125)
    def forward(self, z):

        x = self.linear_layer1(z)
        x = self.activation(x)
        x = x.view(-1,self.channels[0],self.init_dim,self.init_dim)
        x = self.batch_norm[0](x)

        x = self.deconv[0](x)
        for (batch_norm, deconv) in zip(self.batch_norm[1:], self.deconv[1:]):
        #for i in range(1,len(self.channels)-1):
            x = self.activation(x)
            #x = self.batch_norm[i](x)
            #x = self.deconv[i](x)
            x = batch_norm(x)
            x = deconv(x)
        x[:,:,:,-1] = x[:,:,:,-3].clone() + \
            self.dx.clone()*(x[:,:,:,-2].clone()-x[:,:,:,-3].clone())
        x = self.tanh(x)

        pars = self.conv_par_in(x)
        pars = self.activation(pars)
        for (batch_norm, conv_par) in zip(self.batch_norm_pars, self.conv_par):
        #for i in range(len(self.par_neurons)-1):
            #pars = self.batch_norm_pars[i](pars)
            #pars = self.conv_par[i](pars)
            pars = batch_norm(pars)
            pars = conv_par(pars)
            pars = self.activation(pars)
        pars = self.batch_norm_pars_out(pars)

        pars = pars.view(-1,self.par_neurons[-4]*2*2)
        pars = self.pars_linear_layer1(pars)
        pars = self.activation(pars)
        pars = self.pars_linear_layer2(pars)
        pars = self.tanh(pars)
        return x, pars

class ParameterCriticPipeFlow(nn.Module):
    def __init__(self, critic_channels, par_dim, combined_neurons,
                activation=None):
        super().__init__()

        self.channels = critic_channels
        self.channels.reverse()
        self.channels.insert(0, 2)
        self.combined_neurons = combined_neurons
        self.par_dim = par_dim

        dilation = 1
        padding = 0
        kernel_size = 3
        stride = 2

        self.activation = activation

        self.conv = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.conv.append(nn.Conv2d(in_channels=self.channels[i],
                                       out_channels=self.channels[i + 1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       bias=True))

        dim_out = 256
        for i in range(len(self.channels) - 1):
            dim_out = np.floor(((dim_out + 2 * padding - dilation\
                * (kernel_size - 1)- 1) / stride + 1))
        self.dim_out = torch.tensor(int(dim_out))

        self.flatten = nn.Flatten()
        self.combined_neurons.insert(0,self.dim_out*self.dim_out*self.channels[-1]+1)
        self.combined_neurons.append(self.par_dim)

        self.combined_bias = [True for i in range(len(self.combined_neurons)-1)]
        self.combined_bias.append(False)

        self.combined_linear = nn.ModuleList()

        for i in range(len(self.combined_neurons) - 1):
            self.combined_linear.append(nn.Linear(in_features=self.combined_neurons[i],
                                             out_features=self.combined_neurons[i + 1],
                                             bias=self.combined_bias[i]))



    def forward(self, x, par):

        for i in range(len(self.channels) - 1):
            x = self.conv[i](x)
            x = self.activation(x)

        x = self.flatten(x)
        out = torch.cat([x,par],dim=1)

        out = self.combined_linear[0](out)
        for i in range(1, len(self.combined_neurons) - 1):
            out = self.activation(out)
            out = self.combined_linear[i](out)
        return out


if __name__ == '__main__':


    def out_size(in_size, stride, padding, kernel_size, out_pad):
        return (in_size - 1) * stride - 2 * padding + (
                    kernel_size - 1) + out_pad + 1

    stride = [2, 2, 2, 2, 2, 2]
    padding = [0, 0, 0, 0, 0, 0]
    kernel_size = [3, 3, 3, 3, 3, 3]
    out_pad = [0, 1, 1, 1, 0, 0]

    in_size = 3
    for i in range(len(stride)):
        in_size = out_size(in_size, stride[i], padding[i], kernel_size[i],
                           out_pad[i])
        print(in_size)

