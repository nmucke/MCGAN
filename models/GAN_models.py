import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import pdb
from torch.nn.utils import spectral_norm


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class ParameterGeneratorPipeFlowUpsample(nn.Module):
    def __init__(self, latent_dim, parameter_dim, gen_channels, par_neurons):
        super().__init__()

        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        self.par_neurons = par_neurons
        self.channels = gen_channels
        self.channels.append(2)
        self.init_dim = 4

        kernel_size = [3, 3, 3, 3, 3, 3]
        stride = [2, 2, 2, 2, 2, 2]
        padding = [1, 1, 1, 1, 1, 1]
        bias = [True, True, True, True, True, False]

        self.activation = nn.LeakyReLU()

        self.linear_layer1 = nn.Linear(in_features=latent_dim,
                                       out_features=self.channels[0] \
                                                    * self.init_dim *
                                                    self.init_dim)

        self.deconv = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.pad = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.batch_norm.append(nn.BatchNorm2d(self.channels[i]))
            self.upsample.append(nn.Upsample(scale_factor=2,
                                             mode='bilinear',
                                             align_corners=True))
            self.pad.append(nn.ReplicationPad2d(padding=padding[i]))
            self.conv_up.append(nn.Conv2d(self.channels[i],
                                          self.channels[i + 1],
                                          kernel_size=kernel_size[i],
                                          stride=1,
                                          padding=0,
                                          bias=bias[i]))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=par_neurons[-1],
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=par_neurons[-1],
                               out_channels=par_neurons[-2],
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=par_neurons[-2],
                               out_channels=par_neurons[-3],
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)
        self.pars_linear_layer1 = nn.Linear(in_features=par_neurons[-3] * 3 * 3,
                                            out_features=par_neurons[-4])
        self.pars_linear_layer2 = nn.Linear(in_features=par_neurons[-4],
                                            out_features=parameter_dim)

    def forward(self, z):

        x = self.linear_layer1(z)
        x = self.activation(x)
        x = x.view(-1, self.channels[0], self.init_dim, self.init_dim)
        x = self.batch_norm[0](x)

        x = self.upsample[0](x)
        x = self.pad[0](x)
        x = self.conv_up[0](x)
        for i in range(1, len(self.channels) - 1):
            x = self.activation(x)
            x = self.batch_norm[i](x)
            x = self.upsample[i](x)
            x = self.pad[i](x)
            x = self.conv_up[i](x)
        x = self.tanh(x)

        pars = self.conv1(x)
        pars = self.activation(pars)
        pars = self.conv2(pars)
        pars = self.activation(pars)
        pars = self.conv3(pars)
        pars = self.activation(pars)

        pars = pars.view(-1, self.par_neurons[-3] * 3 * 3)
        pars = self.pars_linear_layer1(pars)
        pars = self.activation(pars)
        pars = self.pars_linear_layer2(pars)
        pars = self.sigmoid(pars)

        return x, pars


class ParameterGeneratorPipeFlow(nn.Module):
    def __init__(self, latent_dim, parameter_dim, gen_channels, par_neurons):
        super().__init__()

        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        self.par_neurons = par_neurons
        self.channels = gen_channels
        self.channels.append(2)
        self.init_dim = 3

        kernel_size = [3, 3, 3, 3, 3, 3]
        stride = [2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0]
        output_padding = [0, 0, 0, 0, 0, 1]
        bias = [True, True, True, True, True, False]

        self.activation = nn.LeakyReLU()
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


        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=par_neurons[-1],
                               kernel_size=5,
                               stride=3,
                               padding=0,
                               bias=True)
        self.batch_norm_pars1 = nn.BatchNorm2d(par_neurons[-1])

        self.conv2 = nn.Conv2d(in_channels=par_neurons[-1],
                               out_channels=par_neurons[-2],
                               kernel_size=5,
                               stride=3,
                               padding=0,
                               bias=True)
        self.batch_norm_pars2 = nn.BatchNorm2d(par_neurons[-2])

        self.conv3 = nn.Conv2d(in_channels=par_neurons[-2],
                               out_channels=par_neurons[-3],
                               kernel_size=5,
                               stride=3,
                               padding=0,
                               bias=True)
        self.batch_norm_pars3 = nn.BatchNorm2d(par_neurons[-3])

        self.conv4 = nn.Conv2d(in_channels=par_neurons[-3],
                               out_channels=par_neurons[-4],
                               kernel_size=5,
                               stride=3,
                               padding=0,
                               bias=True)
        self.batch_norm_pars4 = nn.BatchNorm2d(par_neurons[-4])

        self.pars_linear_layer1 = nn.Linear(in_features=par_neurons[-4]*2*2,
                                       out_features=par_neurons[-1])
        #self.pars_linear_layer2 = nn.Linear(in_features=par_neurons[-3],
        #                               out_features=par_neurons[-2])
        #self.pars_linear_layer3 = nn.Linear(in_features=par_neurons[-2],
        #                               out_features=par_neurons[-1])
        self.pars_linear_layer4 = nn.Linear(in_features=par_neurons[-1],
                                       out_features=parameter_dim,
                                        bias=True)
        '''
        self.par_neurons.insert(0,256*2)
        self.par_linear = nn.ModuleList()
        for i in range(len(self.par_neurons) - 1):
            self.par_linear.append(nn.Linear(in_features=par_neurons[i],
                                           out_features=par_neurons[i+1],
                                           bias=True))

        self.par_linear_layer_out = nn.Linear(in_features=par_neurons[-1],
                                           out_features=self.parameter_dim,
                                           bias=False)
        '''
        self.dx = torch.tensor(7.8125)
    def forward(self, z):

        x = self.linear_layer1(z)
        x = self.activation(x)
        x = x.view(-1,self.channels[0],self.init_dim,self.init_dim)
        x = self.batch_norm[0](x)

        x = self.deconv[0](x)
        for i in range(1,len(self.channels)-1):
            x = self.activation(x)
            x = self.batch_norm[i](x)
            x = self.deconv[i](x)
        x[:,:,:,-1] = x[:,:,:,-3].clone() + self.dx.clone()*(x[:,:,:,-2].clone()-x[:,:,:,-3].clone())
        x = self.tanh(x)
        #x[:, :, :, -1] = x[:, :, :, -2]*(2*self.dx*3*self.dx)/(self.dx*2*self.dx) \
        #               + x[:, :, :, -3]*(self.dx*3*self.dx)/(-self.dx*self.dx) \
        #               + x[:, :, :, -4]*(self.dx*2*self.dx)/(2*self.dx*self.dx)

        pars = self.conv1(x)
        pars = self.activation(pars)
        pars = self.batch_norm_pars1(pars)
        pars = self.conv2(pars)
        pars = self.activation(pars)
        pars = self.batch_norm_pars2(pars)
        pars = self.conv3(pars)
        pars = self.activation(pars)
        pars = self.batch_norm_pars3(pars)
        pars = self.conv4(pars)
        pars = self.activation(pars)
        pars = self.batch_norm_pars4(pars)

        pars = pars.view(-1,self.par_neurons[-4]*2*2)
        pars = self.pars_linear_layer1(pars)
        pars = self.activation(pars)
        #pars = self.pars_linear_layer2(pars)
        #pars = self.activation(pars)
        #pars = self.pars_linear_layer3(pars)
        #pars = self.activation(pars)
        pars = self.pars_linear_layer4(pars)
        #pars = self.relu(pars)
        pars = self.sigmoid(pars)
        '''
        pars = self.flatten(x)
        
        for i in range(0,len(self.par_neurons)-1):
            pars = self.par_linear[i](pars)
            pars = self.activation(pars)

        pars = self.par_linear_layer_out(pars)
        pars = self.tanh(pars)
        '''
        return x, pars

class ParameterCriticPipeFlow(nn.Module):
    def __init__(self, critic_channels, parameter_dim, state_neurons,
                 par_neurons, combined_neurons):
        super().__init__()

        self.channels = critic_channels
        self.channels.reverse()
        self.channels.insert(0, 2)
        self.state_neurons = state_neurons
        self.par_neurons = par_neurons
        self.combined_neurons = combined_neurons

        dilation = 1
        padding = 0
        kernel_size = 3
        stride = 2

        self.parameter_dim = parameter_dim

        self.activation = nn.LeakyReLU(0.2)

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
            dim_out = torch.floor(torch.tensor((dim_out + 2 * padding - dilation\
                * (kernel_size - 1)- 1) / stride + 1))
        self.dim_out = int(dim_out)

        self.flatten = nn.Flatten()

        self.state_neurons.insert(0,self.dim_out*self.dim_out*self.channels[-1])
        self.par_neurons.insert(0,1)

        self.combined_neurons.insert(0,self.dim_out*self.dim_out*self.channels[-1]+1)#self.par_neurons[-1]+self.state_neurons[-1])
        self.combined_neurons.append(1)

        self.combined_bias = [True for i in range(len(self.combined_neurons)-1)]
        self.combined_bias.append(False)

        self.state_linear = nn.ModuleList()
        self.par_linear = nn.ModuleList()
        self.combined_linear = nn.ModuleList()

        for i in range(len(self.state_neurons) - 1):
            self.state_linear.append(nn.Linear(in_features=self.state_neurons[i],
                                               out_features=self.state_neurons[i + 1],
                                               bias=True))
        for i in range(len(self.par_neurons) - 1):
            self.par_linear.append(nn.Linear(in_features=self.par_neurons[i],
                                               out_features=self.par_neurons[i + 1],
                                               bias=True))
        for i in range(len(self.combined_neurons) - 1):
            self.combined_linear.append(nn.Linear(in_features=self.combined_neurons[i],
                                             out_features=self.combined_neurons[i + 1],
                                             bias=self.combined_bias[i]))



    def forward(self, x, par):

        for i in range(len(self.channels) - 1):
            x = self.conv[i](x)
            x = self.activation(x)

        x = self.flatten(x)
        #for i in range(0, len(self.state_neurons) - 1):
        #    x = self.state_linear[i](x)
        #    x = self.activation(x)

        #for i in range(0, len(self.par_neurons) - 1):
        #    par = self.par_linear[i](par)
        #    par = self.activation(par)

        out = torch.cat([x,par],dim=1)

        out = self.combined_linear[0](out)
        for i in range(1, len(self.combined_neurons) - 1):
            out = self.activation(out)
            out = self.combined_linear[i](out)
        return out

class ParameterGeneratorPipeFlowOld(nn.Module):
    def __init__(self, latent_dim, parameter_dim, d=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.parameter_dim = parameter_dim
        self.d = d
        self.num_layers = 5
        self.dense_features = d * 2**(self.num_layers+1)

        self.activation = nn.LeakyReLU(0.2)

        self.linear_layer1 = nn.Linear(in_features=latent_dim,
                                      out_features=self.dense_features*4)

        self.deconv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(self.num_layers):
            if i == self.num_layers-1:
                self.deconv.append(nn.ConvTranspose2d(
                    in_channels=d * 2 ** (self.num_layers + 1 - i),
                    out_channels=d * 2 ** (self.num_layers - i),
                    kernel_size=4,
                    stride=2,
                    padding=0,
                    output_padding=1,
                    bias=True))
            else:
                self.deconv.append(nn.ConvTranspose2d(in_channels=d * 2**(
                        self.num_layers+1-i),
                                                      out_channels=d * 2**(self.num_layers-i),
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=0,
                                                      output_padding=0,
                                                      bias=True))
            self.batch_norm.append(nn.BatchNorm2d(d * 2**(self.num_layers-i)))
            print(d * 2**(self.num_layers-i))


        self.deconv_out = nn.ConvTranspose2d(in_channels=d * 2**(self.num_layers-i),
                                             out_channels=2,
                                             kernel_size=4,
                                             stride=2,
                                             padding=0,
                                             output_padding=0,
                                             bias=True)

        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=d,
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=d,
                               out_channels=d*2,
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=d*2,
                               out_channels=d*4,
                               kernel_size=5,
                               stride=4,
                               padding=0,
                               bias=True)
        self.pars_linear_layer1 = nn.Linear(in_features=4*d*3*3,
                                       out_features=d)
        self.pars_linear_layer2 = nn.Linear(in_features=d,
                                       out_features=parameter_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        x = self.linear_layer1(x)
        x = self.activation(x)

        x = x.view(-1,self.dense_features,2,2)

        for _, (deconv,batch_norm) in enumerate(zip(self.deconv,
                                                    self.batch_norm)):

            x = deconv(x)
            x = self.activation(x)
            x = batch_norm(x)

        x = self.deconv_out(x)
        x = self.sigmoid(x)

        pars = self.conv1(x)
        pars = self.activation(pars)
        pars = self.conv2(pars)
        pars = self.activation(pars)
        pars = self.conv3(pars)
        pars = self.activation(pars)

        pars = pars.view(-1,4*self.d*3*3)
        pars = self.pars_linear_layer1(pars)
        pars = self.activation(pars)
        pars = self.pars_linear_layer2(pars)
        pars = self.sigmoid(pars)

        return x, pars

class ParameterCriticPipeFlowOld(nn.Module):
    def __init__(self, d=32, parameter_dim=2):
        super().__init__()

        self.d = d
        self.parameter_dim = parameter_dim
        self.num_layers = 5

        self.activation = nn.LeakyReLU(0.2)

        self.conv_in = nn.Conv2d(in_channels=2,
                                 out_channels=d * 2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=0,
                                 bias=True)

        self.conv = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv.append(nn.Conv2d(in_channels=d * 2**(i+1),
                                       out_channels=d * 2**(i+2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=0,
                                       bias=True))

        self.dense1 = nn.Linear(d * 2**(i+2)* 2 * 2 + parameter_dim, 256, bias=True)
        self.dense2 = nn.Linear(256, 128, bias=True)
        self.dense3 = nn.Linear(128, 64, bias=True)
        self.dense4 = nn.Linear(64, 1, bias=True)
        #self.dense5 = nn.Linear(32, 1, bias=True)

    def forward(self, x, parameters):
        x = self.conv_in(x)
        x = self.activation(x)

        for _, conv in enumerate(self.conv):
            x = conv(x)
            x = self.activation(x)

        x = x.view(-1,self.d * 2**(self.num_layers+1)* 2 * 2)
        x = torch.cat((x,parameters),dim=1)

        x = self.dense1(x)
        x = self.activation(x)

        x = self.dense2(x)
        x = self.activation(x)

        x = self.dense3(x)
        x = self.activation(x)

        return self.dense4(x)


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

