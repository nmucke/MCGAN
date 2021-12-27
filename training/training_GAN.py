import torchvision.transforms as transforms
import torch
import pdb
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import imageio
import matplotlib.pyplot as plt
import gc
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

class TrainParGAN():
    def __init__(self, generator, critic, generator_optimizer, critic_optimizer,
                 latent_dim=100, n_critic=5, gamma=10, save_string='GAN',
                 n_epochs=100, device='cpu'):

        self.to_pil_image = transforms.ToPILImage()

        self.device = device
        self.generator = generator
        self.critic = critic
        self.G_opt = generator_optimizer
        self.C_opt = critic_optimizer

        self.n_epochs = n_epochs
        self.save_string = save_string

        self.generator.train(mode=True)
        self.critic.train(mode=True)

        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gamma = gamma
        self.fixed_z = torch.randn(64, self.latent_dim).to(self.device)

    def train(self, data_loader):
        """Train generator and critic"""

        self.num_data_channels = data_loader.dataset[0][0].shape[0]

        images = []
        generator_loss = []
        critic_loss = []
        gradient_penalty = []
        for epoch in range(1, self.n_epochs + 1):

            # Train one step
            g_loss, c_loss, grad_penalty = self.train_epoch(data_loader)

            print(f'Epoch: {epoch}, g_loss: {g_loss:.3f},', end=' ')
            print(f'c_loss: {c_loss:.3f}, grad_penalty: {grad_penalty:.3f}')


            # Save loss
            generator_loss.append(g_loss)
            critic_loss.append(c_loss)
            gradient_penalty.append(grad_penalty)

            # Save generated images
            generated_img, _ = self.generator(self.fixed_z,)
            generated_img = generated_img.to('cpu').detach()
            del _
            generated_img = make_grid(generated_img[:,0:1])
            images.append(generated_img)
            self.save_generator_image(generated_img,f"outputs_GAN/gen_img{epoch}.png")

            # Save generator and critic weights
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'generator_optimizer_state_dict': self.G_opt.state_dict(),
                'critic_optimizer_state_dict': self.C_opt.state_dict(),
                }, self.save_string)

        # save the generated images as GIF file
        imgs = [np.array(self.to_pil_image(img)) for img in images]
        imageio.mimsave('outputs_GAN/generator_images.gif', imgs)

        # Save generator and critic weights

        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'generator_optimizer_state_dict': self.G_opt.state_dict(),
            'critic_optimizer_state_dict': self.C_opt.state_dict(),
            }, self.save_string)

        self.generator.train(mode=False)
        self.critic.train(mode=False)

        return generator_loss, critic_loss, gradient_penalty

    def train_epoch(self, data_loader):
        """Train generator and critic for one epoch"""

        for bidx, (real_data, parameters) in tqdm(enumerate(data_loader),
                 total=int(len(data_loader.dataset)/data_loader.batch_size)):

            current_batch_size = len(real_data)

            real_data = real_data.to(self.device)
            parameters = parameters.to(self.device)

            c_loss, grad_penalty = self.critic_train_step(real_data, parameters)

            if bidx % self.n_critic == 0:
                g_loss = self.generator_train_step(current_batch_size, parameters)

        return g_loss, c_loss, grad_penalty

    def critic_train_step(self, data, parameters):
        """Train critic one step"""

        self.C_opt.zero_grad()
        batch_size = data.size(0)

        generated_data, generated_pars = self.sample(batch_size)

        grad_penalty = self.gradient_penalty(data, generated_data,
                                             parameters, generated_pars)
        c_loss = self.critic(generated_data,generated_pars).mean() \
                 - self.critic(data,parameters).mean() + grad_penalty
        c_loss.backward()
        self.C_opt.step()

        return c_loss.detach().item(), grad_penalty.detach().item()

    def generator_train_step(self, batch_size, parameters):
        """Train generator one step"""

        self.G_opt.zero_grad()
        generated_data, generated_pars = self.sample(batch_size)

        g_loss = - self.critic(generated_data, generated_pars).mean()
        g_loss.backward()
        self.G_opt.step()

        return g_loss.detach().item()

    def gradient_penalty(self, data, generated_data, parameters,
                         generated_pars):
        """Compute gradient penalty"""

        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, self.num_data_channels, 1, 1,
                             device=self.device)
        epsilon = epsilon.expand_as(data)

        epsilon_par = torch.rand(batch_size, 1, device=self.device)
        epsilon_par = epsilon_par.expand_as(parameters)

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data
        interpolation = torch.autograd.Variable(interpolation,
                                                requires_grad=True)

        interpolation_par = epsilon_par * parameters.data + \
                            (1 - epsilon_par) * generated_pars
        interpolation_par = torch.autograd.Variable(interpolation_par,
                                                    requires_grad=True)

        interpolation_critic_score = self.critic(interpolation,
                                                 interpolation_par)

        grad_outputs = torch.ones(interpolation_critic_score.size(),
                                  device=self.device)

        gradients = torch.autograd.grad(outputs=interpolation_critic_score,
                                        inputs=[interpolation,
                                                interpolation_par],
                                        grad_outputs=grad_outputs,
                                        create_graph=True,
                                        retain_graph=True)

        gradients = torch.cat([gradients[0].view(batch_size, -1),
                               gradients[1].view(batch_size, -1)],
                              dim=1)
        gradients_norm = torch.sqrt(
            torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def sample(self, n_samples):
        """Generate n_samples fake samples"""
        return self.generator(torch.randn(n_samples,
                              self.latent_dim).to(self.device))

    def save_generator_image(self, image, path):
        """Save image"""
        save_image(image, path)






