import numpy as np
import matplotlib.pyplot as plt
import torch
import hamiltorch
import models.GAN_models as GAN_models
import pdb
import matplotlib.patches as patches
from scipy.ndimage.filters import convolve
from scipy.signal import find_peaks, peak_widths, peak_prominences, savgol_filter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import kde
import time
from torch.autograd.gradcheck import zero_gradients

torch.manual_seed(0)
class TransformData():
    def __init__(self):

        self.size_min = 1e-4
        self.size_max = 9e-4

        self.u_min = 2.232358432009038
        self.u_max = 4.665898915031297
        self.pressure_min = 4995123.731936767
        self.pressure_max = 5051183.281894633

        #self.u_min = 2.0930417842491322
        #self.u_max = 4.554016883622388
        #self.pressure_min = 4989224.769038255
        #self.pressure_max = 5066287.159771234

        self.upper = 1
        self.lower = -1

        self.upper_pars = 1
        self.lower_pars = 0

    def transform_state(self, state):
        if state.shape[0]>2:
            state[:,0] = (state[:,0] - self.u_min ) / \
                                (self.u_max - self.u_min) \
                       * (self.upper-self.lower) + self.lower
            state[:,1] = (state[:,1] - self.pressure_min) / \
                                (self.pressure_max - self.pressure_min)\
                       * (self.upper-self.lower) + self.lower

        elif state.shape[0]==1:
            #state[0] = (state[0] - self.u_min ) / \
            #                    (self.u_max - self.u_min) \
            #           * (self.upper-self.lower) + self.lower

            state[0] = (state[0] - self.pressure_min) / \
                       (self.pressure_max - self.pressure_min) \
                       * (self.upper - self.lower) + self.lower

        else:
            state[0] = (state[0] - self.u_min ) / \
                                (self.u_max - self.u_min) \
                       * (self.upper-self.lower) + self.lower
            state[1] = (state[1] - self.pressure_min) / \
                                (self.pressure_max - self.pressure_min)\
                       * (self.upper-self.lower) + self.lower
        return state

    def inverse_transform_state(self, state):
        if state.shape[0]>2:
            state[:, 0] = (state[:, 0] - self.lower)/(self.upper - self.lower) \
                          * (self.u_max - self.u_min) + self.u_min

            state[:, 1] = (state[:, 1] - self.lower)/(self.upper - self.lower) \
                          * (self.pressure_max - self.pressure_min) + self.pressure_min
        elif state.shape[0]==1:
            state[0] = (state[0] - self.lower)/(self.upper - self.lower) \
                          * (self.pressure_max - self.pressure_min) + self.pressure_min
        else:
            state[0] = (state[0] - self.lower)/(self.upper - self.lower) \
                          * (self.u_max - self.u_min) + self.u_min

            state[1] = (state[1] - self.lower)/(self.upper - self.lower) \
                          * (self.pressure_max - self.pressure_min) + self.pressure_min
        return state

    def transform_pars(self,pars):
        pars = (pars - self.size_min ) / \
                (self.size_max - self.size_min)\
           * (self.upper_pars-self.lower_pars) + self.lower_pars
        return pars

    def inverse_transform_pars(self, pars):
        pars = (pars - self.lower_pars)/(self.upper_pars - self.lower_pars) \
              * (self.size_max - self.size_min) + self.size_min
        return pars

transform = TransformData()


xmax = 2000
xmin = 0
num_x = 256
num_t = 256

x_vec = np.linspace(xmin, xmax, num_x)
t_vec = np.linspace(0, 1.1*60, num_t)
X, T = np.meshgrid(x_vec, t_vec)
X_edge, T = np.meshgrid(x_vec[0:-10], t_vec)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)

##### Load Generator #####
latent_dim = 50
gen_channels = [512, 256, 128, 64, 32, 16]
par_neurons = [128,64,32,16,8]
#par_neurons = [128,64,32,16]

checkpoint = torch.load(f'model_weights/par_GAN_pipeflow_extraoplate_friction_latent_50')
generator = GAN_models.ParameterGeneratorPipeFlow(latent_dim=latent_dim,
                                                 parameter_dim=1,
                                                 gen_channels=gen_channels,
                                                 par_neurons=par_neurons).to(device)

generator.load_state_dict(checkpoint['generator_state_dict'])
generator.to(device)
generator.eval()




##### Load image and compute measurements #####

obs_x = [10, 246]
def closest(num, arr):
    curr = arr[0]
    for val in arr:
        if abs (num - val) < abs (num - curr):
            curr = val
    return curr

def compute_edge_location(X):
    k3 = np.array([[0, 0, 0], [-5, 0, 5], [0, 0, 0]])
    res = convolve(X, k3)
    res = np.sum(res, axis=0)
    return x_vec[res.argmax()]

def image_to_measurement(image):
    measurements = image[1:2,:,obs_x]
    return measurements

test_string = '../../../reduced_order_modeling/DG_module/' \
                               'test_data_friction/' \
                               'pipe_flow_state_data_'
test_string_pars = '../../../reduced_order_modeling/DG_module/' \
                               'test_data_pars_friction/' \
                               'pipe_flow_parameter_data_'

test_idx = 6
x = np.load(test_string + str(test_idx) + '.npy', allow_pickle=True)
pars = np.load(test_string_pars + str(test_idx) + '.npy', allow_pickle=True)
pars = np.array([pars[0][0],pars[1][0]])

x = torch.tensor(x).float()
pars = torch.tensor(pars).float()

plt.figure()
plt.imshow(1e-5*np.transpose(x[1].detach().numpy()),origin='lower', extent=[0,66,0,2000],
           vmin=torch.min(1e-5*x[1]), vmax=torch.max(1e-5*x[1]), aspect="auto")
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.savefig('presssure.pdf')
'''
pref = 0
rhoref = 52.67
diameter = 0.508
area = np.pi*diameter*diameter/4
velocity = 308.
pamb = 101325

dt = t_vec[1]-t_vec[0]
dx = x_vec[1]-x_vec[0]

time = 140

x_leak_index = np.argwhere(x_vec == closest(pars[0], x_vec))
pres_drop = np.abs(x[1,time,x_leak_index]-x[1,time,x_leak_index-2])
rho_drop_at_leak = (pres_drop-pref)/velocity/velocity + rhoref
pressureL = x[1,time,x_leak_index]
rhoL = (pressureL-pref)/velocity/velocity + rhoref

rho = (x[1]-pref)/velocity/velocity + rhoref
flow_rate = x[0,time,x_leak_index]*area
#C_d = flow_rate/area/np.sqrt(2*rhoL*pres_drop)
C_d = flow_rate/rhoL/area/x[0,time,x_leak_index]
#np.sqrt((pressureL - pamb) * rhoL)
print(C_d)

q2 = x[0]*area*rho
q2_x = (q2[time,x_leak_index+1]-q2[time,x_leak_index-1])/2/dx
q1_t = (rho[time+1,x_leak_index]*area-x[0,time-1,x_leak_index]*area)/2/dt
rhs = np.sqrt((pressureL - pamb) * rhoL)
C_d = (q1_t + q2_x)/rhs
print(C_d)
pdb.set_trace()
'''

#x_measurement = transform.transform_state(x_measurement)

#pars[1] = transform.transform_pars(pars[1])


x_measurement = image_to_measurement(x)
std_n = 1500
noise_mean = torch.zeros(x_measurement.shape)
noise_std = std_n * torch.ones(x_measurement.shape)
noise = torch.normal(noise_mean, noise_std)
x_measurement = x_measurement + noise
x_measurement = x_measurement.to(device)
noise_mean = noise_mean.to(device)
noise_std = noise_std.to(device)

#x_measurement = transform.transform_state(x_measurement)

x_measurement_no_noise = image_to_measurement(x)
#x_measurement_no_noise = transform.inverse_transform_state(x_measurement_no_noise)

'''
x_measurement = transform.inverse_transform_state(x_measurement)

'''
plt.figure(figsize=(12,4))
plt.plot(t_vec,1e-5*x_measurement[0,:,0].cpu(),linewidth=1.2,color='tab:blue')
plt.plot(t_vec,1e-5*x_measurement_no_noise[0,:,0].cpu(),linewidth=2.,color='tab:green')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.savefig('pressure_measurement_left.pdf')

plt.figure(figsize=(12,4))
plt.plot(t_vec,1e-5*x_measurement[0,:,1].cpu(),linewidth=1.2,color='tab:blue')
plt.plot(t_vec,1e-5*x_measurement_no_noise[0,:,1].cpu(),linewidth=2.,color='tab:green')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.savefig('pressure_measurement_right.pdf')
plt.show()

#pdb.set_trace()
x = transform.transform_state(x)

z_ML = torch.randn(1, latent_dim, requires_grad=True,device=device).float()

true_pars = transform.transform_pars(pars)
loss_list = []
t1 = time.time()
err = 1e8
a = 2.
std_n = a*std_n
for i in range(2):
    z_ML = torch.randn(1, latent_dim, requires_grad=True, device=device).float()

    epochs = 5000
    optim = torch.optim.Adam([z_ML], lr=.1)
    scheduler = ReduceLROnPlateau(optim, factor=0.9, patience=500, min_lr=0.001)
    with tqdm(range(epochs), mininterval=3., postfix=['Loss', dict(loss="0")]) as pbar:
        for epoch in pbar:
            optim.zero_grad()
            gen, gen_pars = generator(z_ML)
            y_pred = image_to_measurement(gen[0])
            y_pred = transform.inverse_transform_state(y_pred)
            #loss = torch.mean(torch.pow(y_pred-x_measurement,2))
            loss = 1 / std_n / std_n * torch.pow(torch.linalg.norm(x_measurement - y_pred), 2) \
                    + torch.pow(torch.linalg.norm(z_ML), 2)
            loss.backward()
            optim.step()

            if epoch % 1000 == 0:
                x_gen, par_gen = generator(z_ML)
                #print(f'{transform.inverse_transform_pars(par_gen[0,0].cpu().detach().numpy()):.5f}')

            scheduler.step(loss)

            pbar.postfix[1] = f"{loss.item():.3f}"

            loss_list.append(loss.item())

    x_gen, par_gen = generator(z_ML)
    y_pred = image_to_measurement(gen[0])
    y_pred = transform.inverse_transform_state(y_pred)
    RMSE = torch.norm(x_measurement - y_pred)
    print(RMSE)

    if RMSE < err:
        z_opt = z_ML.clone().detach()
        err = RMSE

print(time.time() - t1)
x_gen, par_gen = generator(z_opt)
x_gen = x_gen[:, 0:1].cpu().detach().numpy()
x = x.cpu()


MAP_estimate_location = compute_edge_location(x_gen[0,0,:,0:-10])
MAP_estimate_size = transform.inverse_transform_pars(par_gen[0,0].cpu().detach().numpy())
#MAP_estimate_size = par_gen[0,0].cpu().detach().numpy()


print(f'MAP size: {MAP_estimate_size:0.5f}')
print(f'True size: {pars[1]:0.5f}')

print(f'MAP location: {MAP_estimate_location:0.5f}')
print(f'True location: {pars[0]:0.5f}')


plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.transpose(x[0].detach().numpy()),origin='lower', extent=[0,66,0,2000],
           vmin=torch.min(x[0]), vmax=torch.max(x[0]), aspect="auto")
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()

RMSE = np.linalg.norm(x_gen[0,0] - x[0].detach().numpy())/np.linalg.norm(x[0].detach().numpy())
plt.subplot(2, 2, 2)
plt.imshow(np.transpose(x_gen[0,0]),origin='lower', extent=[0,66,0,2000],
           vmin=torch.min(x[0]), vmax=torch.max(x[0]), aspect="auto")
plt.title(f'Relative RMSE: {RMSE:0.2f}',fontsize=20)
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.semilogy(loss_list)
plt.grid()
plt.show()

# Define posterior
mean = torch.zeros(latent_dim).to(device)
std = torch.ones(latent_dim).to(device)
def z_posterior(z):
    z_prior_score = torch.distributions.Normal(mean, std).log_prob(z).sum()

    generated_state, _ = generator(z.view(1, len(z)))
    gen_measurement = image_to_measurement(generated_state[0])
    gen_measurement = transform.inverse_transform_state(gen_measurement)
    error = x_measurement - gen_measurement
    error = error.detach()
    reconstruction_score = torch.distributions.Normal(noise_mean,
                                      2*noise_std).log_prob(error).sum()

    return z_prior_score + reconstruction_score

##### Set up Hamiltonian MC sampler #####
num_samples = 30000
step_size = 1.
num_steps_per_sample = 5
num_burns = 20000
integrator = hamiltorch.Integrator.IMPLICIT

hamiltorch.set_random_seed(123)
params_init = z_opt.clone().detach().view(latent_dim).to(device)
sampler = hamiltorch.Sampler.HMC_NUTS

z_samples = hamiltorch.sample(log_prob_func=z_posterior,
                              params_init=params_init,
                              num_samples=num_samples,
                              step_size=step_size,
                              num_steps_per_sample=num_steps_per_sample,
                              integrator=integrator,
                              sampler=sampler,
                              burn=num_burns,
                              desired_accept_rate=0.3)
x_measurement = x_measurement.to('cpu')


sample_batch_size = 64
z_samples = torch.stack(z_samples).to(device)
z_loader = torch.utils.data.DataLoader(z_samples,
                                       batch_size=sample_batch_size,
                                       shuffle=False,
                                       drop_last=False)

##### Reconstruct and plot #####
reconstructed_state = torch.zeros((num_samples - num_burns, 2, 256, 256))
pars_estimates = torch.zeros((num_samples - num_burns, 1))
edge_estimate = []
for idx, z in enumerate(z_loader):
    generated_state, generated_pars = generator(z)
    generated_state = generated_state.cpu().detach()

    reconstructed_state[idx * sample_batch_size:(idx * sample_batch_size + len(z))] = generated_state

    for i in range(len(z)):
        edge_estimate.append(compute_edge_location(generated_state[i,0,:,0:-10]))

    #pars_estimates[idx * 100:(idx * 100 + len(z)),1] = edge_estimate[-100:]
    pars_estimates[idx * sample_batch_size:(idx * sample_batch_size + len(z)),0] = generated_pars.cpu().detach()[:,0]

reconstructed_state = transform.inverse_transform_state(reconstructed_state)
x = transform.inverse_transform_state(x)

t2 = time.time()
print(f'TIME TO FINISH: {t2-t1} SECONDS')


pars_estimates = transform.inverse_transform_pars(pars_estimates)
#pars[1] = transform.inverse_transform_pars(pars[1])

z_samples = z_samples.to('cpu')

parameter_mean = torch.mean(pars_estimates[:], dim=0)
parameter_std = torch.std(pars_estimates[:], dim=0)

location_estimate = np.mean(edge_estimate)
location_std = np.std(edge_estimate)
true_location = pars[0]

size_estimate = parameter_mean[0]
size_std = parameter_std[0]
true_size = pars[1]



plt.figure()
plt.subplot(1,2,1)
plt.hist(np.asarray(edge_estimate), bins=20, density=True, alpha=0.4)
plt.axvline(x=[true_location], ymin=0, ymax=1e8, color='tab:red',
            linewidth=2, label='True location')
plt.axvline(x=[np.mean(edge_estimate)], ymin=0, ymax=1e8, linestyle='--',
            color='k', linewidth=2, label='MCGAN mean location')
plt.axvline(x=[MAP_estimate_location], ymin=0, ymax=1e8, linestyle='--',
            color='tab:green', linewidth=2, label='MAP location')
plt.xlabel('Lekage location [m]')
plt.legend()

plt.subplot(1,2,2)
plt.hist(pars_estimates[:,0], bins=20, density=True, alpha=0.4)
plt.axvline(x=[true_size], ymin=0, ymax=1e8, color='tab:red',
            linewidth=2, label='True discharge')
plt.axvline(x=[size_estimate], ymin=0, ymax=1e8, linestyle='--',
            color='k', linewidth=2, label='MCGAN mean discharge')
plt.axvline(x=[MAP_estimate_size], ymin=0, ymax=1e8, linestyle='--',
            color='tab:green', linewidth=2, label='MAP discharge')
plt.xlabel('Discharge coefficient')
plt.legend()
plt.show()


print(f'Leak location estimate: {location_estimate},', end=' ')
print(f'Leak location std: {location_std:0.5f}')
print(f'True leak location: {true_location:0.5f}')

print(f'Leak size estimate: {size_estimate:0.5f},', end=' ')
print(f'Leak size std: {size_std:0.5f}')
print(f'True leak size: {true_size:0.5f}')


mean_reconstructed_state = torch.mean(reconstructed_state, dim=0).to(
    'cpu').detach().numpy()
std_reconstructed_state = torch.std(reconstructed_state, dim=0).to(
    'cpu').detach().numpy()

print(f'Edge detection location guess: {np.mean(edge_estimate):0.3f},', end=' ')
print(f'Standard devication: {np.std(edge_estimate):0.3f}')

smooth_std = savgol_filter(np.sum(std_reconstructed_state[0,:,0:-5],axis=0), 11, 3)
std_peaks, _ =  find_peaks(smooth_std)
std_peak_prominences,_,_ = peak_prominences(smooth_std,std_peaks)
std_peak_widths,_,_,_ = peak_widths(smooth_std, std_peaks, rel_height=0.5)
std_peak_width = std_peak_widths[std_peak_prominences.argmax()]

std_guess = x_vec[std_peaks[std_peak_prominences.argmax()]]
std_variance = xmax/256*std_peak_width/2

print(f'Std location guess: {std_guess:0.3f},', end=' ')
print(f'Standard devication: {std_variance:0.3f}')

x_vec = np.linspace(xmin, xmax, num_x)
t_vec = np.linspace(0, 1.1*60, num_t)
X, T = np.meshgrid(x_vec, t_vec)

plt.figure()

plt.subplot(2, 2, 1)
plt.pcolor(T, X, x[0])
plt.title(f'Truth')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.pcolor(T, X, mean_reconstructed_state[0])
plt.title(f'Mean Reconstruction')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.pcolor(T, X, std_reconstructed_state[0])
plt.title(f'Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.pcolor(T, X, np.abs(mean_reconstructed_state[0] - x[0].detach().numpy()))
plt.title(f'Reconstruction Error')
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar()
plt.tight_layout(pad=1.5)
plt.show()

plt.figure()
plt.errorbar([1],np.mean(edge_estimate), np.std(edge_estimate),marker='.',markersize=20,
             elinewidth=4,label='Edge Estimate',uplims=True,lolims=True)
plt.errorbar([1],std_guess, std_variance,marker='.',markersize=20,
             elinewidth=2,label='Std Estimate',uplims=True,lolims=True)
plt.plot([1], true_location, '.',markersize=20, label='True Values')
plt.legend()
plt.grid()
x_values = ['Leak Location']
plt.xticks([1, 2], x_values)
plt.xlim([0, 3])
plt.title(f'Parameter Estimates')
plt.show()

print(f'edge mean: {np.mean(edge_estimate)}, edge std: {np.std(edge_estimate)}')
print(f'std mean: {std_guess}, std std: {std_variance}')

time_1 = 0
time_2 = 230

plt.figure()
plt.plot(x_vec,x[0, time_2, :], 'tab:blue', label=f'Truth t={t_vec[time_2]}')


plt.plot(x_vec,mean_reconstructed_state[0, time_2, :], 'tab:red',
         label=f'Truth t={t_vec[0]}')
plt.fill_between(x_vec,
                 mean_reconstructed_state[0,time_2,:]-std_reconstructed_state[0,time_2,:],
                 mean_reconstructed_state[0,time_2,:]+std_reconstructed_state[0,time_2,:],
                 alpha=0.2,color='tab:red')

plt.grid()
plt.savefig('pipeflow_GAN_reconstruction_at_times')
plt.show()

time_1_plot = 100
time_2_plot = 170
time_3_plot = 250

plt.figure()
plt.imshow(np.transpose(x[0].detach().numpy()),origin='lower', extent=[0,66,0,2000],
           vmin=torch.min(x[0]), vmax=torch.max(x[0]), aspect="auto")
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.savefig('pipe_true_state.pdf')

RMSE = np.linalg.norm(mean_reconstructed_state[0] - x[0].detach().numpy())/np.linalg.norm(x[0].detach().numpy())
plt.figure()
plt.imshow(np.transpose(mean_reconstructed_state[0]),origin='lower', extent=[0,66,0,2000],
           vmin=torch.min(x[0]), vmax=torch.max(x[0]), aspect="auto")
plt.title(f'Relative RMSE: {RMSE:0.2f}',fontsize=20)
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.savefig('pipe_MCGAN_state.pdf')

mean_std_state = np.mean(std_reconstructed_state[0])
plt.figure()
plt.imshow(np.transpose(std_reconstructed_state[0]),origin='lower', extent=[0,66,0,2000], aspect="auto",
           cmap='cividis')
plt.title(f'Average Std: {mean_std_state:0.2f}',fontsize=20)
plt.ylabel('Space [m]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.savefig('pipe_MCGAN_state_std.pdf')

plt.figure()
plt.plot(x_vec,mean_reconstructed_state[0,time_1_plot,:],'tab:green',
         label=f'MCGAN t={t_vec[time_1_plot]:.2f}', linestyle='--',linewidth=2)
plt.fill_between(x_vec,mean_reconstructed_state[0,time_1_plot,:]+std_reconstructed_state[0,time_1_plot,:],
                 mean_reconstructed_state[0,time_1_plot,:]-std_reconstructed_state[0,time_1_plot,:],
                 alpha=0.2,color='tab:green')
plt.plot(x_vec,x[0,time_1_plot,:],'tab:blue', linewidth=2,
         label=f'Truth t={t_vec[time_1_plot]:.2f}')
plt.ylim(np.min(x[0].detach().numpy()-0.15),
         np.max(x[0].detach().numpy())+0.15)
plt.xlabel('Space [m]')
plt.ylabel('Velocity [m/s]')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('pipe_reconstruction_at_t_0.pdf')

plt.figure()
plt.plot(x_vec,mean_reconstructed_state[0,time_2_plot,:],'tab:green',
         label=f'MCGAN t={t_vec[time_2_plot]:.2f}', linestyle='--',linewidth=2)
plt.fill_between(x_vec,mean_reconstructed_state[0,time_2_plot,:]+std_reconstructed_state[0,time_2_plot,:],
                 mean_reconstructed_state[0,time_2_plot,:]-std_reconstructed_state[0,time_2_plot,:],
                 alpha=0.2,color='tab:green')
plt.plot(x_vec,x[0,time_2_plot,:],'tab:blue', linewidth=2,
         label=f'Truth t={t_vec[time_2_plot]:.2f}')
plt.ylim(np.min(x[0].detach().numpy()-0.15),
         np.max(x[0].detach().numpy())+0.15)
plt.xlabel('Space [m]')
plt.ylabel('Velocity [m/s]')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('pipe_reconstruction_at_t_05.pdf')

plt.figure()
plt.plot(x_vec,mean_reconstructed_state[0,time_3_plot,:],'tab:green',
         label=f'MCGAN t={t_vec[time_3_plot]:.2f}', linestyle='--',linewidth=2)
plt.fill_between(x_vec,mean_reconstructed_state[0,time_3_plot,:]+std_reconstructed_state[0,time_3_plot,:],
                 mean_reconstructed_state[0,time_3_plot,:]-std_reconstructed_state[0,time_3_plot,:],
                 alpha=0.2,color='tab:green')
plt.plot(x_vec,x[0,time_3_plot,:],'tab:blue', linewidth=2,
         label=f'Truth t={t_vec[time_3_plot]:.2f}')
plt.ylim(np.min(x[0].detach().numpy()-0.15),
         np.max(x[0].detach().numpy())+0.15)
plt.xlabel('Space [m]')
plt.ylabel('Velocity [m/s]')
plt.legend(loc='lower left')
plt.grid()
plt.savefig('pipe_reconstruction_at_t_1.pdf')


loc_density = kde.gaussian_kde(np.asarray(edge_estimate))
size_density = kde.gaussian_kde(pars_estimates[:,0])

x_loc_density = np.linspace(0,2000,1000)
x_size_density = np.linspace(1e-4,9e-4,1000)
adv_density = loc_density(x_loc_density)
diff_density = size_density(x_size_density)

plt.figure()
plt.hist(np.asarray(edge_estimate), bins=20, density=True, alpha=0.4)
plt.plot(x_loc_density,adv_density, linewidth=3, color='tab:blue')
plt.plot(x_loc_density,1/(x_loc_density[-1]-x_loc_density[0])*np.ones(x_loc_density.shape),
         linewidth=1.5, color='black', label='Prior')
plt.axvline(x=[true_location], ymin=0, ymax=1e8, color='tab:red',
            linewidth=2, label='True location')
plt.axvline(x=[np.mean(edge_estimate)], ymin=0, ymax=1e8, linestyle='--',
            color='k', linewidth=2, label='MCGAN mean location')
plt.axvline(x=[MAP_estimate_location], ymin=0, ymax=1e8, linestyle='--',
            color='tab:green', linewidth=2, label='MAP location')
plt.xlabel('Leakage location [m]')
plt.legend()
plt.savefig('pipe_location_histogram.pdf')

plt.figure()
plt.hist(pars_estimates[:,0], bins=20, density=True, alpha=0.4)
plt.plot(x_size_density,diff_density, linewidth=3, color='tab:blue')
plt.plot(x_size_density,1/(x_size_density[-1]-x_size_density[0])*np.ones(x_size_density.shape),
         linewidth=1.5, color='black', label='Prior')
plt.axvline(x=[true_size], ymin=0, ymax=1e8, color='tab:red',
            linewidth=2, label='True discharge')
plt.axvline(x=[size_estimate], ymin=0, ymax=1e8, linestyle='--',
            color='k', linewidth=2, label='MCGAN mean discharge')
plt.axvline(x=[MAP_estimate_size], ymin=0, ymax=1e8, linestyle='--',
            color='tab:green', linewidth=2, label='MAP discharge')
plt.xlabel('Discharge coefficient')
plt.legend()
plt.savefig('pipe_size_histogram.pdf')


RMSE = np.linalg.norm(mean_reconstructed_state[0] - x[0].detach().numpy())/np.linalg.norm(x[0].detach().numpy())
RMSE += np.linalg.norm(mean_reconstructed_state[-1] - x[1].detach().numpy())/np.linalg.norm(x[1].detach().numpy())
RMSE = RMSE/2

std = np.mean(std_reconstructed_state[0])

print(f'RMSE = {np.mean(RMSE)}')
print(f'STD = {np.mean(std)}')

print(f'Location: mean={np.mean(edge_estimate):0.4f},std={np.std(edge_estimate):0.4f}')
print(f'Discharge: mean={size_estimate:0.8f},std={np.std(pars_estimates[:,0].detach().numpy()):0.8f}')


par_RMSE = np.sqrt((true_location - np.mean(edge_estimate))**2+(true_size - size_estimate)**2)/ \
                    np.sqrt((true_location)**2+(true_size)**2)

print(f'PAR RMSE = {par_RMSE}')