import numpy as np
import matplotlib.pyplot as plt


def plot_contours(MCGAN_results, true_state, domain, pressure_or_velocity='velocity'):
    x_vec = np.linspace(domain['xmin'], domain['xmax'], domain['numx'])
    t_vec = np.linspace(0, 1.1 * 60, domain['numx'])
    X, T = np.meshgrid(x_vec, t_vec)

    if pressure_or_velocity == 'pressure':
        idx = 1
    else:
        idx = 0

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.pcolor(T, X, true_state[idx].cpu().detach().numpy())
    plt.title(f'Truth')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.pcolor(T, X, MCGAN_results['state_mean'][idx].cpu().detach().numpy())
    plt.title(f'Mean Reconstruction')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.pcolor(T, X, MCGAN_results['state_std'][idx].cpu().detach().numpy())
    plt.title(f'Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.pcolor(T, X,
               np.abs(MCGAN_results['state_mean'][idx].cpu().detach().numpy()\
                      - true_state[idx].cpu().detach().numpy()))
    plt.title(f'Reconstruction Error')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.colorbar()
    plt.tight_layout(pad=1.5)
    plt.show()

#def plot_state(MCGAN_results, true_data):


#def plot_par_histograms(MCGAN_results, true_data):