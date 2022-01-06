import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde


def plot_contours(MCGAN_results, true_state, domain,
                  pressure_or_velocity='velocity',
                  save_string='MCGAN_contour_plots.pdf'):

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

    plt.savefig(save_string)
    plt.show()

def plot_state(MCGAN_results, true_state, domain,
               time_plot_ids, pressure_or_velocity='velocity',
               save_string='MCGAN_state_plots.pdf'):

    x_vec = np.linspace(domain['xmin'], domain['xmax'], domain['numx'])
    t_vec = np.linspace(domain['tmin'], domain['tmax'], domain['numt'])
    X, T = np.meshgrid(x_vec, t_vec)

    if pressure_or_velocity == 'pressure':
        idx = 1
    else:
        idx = 0

    MCGAN_state = MCGAN_results['state_mean'][idx, :, :].cpu().detach().numpy()
    MCGAN_std = MCGAN_results['state_std'][idx, :, :].cpu().detach().numpy()
    true_state = true_state.cpu().detach().numpy()

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.plot(x_vec, MCGAN_state[time_plot_ids[0], :], 'tab:green',
             label=f'MCGAN t={t_vec[time_plot_ids[0]]:.2f}', linestyle='--',
             linewidth=2)
    plt.fill_between(x_vec,
                     MCGAN_state[time_plot_ids[0], :] + MCGAN_std[time_plot_ids[0], :],
                     MCGAN_state[time_plot_ids[0], :] - MCGAN_std[time_plot_ids[0], :],
                     alpha=0.2, color='tab:green')
    plt.plot(x_vec, true_state[idx, time_plot_ids[0], :], 'tab:blue', linewidth=2,
             label=f'Truth t={t_vec[time_plot_ids[0]]:.2f}')
    plt.ylim(np.min(MCGAN_state) - 0.03*np.min(MCGAN_state),
             np.max(MCGAN_state) + 0.03*np.max(MCGAN_state))
    plt.xlabel('Space [m]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc='lower left')
    plt.grid()

    plt.subplot(1,3,2)
    plt.plot(x_vec, MCGAN_state[time_plot_ids[1], :], 'tab:green',
             label=f'MCGAN t={t_vec[time_plot_ids[1]]:.2f}', linestyle='--',
             linewidth=2)
    plt.fill_between(x_vec,
                     MCGAN_state[time_plot_ids[1], :] + MCGAN_std[time_plot_ids[1], :],
                     MCGAN_state[time_plot_ids[1], :] - MCGAN_std[time_plot_ids[1], :],
                     alpha=0.2, color='tab:green')
    plt.plot(x_vec, true_state[idx, time_plot_ids[1], :], 'tab:blue',
             linewidth=2,
             label=f'Truth t={t_vec[time_plot_ids[1]]:.2f}')
    plt.ylim(np.min(MCGAN_state) - 0.03 * np.min(MCGAN_state),
             np.max(MCGAN_state) + 0.03 * np.max(MCGAN_state))
    plt.xlabel('Space [m]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc='lower left')
    plt.grid()

    plt.subplot(1,3,3)
    plt.plot(x_vec, MCGAN_state[time_plot_ids[2], :], 'tab:green',
             label=f'MCGAN t={t_vec[time_plot_ids[2]]:.2f}', linestyle='--',
             linewidth=2)
    plt.fill_between(x_vec,
                     MCGAN_state[time_plot_ids[2], :] + MCGAN_std[time_plot_ids[2], :],
                     MCGAN_state[time_plot_ids[2], :] - MCGAN_std[time_plot_ids[2], :],
                     alpha=0.2, color='tab:green')
    plt.plot(x_vec, true_state[idx, time_plot_ids[2], :], 'tab:blue',
             linewidth=2,
             label=f'Truth t={t_vec[time_plot_ids[2]]:.2f}')
    plt.ylim(np.min(MCGAN_state) - 0.03 * np.min(MCGAN_state),
             np.max(MCGAN_state) + 0.03 * np.max(MCGAN_state))
    plt.xlabel('Space [m]')
    plt.ylabel('Velocity [m/s]')
    plt.legend(loc='lower left')
    plt.grid()

    plt.savefig(save_string)
    plt.show()


def plot_par_histograms(MCGAN_results, true_pars):
    loc_density = kde.gaussian_kde(MCGAN_results['gen_pars'][:,0])
    size_density = kde.gaussian_kde(MCGAN_results['gen_pars'][:,1])

    x_loc_density = np.linspace(0, 2000, 1000)
    x_size_density = np.linspace(1e-4, 9e-4, 1000)
    adv_density = loc_density(x_loc_density)
    diff_density = size_density(x_size_density)

    plt.figure()
    plt.hist(MCGAN_results['gen_pars'][:,0], bins=20, density=True, alpha=0.4)
    plt.plot(x_loc_density, adv_density, linewidth=3, color='tab:blue')
    plt.plot(x_loc_density,
             1 / (x_loc_density[-1] - x_loc_density[0]) * np.ones(
                 x_loc_density.shape),
             linewidth=1.5, color='black', label='Prior')
    plt.axvline(x=[true_pars[0]], ymin=0, ymax=1e8, color='tab:red',
                linewidth=2, label='True location')
    plt.axvline(x=[MCGAN_results['par_mean'][0]], ymin=0, ymax=1e8, linestyle='--',
                color='k', linewidth=2, label='MCGAN mean location')
    #plt.axvline(x=[MAP_estimate_location], ymin=0, ymax=1e8, linestyle='--',
    #            color='tab:green', linewidth=2, label='MAP location')
    plt.xlabel('Leakage location [m]')
    plt.legend()
    plt.savefig('pipe_location_histogram.pdf')
    plt.show()

    plt.figure()
    plt.hist(MCGAN_results['gen_pars'][:,1], bins=20, density=True, alpha=0.4)
    plt.plot(x_size_density, diff_density, linewidth=3, color='tab:blue')
    plt.plot(x_size_density,
             1 / (x_size_density[-1] - x_size_density[0]) * np.ones(
                 x_size_density.shape),
             linewidth=1.5, color='black', label='Prior')
    plt.axvline(x=[true_pars[1]], ymin=0, ymax=1e8, color='tab:red',
                linewidth=2, label='True discharge')
    plt.axvline(x=[MCGAN_results['par_mean'][1]], ymin=0, ymax=1e8, linestyle='--',
                color='k', linewidth=2, label='MCGAN mean discharge')
    #plt.axvline(x=[MAP_estimate_size], ymin=0, ymax=1e8, linestyle='--',
    #            color='tab:green', linewidth=2, label='MAP discharge')
    plt.xlabel('Discharge coefficient')
    plt.legend()
    plt.savefig('pipe_size_histogram.pdf')
    plt.show()