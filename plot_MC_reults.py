import matplotlib.pyplot as plt         # For plotting in post processing
from scipy.signal import savgol_filter  # Smoothen the data for plotting
import numpy as np


def plot_MC_reults(MC_output, plot_pars):
    
    
    # Unpack MC_output parameters
    t_out = MC_output[0]
    Rates_out = MC_output[1]
    R_out = MC_output[2]
    D_out = MC_output[3]
    G_out = MC_output[4]
    Mn_out = MC_output[5]
    Mw_out = MC_output[6]
    suma_n_tot = MC_output[7]
    RD_column_sums = MC_output[8]
    G = MC_output[9]
    
    # Unpack other paramaeters for plotting
    t = plot_pars[0]
    Mn_ODE = plot_pars[1]
    Mw_ODE = plot_pars[2]
    MW = plot_pars[3]
    
    # Other parameters for plotting
    small_number = 1e-6      # Small number for numerical stability

    
    # Plot reaction rates
    plt.figure(1)
    sumRates = np.sum(Rates_out, axis=0) + small_number # axis=0 to sum over columns
    for i in range(9):
        if i < 6:
            plt.plot(t_out, Rates_out[i, :] / sumRates * 100)   # make it % by *100
        elif i == 6:
            plt.plot(t_out, (Rates_out[6, :] + Rates_out[7, :]) / sumRates * 100)
        elif i == 7:
            plt.plot(t_out, (Rates_out[8, :] + Rates_out[9, :]) / sumRates * 100)
        elif i == 9:
            plt.plot(t_out, (Rates_out[10, :] + Rates_out[11, :]) / sumRates * 100)

    plt.xlabel('Time (s)')
    plt.ylabel('Relative reaction rates (%)')
    plt.legend(['Propagation', 'Depropagation',
                'Chain transfer', 'Scission (R)', 'Scission (D)', 'Scission (G)',
                'Transesterification (R+D)', 'Transesterification (R+R)',
                'Transesterification (R+G)'])
    plt.grid(True)
    plt.tight_layout()

    # Plot chain concentrations
    plt.figure(2)
    plt.plot(t_out, R_out, label='Active chains')
    plt.plot(t_out, D_out, label='Dormant chains')
    plt.plot(t_out, G_out, label='Terminated chains')
    plt.plot(t_out, suma_n_tot, label='Total number of chains')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of chains (-)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot average molecular weights
    plt.figure(3)
    plt.plot(t_out, savgol_filter(Mn_out, 5, 3), 'b-', label='M_n (MC)')    # Smoothing the data using savgol_filter 
    plt.plot(t_out, savgol_filter(Mw_out, 5, 3), 'r-', label='M_w (MC)')
    plt.plot(t, Mn_ODE, 'b.', label='M_n (ODE)')
    plt.plot(t, Mw_ODE, 'r.', label='M_w (ODE)')
    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # Plot MWD (mlecular weight ddistribution)
    plt.figure(4)
    all_chains = MW*np.concatenate((RD_column_sums, G))
    logMW = np.log10(all_chains)
    plt.hist(logMW, bins=30)
    plt.xlabel('Log MW (kg/mol)')
    plt.ylabel('Frequency')
    # plt.xlim([-2,max(logMW)+1])
    plt.tight_layout()

    plt.show()
