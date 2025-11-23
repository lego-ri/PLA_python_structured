import numpy as np                      # Numerical library in Python
import matplotlib.pyplot as plt         # For plotting in post processing
from scipy.signal import savgol_filter  # Smoothen the data for plotting
from measure_time import measure_time   # Handle that measures time a function takes to execute
from model_pars import ModelPars        # Model parameters

@measure_time
def plot_MC_results(MC_output, plot_pars):
    
    
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
    Mw_ODE_2 = plot_pars[3]
    Mw_ODE_3 = plot_pars[4]
  
    MW = ModelPars.MW        # Molecular weight of lactoyl (monomer) group, kg/mol
    
    # Other parameters for plotting
    small_number = 1e-6      # Small number for numerical stability

    
    # Plot reaction rates
    plt.figure(1)
    sumRates = np.sum(Rates_out, axis=0) + small_number # axis=0 to sum over columns
    for i in range(9):
        if i < 6:
            plt.plot(t_out, Rates_out[i, :] / sumRates * 100)   # make it % by *100
        elif i == 6:
            plt.plot(t_out, Rates_out[6, :]  / sumRates * 100)
        elif i == 7:
            plt.plot(t_out, Rates_out[7, :]  / sumRates * 100)
        elif i == 9:
            plt.plot(t_out, Rates_out[8, :] / sumRates * 100)

    plt.xlabel('Time (s)')
    plt.ylabel('Relative reaction rates (%)')
    plt.legend(['Propagation', 'Depropagation',
                'Chain transfer', 'Scission (R)', 'Scission (D)', 'Scission (G)',
                'Transesterification (R+D)', 'Transesterification (R+R)',
                'Transesterification (R+G)'])
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot transesterification rates only
    plt.figure(2)
    sumRates = np.sum(Rates_out, axis=0) + small_number # axis=0 to sum over columns
    plt.plot(t_out, (Rates_out[6, :] / sumRates * 100))
    plt.plot(t_out, (Rates_out[7, :]  / sumRates * 100))
    plt.plot(t_out, (Rates_out[8, :] / sumRates * 100))
    plt.xlabel('Time (s)')  
    plt.ylabel('Relative reaction rates (%)')
    plt.legend(['Transesterification (R+D)', 'Transesterification (R+R)', 'Transesterification (R+G)'])
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    
     # Plot transesterification rates only
    plt.figure(3)
    sumRates = np.sum(Rates_out, axis=0) + small_number # axis=0 to sum over columns
    plt.plot(t_out, Rates_out[6, :] / sumRates * 100)
    plt.plot(t_out, Rates_out[7, :] / sumRates * 100)

    plt.xlabel('Time (s)')  
    plt.ylabel('Relative reaction rates (%)')
    plt.legend(['Transesterification (R+D) active','Transesterification (R+D) passive'])
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # Plot chain concentrations
    plt.figure(4)
    plt.plot(t_out, R_out, label='Active branches')
    plt.plot(t_out, D_out, label='Dormant branches')
    plt.plot(t_out, G_out, label='Terminated branches')
    plt.plot(t_out, (R_out+D_out+G_out), label='Total number of branches')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of branches (-)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # Plot average molecular weights
    plt.figure(5)
    plt.plot(t_out, savgol_filter(Mn_out, 5, 3), 'b-', label='M_n (MC)')    # Smoothing the data using savgol_filter 
    plt.plot(t_out, savgol_filter(Mw_out, 5, 3), 'r-', label='M_w (MC)')
    plt.plot(t, Mn_ODE, 'b.', label='M_n (ODE)')
    plt.plot(t, Mw_ODE, 'r.', label='M_w (ODE)')
    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


    # Plot MWD (mlecular weight ddistribution)
    plt.figure(6)
    all_chains = MW*np.concatenate((RD_column_sums, G))
    logMW = np.log10(all_chains)
    plt.hist(logMW, bins=30)
    plt.xlabel('Log MW (kg/mol)')
    plt.ylabel('Frequency')
    # plt.xlim([-2,max(logMW)+1])
    plt.tight_layout()
    plt.show(block=False)
    
    #* Plot ratio between Mw,Mn from ODE and MC
    # Find the closest indices in `t_out` for each value in `t`
    closest_indices = [np.argmin(np.abs(t_out - t_val)) for t_val in t]

    # Extract the corresponding `Mn_out` values using the closest indices
    Mn_out_closest = Mn_out[closest_indices]

    # Filter the data to include only time points from 1 second onward (mismatch too big for t<1s)
    exclude_time = 30 # s
    t_filtered = t[t >= exclude_time]
    Mn_ODE_filtered = Mn_ODE[t >= exclude_time]
    Mn_out_closest_filtered = Mn_out_closest[t >= exclude_time]

    # Calculate the ratio for the filtered data
    ratio_Mn_filtered = Mn_out_closest_filtered / Mn_ODE_filtered
        
    # Extract the corresponding `Mw_out` values using the closest indices
    Mw_out_closest = Mw_out[closest_indices]

    # Filter the data to include only time points from exclude_time second onward
    Mw_ODE_filtered = Mw_ODE[t >= exclude_time]
    Mw_out_closest_filtered = Mw_out_closest[t >= exclude_time]

    # Calculate the ratio for the filtered data
    ratio_Mw_filtered = Mw_out_closest_filtered / Mw_ODE_filtered

    # Plot the ratio from exclude_time second onward
    plt.figure(7)
    plt.plot(t_filtered, ratio_Mn_filtered, 'b-', label='Mn (MC) / Mn (ODE)')
    plt.plot(t_filtered, ratio_Mw_filtered, 'r-', label='Mw (MC) / Mw (ODE)')
    plt.xlabel('Time (s)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    
    #* plot Mw_ODE 
    plt.figure(8)
    plt.plot(t, Mw_ODE, 'r.', label='M_w (ODE)')
    plt.plot(t, Mw_ODE_2, 'b-', label='M_w (ODE) 2')
    plt.plot(t, Mw_ODE_3, 'g*', label='M_w (ODE) 3')
    plt.show(block=False)


    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot average molecular weights without smoothing
    plt.figure(9)
    plt.plot(t_out, Mn_out, 'b-', label='M_n (MC, raw)')
    plt.plot(t_out, Mw_out, 'r-', label='M_w (MC, raw)')
    plt.plot(t, Mn_ODE, 'b.', label='M_n (ODE)')
    plt.plot(t, Mw_ODE, 'r.', label='M_w (ODE)')
    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # ===============================================================
    # NEW FIGURE: Compare MW_ODE_4 / Mn_ODE with MC results
    # ===============================================================

    Mw_ODE_4 = plot_pars[5]   # <- new input

    # Plot MC vs deterministic including MW_ODE_4
    plt.figure(10)
    plt.plot(t_out, savgol_filter(Mn_out, 5, 3), 'b-', label='M_n (MC)')
    plt.plot(t_out, savgol_filter(Mw_out, 5, 3), 'r-', label='M_w (MC)')
    
    plt.plot(t, Mn_ODE, 'b.', markersize=4, label='M_n (ODE)')
    plt.plot(t, Mw_ODE_4, 'm.', markersize=4, label='M_w (ODE_4, branched correct)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    # ===============================================================
    # NEW FIGURE: direct side-by-side Mn + Mw for MC vs ODE_4
    # ===============================================================
    plt.figure(11)

    # MC
    plt.plot(t_out, Mn_out, 'b-', alpha=0.6, label='M_n (MC)')
    plt.plot(t_out, Mw_out, 'r-', alpha=0.6, label='M_w (MC)')

    # ODE
    plt.plot(t, Mn_ODE, 'b.', markersize=4, label='M_n (ODE)')
    plt.plot(t, Mw_ODE_4, 'm.', markersize=4, label='M_w (ODE_4)')

    plt.xlabel('Time (s)')
    plt.ylabel('Average molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    
    plt.show()
