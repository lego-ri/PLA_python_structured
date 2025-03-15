""" 
    Plotting function for batch L,L-lactide ROP
"""
""" 
    Uses:
    data from the main_process.py - solution of the ODE model
    parameters from model_pars.py - called from the main_process.py
""" 
"""
    Used in: 
    main_process
"""

#* Import necessary libraries 
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from measure_time import measure_time

                

#* The main function that plots the results and saves the data for MC simulation
@measure_time
def plot_results(solution, pars):
    """ 
        Plots the results and saves them to .csv and .npy files
    """
    #* Extract results from the main_process
    t = solution.t  # Reaction time, s
    y = solution.y  # All the state variables at each time step t

    #* Assign the State variables 
    M = y[0]     # Monomer concentration, mol/m3
    C = y[1]     # Catalyst concentration, mol/m3
    A = y[2]     # Acid concentration, mol/m3
    la0 = y[3]   # 0th moment of active chains, mol/m3
    la1 = y[4]   # 1st moment of active chains, 
    la2 = y[5]   # 2nd moment of active chains, 
    mu0 = y[6]   # 0th moment of dormant chains, mol/m3
    mu1 = y[7]   # 1st moment of dormant chains,
    mu2 = y[8]   # 2nd moment of dormant chains,
    ga0 = y[9]   # 0th moment of terminated chains, mol/m3
    ga1 = y[10]  # 1st moment of terminated chains
    ga2 = y[11]  # 2nd moment of terminated chains

    t_h = t / 3600     # Time in hours (for plotting)

    #* Conversion of monomer (%)
    conv = (pars.M0 - M) / pars.M0 * 100

    #* Number-average molecular weight of polymer
    Mn_ODE = pars.MW * (la1 + mu1 + ga1) / (la0 + mu0 + ga0)

    #* Weight-average molecular weight of polymer
    Mw_ODE = pars.MW * (la2 + mu2 + ga2) / (la1 + mu1 + ga1)

    #* Print final conversion, Mw, and PDI
    print(f'Conversion = {conv[-1]:.2f} [%]')           # [-1] is the last element of the array,
    print(f'Mw = {Mw_ODE[-1]:.2f} [kg/mol]')                 #  :.2f formats the number to 2 decimal places
    print(f'PDI = {Mw_ODE[-1] / Mn_ODE[-1]:.2f}')   # Polydispersity index ("Zp"), width of the distribution
    # Zp = 1 -> 100% monodisperse polymer, Zp = 1.5..2.5 -> normal distribution,  Zp > 5 -> broad distribution

    #* Rates of elementary reaction steps
    R_a1  = pars.k_a1 * C * mu0
    R_a2  = pars.k_a2 * A * la0
    R_p   = pars.k_p  * la0 * M
    R_d   = pars.k_d  * la0
    R_s   = pars.k_s  * la0 * mu0
    R_te1 = pars.k_te * la0 * la0
    R_te2 = pars.k_te * la0 * mu0
    R_te3 = pars.k_te * la0 * ga0
    R_de1 = pars.k_de * la0
    R_de2 = pars.k_de * mu0
    R_de3 = pars.k_de * ga0
    R_sum = R_a1 + R_a2 + R_p + R_d + R_s + R_te1 + R_te2 + R_te3 + R_de1 + R_de2 + R_de3

    #* Plot conversion vs. time
    plt.figure()
    plt.plot(t_h, conv)
    plt.xlabel('Time (hours)')
    plt.ylabel('Conversion (%)')
    plt.grid(True)
    plt.show(block=False)   # Display the plot and make sure the code keeps running (stops by default - block)

    #* Plot Mn, Mw vs. time
    plt.figure()
    plt.plot(t_h, Mn_ODE, label='M_n')
    plt.plot(t_h, Mw_ODE, label='M_w')
    plt.xlabel('Time (hours)')
    plt.ylabel('Average polymer molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    #* Plot Mn, Mw vs. conversion
    plt.figure()
    plt.plot(conv, Mn_ODE, label='M_n')
    plt.plot(conv, Mw_ODE, label='M_w')
    plt.xlabel('Conversion (%)')
    plt.ylabel('Average polymer molecular weight (kg/mol)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    #* Plot concentration of species
    plt.figure()
    plt.plot(t_h, M / 1e3, label='Monomer (div. by 1000)')
    plt.plot(t_h, C, label='Catalyst')
    plt.plot(t_h, A, label='Acid')
    plt.plot(t_h, la0, label='Active chains')
    plt.plot(t_h, mu0, label='Dormant chains')
    plt.plot(t_h, ga0, label='Terminated chains')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mol/m^3)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    #* Plot relative reaction rates with improved readability
    plt.figure(figsize=(12, 6))  # Increase figure size for better visibility
    # Custom color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

    plt.plot(t_h, R_a1  / R_sum * 100, color=colors[0], label='Catalyst activation', linestyle='-.', linewidth=3)
    plt.plot(t_h, R_a2  / R_sum * 100, color=colors[1], label='Catalyst deactivation', linestyle='--')
    plt.plot(t_h, R_p   / R_sum * 100, color=colors[2], label='Propagation', linestyle='--')
    plt.plot(t_h, R_d   / R_sum * 100, color=colors[3], label='Depropagation', linestyle='--')
    plt.plot(t_h, R_s   / R_sum * 100, color=colors[4], label='Chain transfer', linestyle='-.')
    plt.plot(t_h, R_te1 / R_sum * 100, color=colors[5], label='Transesterification (active)', linestyle='-', linewidth=3)
    plt.plot(t_h, R_te2 / R_sum * 100, color=colors[6], label='Transesterification (dormant)', linestyle='--')
    plt.plot(t_h, R_te3 / R_sum * 100, color=colors[7], label='Transesterification (terminated)', linestyle=':')
    plt.plot(t_h, R_de1 / R_sum * 100, color=colors[8], label='Chain scission (active)', linestyle='-.', linewidth=3)
    plt.plot(t_h, R_de2 / R_sum * 100, color=colors[9], label='Chain scission (dormant)', linestyle='--')
    plt.plot(t_h, R_de3 / R_sum * 100, color=colors[10], label='Chain scission (terminated)', linestyle=':')

    plt.xlim(0, max(t_h))  # Set x-axis limits
    plt.xlabel('Time (hours)')
    plt.ylabel('Relative reaction rate (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show(block=False)    
    
    #* Plot chain moments
    plt.figure()

    plt.subplot(3, 2, 1)
    plt.plot(t_h, la0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Active chains, 0-moment')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(t_h, la1)
    plt.xlabel('Time (hours)')
    plt.ylabel('Active chains, 1-moment')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(t_h, mu0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Dormant chains, 0-moment')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t_h, mu1)
    plt.xlabel('Time (hours)')
    plt.ylabel('Dormant chains, 1-moment')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t_h, ga0)
    plt.xlabel('Time (hours)')
    plt.ylabel('Dead chains, 0-moment')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(t_h, ga1)
    plt.xlabel('Time (hours)')
    plt.ylabel('Dead chains, 1-moment')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    #* Save the variables relevant for the Monte Carlo simulation
    def save_to_csv(filename):
        # Create a dictionary of data
        data = {
            'Time (s)': t,
            'Monomer Concentration': M,
            'Catalyst Concentration': C,
            'Acid Concentration': A,
            'Mn_ODE': Mn_ODE,
            'Mw_ODE': Mw_ODE
        }
        data_frame = pd.DataFrame(data)                 # Convert the dictionary to a Pandas DataFrame
        data_frame.to_csv(filename, index=False)        # Index=False ensures that row names are not included in the .csv

    #* Save the data relevant for the Monte Carlo simulation
    def save_to_npy(filename):
        # Create a structured array
        data = np.array(list(zip(t, M, C, A, Mn_ODE, Mw_ODE, la0, mu0, ga0)), 
                        dtype=[('Time', float), ('Monomer', float), ('Catalyst', float), 
                               ('Acid', float), ('Mn_ODE', float), ('Mw_ODE', float), ('la0', float), ('mu0', float),('ga0', float)])
        np.save(filename, data)

    # Save the data to a .npy file
    save_to_npy('deterministic_results_for_MC.npy')
    
    # Save the data to a .csv file
    save_to_csv('deterministic_results_for_MC.csv')