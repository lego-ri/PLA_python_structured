"""
Batch ring-opening polymerization of L,L-lactide simulated using       
 Monte Carlo method (Gillespie algorithm)                               
 Written by Alexandr Zubov, DTU Chemical Engineering                   
 Last update: Oct 2024 - Lego Richard UCT Prague
"""

""" Uses:
model_pars.py: contains parameters of batch L,L-lactide ROP
'deterministic_results_for_MC.npy': data from ODE model, which is used for the MC model
"""

#* Import necessary modules and libraries
import itertools                        # for handling data in arrays (creationg subarrays etc.)
import sys                              # for exitting the code if something goes wrong
from time import process_time           # To measure the time taken by the simulation
from model_pars import ModelPars        # Parameters of batch L,L-lactide ROP (Ring opening polymerization)
import numpy as np                      # For mathematical operations and data handling (e.g. importing and exporting)
from find_Nx import find_Nx
from build_matrixes_forMC import build_matrixes_forMC
from monte_carlo_algorithm import monte_carlo_algorithm
from plot_MC_results import plot_MC_results

total_time = process_time()      # Start the timer to measure the total run time of the file

print("Program main_MC_branched running...")

#* Load variables evaluated by the main_process 
data = np.load('deterministic_results_for_MC_branched.npy')  # Load the data from the main_process 
t = data['Time']                                    # Time within the reaction, s
M = data['Monomer']                                 # Monomer concentration, mol/m3
C = data['Catalyst']                                # Catalyst concentration, mol/m3
A = data['Acid']                                    # Acid concentration, mol/m3
Mn_ODE = data['Mn_ODE']                             # Number-average molecular weight of polymer
Mw_ODE = data['Mw_ODE']                             # Weight-average molecular weight of polymer
la0 = data['la0']                                   # 0th moment of active chains - concentration, mol/m3
mu0 = data['mu0']                                   # 0th moment of dormant chains - concentration, mol/m3
ga0 = data['ga0']                                   # 0th moment of terminated chains - concentration, mol/m3

#* "Instantenious equilibrium R vs D"
# The concentration of dormant and active chains becomes constant very quickly,
# thus can be assumed constant from the beggining throughout the MC simulation.
R_conc = la0[-1]    # concentration of active chains
D_conc = mu0[-1]    # concentration of dormant chains

#! #######################################################################################################################
#TODO: Define the initial composition of cocatalysts:
max_branches = 3                                              # Number of branches of the most branched cocatalyst molecule
D0_composition = np.zeros(max_branches)                      
D0_composition[0]      = 0#0.15                               # Fraction of chains (branches) in linear cocatalysts
D0_composition[1]      = 0#0.33                               # Fraction of chains in cocatalysts with 2 branches
D0_composition[-1]      = 1 - np.sum(D0_composition[:-1])     # Fraction of chains in cocatalysts with max_branches branches
#! #######################################################################################################################

print(f"User input D0_composition is: {D0_composition}")

# Check if the given fractions for cocatalyst composition sum up to 1
if D0_composition[-1] < 0: 
    print('Error: The initial composition does not sum up to 1!!!')
    exit()  
    
#* Check if the model can be run with this initial composition and find the corresponding Nx
# Define the parameters for the optimal Nx search
min_Nx = 500                         # Minimal Nx
max_Nx = 5000                        # Maximal Nx    
max_difference_RD_dec_round = 1e-2   # Maximal difference between rounded and non-rounded number of D,R
eps_fraction = 1e-3                  # Margin for finding close fractions to the defined inlet composition

# Pack pars for Nx search
Nx_pars = [min_Nx, max_Nx, max_difference_RD_dec_round, eps_fraction] 
process_pars = [D_conc, R_conc, ga0, t]

# Find the optimal Nx
D0_composition, Nx, D_round, R_round = find_Nx(D0_composition, Nx_pars, process_pars)

#* Build matrixes for D and R chains for the MC simulation
D, R, G, total_num_D_mol = build_matrixes_forMC(D0_composition, Nx, R_round, max_branches)

#* Main MC simulation loop 
# Pack parameters for the MC simulation
mc_pars = [Nx, M, C, A, R, D, G, total_num_D_mol]

# Run the MC simulation
t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, R, D, G = monte_carlo_algorithm(mc_pars, process_pars, ModelPars)

#* Post processing
# Pack MC results for plotting
MC_output = [t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, G]
plot_pars = [t, Mn_ODE, Mw_ODE]

# Plotting
plot = plot_MC_results(MC_output, plot_pars)




########################################################################3
total_time = process_time() - total_time
print(f"Total processor time taken to run the whole main_MC_branched program: {total_time} seconds")

