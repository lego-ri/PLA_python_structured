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
# TODO: clean up the code

#* Import necessary modules and libraries
import itertools                        # for handling data in arrays (creationg subarrays etc.)
import sys                              # for exitting the code if something goes wrong
from time import process_time           # To measure the time taken by the simulation
from model_pars import ModelPars        # Parameters of batch L,L-lactide ROP (Ring opening polymerization)
import numpy as np                      # For mathematical operations and data handling (e.g. importing and exporting)
from scipy import interpolate           # To fit the monomer profile as piecewise linear
from select_reaction import SelReac     # For semi-random chosing of the reaction to occure
import matplotlib.pyplot as plt         # For plotting in post processing
from scipy.signal import savgol_filter  # Smoothen the data for plotting
from find_Nx import find_Nx
from build_matrixes_forMC import build_matrixes_forMC

total_time = process_time()      # Start the timer to measure the total run time of the file

#* Load variables evaluated by the main_process 
data = np.load('deterministic_results_for_MC.npy')  # Load the data from the main_process 
t = data['Time']                                    # Time within the reaction, s
M = data['Monomer']                                 # Monomer concentration, mol/m3
C = data['Catalyst']                                # Catalyst concentration, mol/m3
A = data['Acid']                                    # Acid concentration, mol/m3
Mn_ODE = data['Mn_ODE']                             # Number-average molecular weight of polymer
Mw_ODE = data['Mw_ODE']                             # Weight-average molecular weight of polymer
la0 = data['la0']                                   # 0th moment of active chains - concentration, mol/m3
mu0 = data['mu0']                                   # 0th moment of dormant chains - concentration, mol/m3
ga0 = data['ga0']                                   # 0th moment of terminated chains - concentration, mol/m3

#* Parameters of the Monte Carlo simulation
small_number = 1e-6      # Small number for numerical stability
N_A = 6.022140858e23     # Avogadro number, 1/mol
MW = ModelPars.MW        # Molecular weight of lactoyl (monomer) group, kg/mol
time_end = t[-1]         # Simulation time, s
# D0 = ModelPars.D0        # Initial concentration of cocatalyst (i.e. initial dormant chains), mol/m3
# k_a1 = ModelPars.k_a1    # Activation rate coefficient, m3/mol/s
# k_a2 = ModelPars.k_a2    # Deactivation rate coefficient, m3/mol/s
k_p = ModelPars.k_p      # Propagation rate coefficient, m3/mol/s
k_d = ModelPars.k_d      # Depropagation rate coefficient, 1/s
k_s = ModelPars.k_s      # Chain-transfer rate coefficient m3/mol/s
k_te = ModelPars.k_te    # Intermolecular transesterification rate coefficient, m3/mol/s
k_de = ModelPars.k_de    # Nonradical random chain scission rate coefficient, 1/s
fq = 10                  # Fequency of data output (every fq in seconds) 

# "Instantenious equilibrium R vs D"
R_conc = la0[-1]    # concentration of active chains
D_conc = mu0[-1]    # concentration of dormant chains

#! #######################################################################################################################
#TODO: Define the initial composition of cocatalysts:
max_branches = 3 # Number of branches of the most branched cocatalyst molecule #!  Define the number of max branches and the inlet composition
D0_composition = np.zeros(max_branches)                      
D0_composition[0]      = 0#0.15        #0.2                   # Fraction of chains (branches) in linear cocatalysts
D0_composition[1]      = 0#0.33         # 0.3                 # Fraction of chains in cocatalysts with 2 branches
# D0_composition[2]      = 0.05                               # Input something if we have 4-branched cocatalysts etc..
D0_composition[-1]      = 1 - np.sum(D0_composition[:-1])     # Fraction of chains in cocatalysts with x branches
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
process_pars = [D_conc, R_conc, ga0]

# Find the optimal Nx
D0_composition, Nx, D_round, R_round = find_Nx(D0_composition, Nx_pars, process_pars)

#* Build matrixes for D and R chains for the MC simulation
D, R, G = build_matrixes_forMC(D0_composition, Nx, R_round, max_branches)

#* Convert deterministic kinetic rates to stochastic (MC)
V = Nx / (N_A*(D_conc + R_conc + ga0[-1]))        # Simulated volume, m3 #! terminated chains incl.
k_p_MC   = k_p  / (V*N_A)        # Propagation, 1/s
k_d_MC   = k_d                   # Depropagation, 1/s
k_s_MC   = k_s  / (V*N_A)        # Chain transfer, 1/s
k_te_MC  = k_te / (V*N_A)        # Transesterification, 1/s
k_de_MC  = k_de                  # Scission, 1/s

#* Convert specie concetrations to number values for MC
M_MC    = M  * N_A * V                          # Number of monomer molecules, -
C_MC    = C  * N_A * V                          # Number of catalyst molecules, -
A_MC    = A  * N_A * V                          # Number of acid molecules, -

#* Initiate vectors for output
reac_num  = 12              # Number of elementary reactions considered, used in random selection
Rates_out = np.zeros(reac_num)  # The reaction rates
R_out     = np.zeros(1)         # The active chains (radicals)
D_out     = np.zeros(1)         # The dormant chains
G_out     = np.zeros(1)         # The terminated chains
Mn_out    = np.zeros(1)         # The number-average molecular weight of polymer
Mw_out    = np.zeros(1)         # The weight-average molecular weight of polymer
t_out     = np.zeros(1)         # The reaction time in simulation, s
out_idx   = 0                   # With fq for registering the data from the model in the while loop 
current_time_counter   = 1                   # For current simulation duration measuring
suma_n_tot = np.zeros(1)        # The total number of polymer chains

#* Pre-process concentration profiles (cur~current conc.) from the ODE model
C_cur = C_MC[-1]                                        # Catalyst concentration considered constant 
A_cur = A_MC[-1]                                        # Acid concentration considered constant 
M_fit = interpolate.interp1d(t, M_MC, kind='linear')    # Fit monomer profile as piecewise linear, later used in the main loop 

#* Main MC simulation loop 
#? line 273 in the unstructured model



###########################################################################################
###########################################################################################
total_time = process_time() - total_time
print(f"Total processor time taken to run the whole main_MC_branched: {total_time} seconds")