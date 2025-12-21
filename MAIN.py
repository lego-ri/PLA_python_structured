"""
 Batch ring-opening polymerization of L,L-lactide
 Rewritten after Yu Y., Storti G., Morbidelli M., Ind. Eng. Chem. Res,
                 2011, 50, 7927-7940
 by Alexandr Zubov, DTU Chemical Engineering
 Last update: 2025 by Richard Lego UCT Prague
"""

#* Load necessary 3rd party modules and libraries 
from model_pars import ModelPars            # Model parameters
from model_eqs import model_eqs             # Model equations
from plot_deterministic_results import plot_deterministic_results       # Plotting function and saving to .csv and .npy files
from scipy.integrate import solve_ivp       # For ODE solving
from measure_time import measure_time       # Decorator for time measurements
from time import process_time           # To measure the time taken by the simulation
from time import time
from find_Nx import find_Nx
from build_matrixes_forMC import build_matrixes_forMC
from monte_carlo_algorithm import monte_carlo_algorithm
from plot_MC_results import plot_MC_results
from deterministic_post_proces import deterministic_post_proces
from D0_compositions import get_process_viable_Nx
from D0_compositions  import find_Nx_match
from D0_compositions  import map_all_compositions
import numpy as np

total_time = process_time()      # Start the timer to measure the total run time of the file
total_total_time = time()        # Start the timer to measure the total run time of the file
print("----------------------------------------------------------------------------")
print("PLA branched polymerization model has started.")
print("A deterministic ODE and a stochastic Monte Carlo algorithms will now be implemented...")
#**************************************************************************************************************************************************************************************************************************************************************
#*    DEFINE THE CO-CATALYST COMPOSITION
#**************************************************************************************************************************************************************************************************************************************************************
max_branches = 3                                             # Number of branches of the most branched cocatalyst molecule
D0_composition = np.zeros(max_branches)                      
#TODO#######################################################################################################################
#TODO: Define the initial composition of cocatalysts:
# good to test are [0.4, 0.2, 0.4], [0.2, 0, 0.8]
D0_composition[0]      = 0.25#0.2#                               # Fraction of primary alcohol cocatalysts
D0_composition[1]      = 0.65#0.2                               # Fraction of secondary alcohol cocatalysts
# D0_composition[2]      = 0.4#0.33#0.33                               # Fraction of tertiary alcohol cocatalysts
#TODO#######################################################################################################################
#***********************************************************************************************************
D0_composition[-1]      = 1 - np.sum(D0_composition[:-1])     # Fraction of max_branches alcohol cocatalysts
print(f"User input D0_composition is: {D0_composition}")

# Check the user input composition
if max_branches != len(D0_composition):
    print('Error: The defined max_branched co-catalyst number of branches does not match the length of the co-catalyst composition D0_composition array!!!')
    exit()  

if D0_composition[-1] < 0: 
    print('Error: The initial composition of co-catalysts does not sum up to 1!!!')
    exit()  
    
if not np.isclose(np.sum(D0_composition), 1.0, atol=1e-8):
    raise ValueError("D0_composition must sum to 1")

#**************************************************************************************************************************
#*    DETERMINISTIC MODEL
#**************************************************************************************************************************
print("\n----------------------------------------------------------------------------")
print("Deterministic model:")

#* Apply initial conditions
small_number = 1e-7    # Define the small_number
M0 = ModelPars.M0      # Monomer concentration
C0 = ModelPars.C0      # Catalyst concentration
A0 = ModelPars.A0      # Acid concentration
la00 = small_number    # Active chains 0th moment (concentration)
la10 = small_number    # Active chains 1st moment
la20 = small_number    # Active chains 2nd moment
mu00 = ModelPars.D0    # Dormant chains 0th moment (concentration)
mu10 = small_number    # Dormant chains 1st moment
mu20 = small_number    # Dormant chains 2nd moment
ga00 = small_number    # Terminated chains 0th moment (concentration)
ga10 = small_number    # Terminated chains 1st moment
ga20 = small_number    # Terminated chains 2nd moment


# Pack initial conditions 
y0 = [M0, C0, A0, la00, la10, la20, mu00, mu10, mu20, ga00, ga10, ga20]

#* Integration of model equations
# Set the integration limits
tbeg = 0               # simulation start, hours
tbeg = tbeg * 3600     # hours --> seconds conversion
tend = 0.3             # simulation end, hours
tend = tend * 3600     # hours --> seconds conversion
t_span = [tbeg, tend]  # time interval for the ODE integration

# ODE numerical deterministic solving                    
@measure_time
def solve_ode_system():
    # Use solve_ivp to solve the ODE system
    solution = solve_ivp(lambda t, y: model_eqs(t, y, ModelPars), t_span, y0, method='BDF') 
    # For a stiff problem, implicit methods might be better than the used RK23 like BDF or Radau?
    return solution

solution = solve_ode_system() # Solve the ODE system

#* Deterministic model - solution
# Extract results from the main_process
t = solution.t  # Reaction time, s
y = solution.y  # All the state variables at each time step t

# Extract the State variables from solution
M = y[0]     # Monomer concentration, mol/m3
C = y[1]     # Catalyst concentration, mol/m3
A = y[2]     # Acid concentration, mol/m3
la0 = y[3]   # 0th moment of active chains, mol/m3
la1 = y[4]   # 1st moment of active chains, mol/m3
la2 = y[5]   # 2nd moment of active chains, mol/m3
mu0 = y[6]   # 0th moment of dormant chains, mol/m3
mu1 = y[7]   # 1st moment of dormant chains, mol/m3
mu2 = y[8]   # 2nd moment of dormant chains, mol/m3
ga0 = y[9]   # 0th moment of terminated chains, mol/m3
ga1 = y[10]  # 1st moment of terminated chains, mol/m3
ga2 = y[11]  # 2nd moment of terminated chains, mol/m3

#* Deterministic model - post processing
# Conversion of monomer (%)
conv = (M0 - M) / M0 * 100
print("----------------------------------------------------------------------------")
print("Deterministic model results")
print(f'Conversion = {conv[-1]:.2f} [%]')   # Print the conversion of monomer

#!-----------------wrong approach -----------------------!
avg_bran_in_macromol = sum(D0_composition[i]*(i+1) for i in range(max_branches))
# Number-average molecular weight of polymer
Mn_ODE_1 = ModelPars.MW * (la1 + mu1 + ga1) / (ga0 + (la0 + mu0)  * (sum(D0_composition[i]/(i+1) for i in range(max_branches))))  

# Weight-average molecular weight of polymer
Mw_ODE_2 = ModelPars.MW * (ga2 + ((((mu2 + la2)/(la0+mu0))**0.5 )*avg_bran_in_macromol)**2*((la0+mu0)/avg_bran_in_macromol)) / (la1 + mu1 + ga1) 
Mw_ODE_3 = ModelPars.MW * (ga2 + (((mu2/mu0)**0.5 )*avg_bran_in_macromol)**2*((mu0)/avg_bran_in_macromol) + (((la2/la0)**0.5 )*avg_bran_in_macromol)**2*((la0)/avg_bran_in_macromol) ) / (la1 + mu1 + ga1) 
Mw_ODE_1 = ModelPars.MW * (ga2 + (mu2 + la2) * avg_bran_in_macromol) / (la1 + mu1 + ga1) 
#!---------------------------------------------------------

#*************************************************************************************************************
Mn_ODE, Mw_ODE = deterministic_post_proces(y,D0_composition,'molecule',small_number)

# Print final conversion, Mw, and PDI
print(f'Mw_ODE = {Mw_ODE[-1]:.2f} [kg/mol]')
print(f'PDI_ODE = {Mw_ODE[-1] / Mn_ODE[-1]:.2f}') # PDI = polydispersity index


#**************************************************************************************************************************
#*    MONTE CARLO MODEL
#**************************************************************************************************************************
print("\n----------------------------------------------------------------------------")
print("Monte Carlo model:")

#* "Instantenious equilibrium R vs D"
# The concentration of dormant and active chains becomes constant very quickly,
# thus can be assumed constant from the beggining throughout the MC simulation.
R_conc = la0[-1]    # concentration of active chains
D_conc = mu0[-1]    # concentration of dormant chains
    
#* Check if the model can be run with this initial composition and find the corresponding Nx
# Define the parameters for the optimal Nx search
min_Nx = 500                         # Minimal Nx
max_Nx = 6000                        # Maximal Nx    
max_difference_RD_dec_round = 1e-2   # Maximal difference between rounded and non-rounded number of D,R
eps_fraction = 1e-3                  # Margin for finding close fractions to the defined inlet composition

# Validate that min_Nx and max_Nx are (near) integers
if not isinstance(min_Nx, int) or not isinstance(max_Nx, int):
    raise ValueError("min_Nx and max_Nx must be integer values")


# Pack pars for Nx search
Nx_pars = [min_Nx, max_Nx, max_difference_RD_dec_round, eps_fraction, max_branches] 
process_pars = [D_conc, R_conc, ga0, t]

#TODO::::::::::::::::::::::::::::::::::::::::::::::::::;
# Find the optimal Nx
#! old below
D0_composition, Nx, D_round, R_round = find_Nx(D0_composition, Nx_pars, process_pars)
#! old above
# Get the list of Nx values that satisfy R/D equilibrium
viable_Nx_array = get_process_viable_Nx(Nx_pars, process_pars)
print(f"Found {len(viable_Nx_array)} viable Nx values.\n")

# --- Find Match for Specific Composition (User Request) ---
matched_Nx = find_Nx_match(D0_composition, viable_Nx_array)
if matched_Nx:
    print(f"SUCCESS: Composition {D0_composition} is valid with Nx = {matched_Nx}\n")
else:
    print(f"FAILURE: Composition {D0_composition} not found in current Nx range.\n")

# We pass the already calculated viable_Nx_array into this function
full_dictionary = map_all_compositions(viable_Nx_array, max_branches=3)

print(f"Generated map for {len(full_dictionary)} Nx values.")
# Example print of first found
first_key = list(full_dictionary.keys())[0]
print(f"Example: Nx={first_key} supports {len(full_dictionary[first_key])} compositions, 1st one is: {full_dictionary[first_key][0]})")

#TODO:::::::::::::::::::::::::::::::::::::::::::::::::

# Build matrixes for D and R chains for the MC simulation
D, R, G, total_num_D_mol = build_matrixes_forMC(D0_composition, Nx, R_round, max_branches)

#****************************************************************************************
#*          Main MC simulation loop 
#****************************************************************************************
# Pack parameters for the MC simulation
mc_pars = [Nx, M, C, A, R, D, G, total_num_D_mol]

# Run the MC simulation
t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, R, D, G, out_idx = monte_carlo_algorithm(mc_pars, process_pars, ModelPars, y)
# print(f"out_idx is: {out_idx}")

#* Post processing MC
# Pack MC results for plotting
MC_output = [t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, G]
plot_pars = [t, Mn_ODE_1, Mw_ODE_1, Mw_ODE_2, Mw_ODE_3, Mw_ODE, Mn_ODE]

# Plotting
plot = plot_MC_results(MC_output, plot_pars)


#* -----------------------------------------------------------------------------------
total_time = process_time() - total_time
total_total_time = time() - total_total_time
print("\n----------------------------------------------------------------------------")
print(f"Total time taken to run the Monte Carlo simulation: {total_total_time} seconds")
print(f"Total processor time taken to run the whole program: {total_time} seconds")