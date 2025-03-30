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
from find_Nx import find_Nx
from build_matrixes_forMC import build_matrixes_forMC
from monte_carlo_algorithm import monte_carlo_algorithm
from plot_MC_results import plot_MC_results
import numpy as np

#TODO: pro rozsireny zmenit momenty

total_time = process_time()      # Start the timer to measure the total run time of the file

#**************************************************************************************************************************************************************************************************************************************************************
#*    USER INPUT
#**************************************************************************************************************************************************************************************************************************************************************

max_branches = 3                                              # Number of branches of the most branched cocatalyst molecule
D0_composition = np.zeros(max_branches)                      

#TODO#######################################################################################################################
#TODO: Define the initial composition of cocatalysts:
D0_composition[0]      = 0.5#0.15                               # Fraction of chains (branches) in linear cocatalysts
D0_composition[1]      = 0.5#0.33                               # Fraction of chains in cocatalysts with 2 branches
#TODO#######################################################################################################################

D0_composition[-1]      = 1 - np.sum(D0_composition[:-1])     # Fraction of chains in cocatalysts with max_branches branches
print(f"User input D0_composition is: {D0_composition}")

# Check if the given fractions for cocatalyst composition sum up to 1
if D0_composition[-1] < 0: 
    print('Error: The initial composition does not sum up to 1!!!')
    exit()  

#**************************************************************************************************************************
#*    DETERMINISTIC MODEL
#**************************************************************************************************************************

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
tend = 0.2             # simulation end, hours
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

#* Deterministic model - post processing
# Extract results from the main_process
t = solution.t  # Reaction time, s
y = solution.y  # All the state variables at each time step t

# Assign the State variables 
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

# Conversion of monomer (%)
conv = (M0 - M) / M0 * 100

#TODO: DONE
# Number-average molecular weight of polymer
Mn_ODE = ModelPars.MW * (la1 + mu1 + ga1) / (ga0 + (la0 + mu0)  * (np.sum(D0_composition[i]/(i+1) for i in range(max_branches))))  #!!!!!!!!!

# Weight-average molecular weight of polymer
Mw_ODE = ModelPars.MW * (la2 + mu2 + ga2) / (la1 + mu1 + ga1) 

# Print final conversion, Mw, and PDI
print(f'Conversion = {conv[-1]:.2f} [%]')           # [-1] is the last element of the array,
print(f'Mw = {Mw_ODE[-1]:.2f} [kg/mol]')                 #  :.2f formats the number to 2 decimal places
print(f'PDI = {Mw_ODE[-1] / Mn_ODE[-1]:.2f}')   # Polydispersity index ("Zp"), width of the distribution

# Plot the results and save them to a .csv and .npy file
# plot_deterministic_results( solution, ModelPars )


#**************************************************************************************************************************
#*    MONTE CARLO MODEL
#**************************************************************************************************************************

print("\nMonte Carlo model:")

#* "Instantenious equilibrium R vs D"
# The concentration of dormant and active chains becomes constant very quickly,
# thus can be assumed constant from the beggining throughout the MC simulation.
R_conc = la0[-1]    # concentration of active chains
D_conc = mu0[-1]    # concentration of dormant chains

    
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
print(f"Total processor time taken to run the whole program: {total_time} seconds")