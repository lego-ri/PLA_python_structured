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
# from tester import find_Nx
from build_matrixes_forMC import build_matrixes_forMC
from monte_carlo_algorithm import monte_carlo_algorithm
from plot_MC_results import plot_MC_results
import numpy as np

total_time = process_time()      # Start the timer to measure the total run time of the file
total_total_time = time()        # Start the timer to measure the total run time of the file

#**************************************************************************************************************************************************************************************************************************************************************
#*    USER INPUT
#**************************************************************************************************************************************************************************************************************************************************************

max_branches = 3                                             # Number of branches of the most branched cocatalyst molecule
D0_composition = np.zeros(max_branches)                      

#TODO#######################################################################################################################
#TODO: Define the initial composition of cocatalysts:
D0_composition[0]      = 0.2#0.25#0.15                               # Fraction of chains (branches) in linear cocatalysts
D0_composition[1]      = 0.2#0.33                               # Fraction of chains in cocatalysts with 2 branches
# D0_composition[2]      = 0.4#0.33#0.33                               # Fraction of chains in cocatalysts with 2 branches
#TODO#######################################################################################################################

D0_composition[-1]      = 1 - np.sum(D0_composition[:-1])     # Fraction of chains in cocatalysts with max_branches branches
print(f"User input D0_composition is: {D0_composition}")

# Check the user input composition
if max_branches != len(D0_composition):
    print('Error: The number of branches does not match the length of the D0_composition array!!!')
    exit()  

if D0_composition[-1] < 0: 
    print('Error: The initial composition does not sum up to 1!!!')
    exit()  
    
if not np.isclose(np.sum(D0_composition), 1.0, atol=1e-8):
    raise ValueError("D0_composition must sum to 1")


#**************************************************************************************************************************
#*    DETERMINISTIC MODEL
#**************************************************************************************************************************

print("\nDeterministic model:")

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
    solution = solve_ivp(lambda t, y: model_eqs(t, y, ModelPars, D0_composition, max_branches), t_span, y0, method='BDF') 
    # For a stiff problem, implicit methods might be better than the used RK23 like BDF or Radau?
    return solution

solution = solve_ode_system() # Solve the ODE system

#* Deterministic model - solution
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

#* Deterministic model - post processing
# Conversion of monomer (%)
conv = (M0 - M) / M0 * 100

#TODO: 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
avg_bran_in_macromol = sum(D0_composition[i]*(i+1) for i in range(max_branches))
# Number-average molecular weight of polymer
Mn_ODE = ModelPars.MW * (la1 + mu1 + ga1) / (ga0 + (la0 + mu0)  * (sum(D0_composition[i]/(i+1) for i in range(max_branches))))  

# Weight-average molecular weight of polymer
Mw_ODE_2 = ModelPars.MW * (ga2 + ((((mu2 + la2)/(la0+mu0))**0.5 )*avg_bran_in_macromol)**2*((la0+mu0)/avg_bran_in_macromol)) / (la1 + mu1 + ga1) 
Mw_ODE_3 = ModelPars.MW * (ga2 + (((mu2/mu0)**0.5 )*avg_bran_in_macromol)**2*((mu0)/avg_bran_in_macromol) + (((la2/la0)**0.5 )*avg_bran_in_macromol)**2*((la0)/avg_bran_in_macromol) ) / (la1 + mu1 + ga1) 
Mw_ODE = ModelPars.MW * (ga2 + (mu2 + la2) * avg_bran_in_macromol) / (la1 + mu1 + ga1) 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Print final conversion, Mw, and PDI
print(f'Conversion = {conv[-1]:.2f} [%]')           # [-1] is the last element of the array,
print(f'Mw = {Mw_ODE[-1]:.2f} [kg/mol]')                 #  :.2f formats the number to 2 decimal places
print(f'PDI = {Mw_ODE[-1] / Mn_ODE[-1]:.2f}')   # Polydispersity index ("Zp"), width of the distribution

#*************************************************************************************************************
# === Compute exact molecular Mn and Mw under the stated assumptions ===
# Inputs assumed present: la0, la1, la2, mu0, mu1, mu2, ga0, ga1, ga2 (arrays), 
#                        D0_composition (array), ModelPars.MW

# normalize input (so we work with fractions)
mode = "chain"
p_raw = np.array(D0_composition, dtype=float)
p_sum = np.sum(p_raw)
p_norm = p_raw / p_sum
f = np.arange(1, len(p_norm) + 1)  # functionalities: 1,2,3,...

if mode == 'molecule':
    q = p_norm
    f_avg = float(np.sum(q * f))
    inv_f_avg = 1.0 / f_avg
    factor_sum = inv_f_avg   # N_mol_att = N_att * (1/f_avg) = N_att * factor_sum
elif mode == 'chain':
    # input p_norm are p_f (fraction of branches)
    p = p_norm
    # use sum(p_f / f) which equals 1/f_avg
    factor_sum = float(np.sum(p / f))
    # (optional) compute implied f_avg for diagnostics:
    # f_avg_implied = 1.0 / factor_sum
else:
    raise ValueError("mode must be 'chain' or 'molecule'")

# Branch-level totals
N_att = la0 + mu0
S1_att = la1 + mu1
S2_att = la2 + mu2

# Per-branch raw moments
N_att_safe = np.maximum(N_att, small_number)
m1_att = S1_att / N_att_safe
m2_att = S2_att / N_att_safe

# Compute number of attached molecules per volume:
# N_mol_att = N_att / f_avg  (if molecule-mode), equivalently N_att * factor_sum
N_mol_att = N_att * factor_sum

# Per-attached-molecule moments (same derivation as before; uses global branch moments)
# mu1_mol_att = f_avg * m1_att  -- if we need f_avg and mode=='chain', we can compute f_avg implied
# But we can get mu1_mol_att via: mu1_mol_att = (N_att * m1_att) / N_mol_att = S1_att / N_mol_att
# Simpler and robust:
mu1_mol_att = np.where(N_mol_att > small_number, S1_att / N_mol_att, 0.0)

# For mu2_mol_att it's safer to use the formula based on f_avg and E[f(f-1)].
# If mode == 'molecule' we can compute f_avg and Eff1 directly; if 'chain' we can recover q_f:
if mode == 'molecule':
    q = p_norm
else:  # 'chain'
    p = p_norm
    # recover q from p: q_f = p_f / f  / sum(p_g / g)  -> denominator == factor_sum
    q = (p / f) / factor_sum

f_avg = float(np.sum(q * f))
E_ff1 = float(np.sum(q * f * (f - 1)))

mu2_mol_att = f_avg * m2_att + (m1_att ** 2) * E_ff1

# Total molecular counts and moments per volume
N_g = ga0
S1_g = ga1
S2_g = ga2

N_mol_total = N_mol_att + N_g
S1_mol_total = N_mol_att * mu1_mol_att + S1_g
S2_mol_total = N_mol_att * mu2_mol_att + S2_g

N_mol_safe = np.maximum(N_mol_total, small_number)
S1_mol_safe = np.maximum(S1_mol_total, small_number)

DP_n = S1_mol_total / N_mol_safe
DP_w = S2_mol_total / S1_mol_safe

Mn_ODE_4 = ModelPars.MW * DP_n
Mw_ODE_4 = ModelPars.MW * DP_w

print(f'Mw_ODE_4 = {Mw_ODE_4[-1]:.2f} [kg/mol]')
print(f'PDI_ODE_4 = {Mw_ODE_4[-1] / Mn_ODE_4[-1]:.2f}')


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
max_Nx = 6000                        # Maximal Nx    
max_difference_RD_dec_round = 1e-2   # Maximal difference between rounded and non-rounded number of D,R
eps_fraction = 1e-3                  # Margin for finding close fractions to the defined inlet composition

# Pack pars for Nx search
Nx_pars = [min_Nx, max_Nx, max_difference_RD_dec_round, eps_fraction, max_branches] 
process_pars = [D_conc, R_conc, ga0, t]

# Find the optimal Nx
D0_composition, Nx, D_round, R_round = find_Nx(D0_composition, Nx_pars, process_pars)

#* Build matrixes for D and R chains for the MC simulation
D, R, G, total_num_D_mol = build_matrixes_forMC(D0_composition, Nx, R_round, max_branches)

#* Main MC simulation loop 
# Pack parameters for the MC simulation
mc_pars = [Nx, M, C, A, R, D, G, total_num_D_mol]
det_results = y


# Run the MC simulation
t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, R, D, G, out_idx = monte_carlo_algorithm(mc_pars, process_pars, ModelPars, det_results)

# print(f"out_idx is: {out_idx}")

#* Post processing MC
# Pack MC results for plotting
MC_output = [t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, G]
plot_pars = [t, Mn_ODE, Mw_ODE, Mw_ODE_2, Mw_ODE_3, Mw_ODE_4, Mn_ODE_4]

# Plotting
plot = plot_MC_results(MC_output, plot_pars)


def diagnostics_branching(
    la0, la1, la2, mu0, mu1, mu2, ga0, ga1, ga2,
    D0_composition, t, t_out, Mn_out, Mw_out, ModelPars
):
    """
    Diagnostics for deterministic vs Monte Carlo branching consistency.
    Prints checks for:
        1. Indexing of D0_composition
        2. Functionality distribution integrity
        3. Branch-level mass balance
        4. Molecular-level mass balance
        5. Correct molecule count per volume
        6. ODE vs MC Mn, Mw alignment
    """

    print("\n==============================")
    print(" BRANCHING DIAGNOSTICS REPORT")
    print("==============================\n")

    small_number = 1e-30

    # =======================================================
    # 1) Functionality indexing and distribution
    # -------------------------------------------------------
    f = np.arange(1, len(D0_composition) + 1)  # +1 because index 0 = f=1
    p_raw = np.array(D0_composition, dtype=float)
    p_sum = p_raw.sum()

    print("1) INDEXING CHECK")
    print("------------------")
    print(f"Functionality values f      = {f}")
    print(f"User D0_composition (raw)   = {p_raw}")
    print(f"Sum p_raw                   = {p_sum:.6f}")

    if not np.isclose(p_sum, 1.0, atol=1e-8):
        print("!! WARNING: D0_composition does not sum to 1. Normalizing...")
    p = p_raw / (p_sum + small_number)

    f_avg = np.sum(p * f)
    E_ff1 = np.sum(p * f * (f - 1))

    print(f"Normalized p                = {p}")
    print(f"Average functionality f_avg = {f_avg:.6f}")
    print(f"E[f(f-1)]                   = {E_ff1:.6f}")
    print()

    if f_avg <= 0:
        raise ValueError("ERROR: Average functionality is zero or negative. Check D0_composition.")

    # =======================================================
    # 2) Branch-level quantities
    # -------------------------------------------------------
    print("2) BRANCH-LEVEL CHECKS")
    print("------------------------")
    N_att = la0 + mu0
    S1_att = la1 + mu1
    S2_att = la2 + mu2

    print(f"Total attached branches at end = {N_att[-1]:.6e}")

    # =======================================================
    # 3) Mass balance for monomer units in chains
    # -------------------------------------------------------
    print("\n3) BRANCH MASS BALANCE")
    print("------------------------")
    branch_mass = la1 + mu1 + ga1  # total monomer units in all chains
    # should match initial monomers consumed (up to numerical error)
    # can't check directly without M0, so check consistency only:
    print("Max(la1+mu1+ga1 - S1_att - S1_g) == 0?")

    diff_branch_mass = np.max(np.abs((la1 + mu1 + ga1) - (S1_att + ga1)))
    print(f"Max difference = {diff_branch_mass:.6e}  (should be near 0)")
    print()

    # =======================================================
    # 4) TERMINATED CHAIN CHECK (should be linear)
    # -------------------------------------------------------
    print("4) TERMINATED CHAIN CHECK")
    print("--------------------------")
    # If all ga molecules are linear, per-molecule DP = S1_g / ga0
    ga_dp = np.where(ga0 > 0, ga1 / (ga0 + small_number), 0)

    print(f"Final average DP of terminated chains = {ga_dp[-1]:.4f}")
    print("If this is NOT linear DP, your MC model may generate multi-arm ga.")
    print()

    # =======================================================
    # 5) Molecular counts — attached and total
    # -------------------------------------------------------
    print("5) MOLECULAR COUNT CHECKS")
    print("---------------------------")

    N_mol_att = N_att / f_avg
    N_mol_total = N_mol_att + ga0

    print(f"Final attached molecules       = {N_mol_att[-1]:.6e}")
    print(f"Final terminated molecules     = {ga0[-1]:.6e}")
    print(f"Final TOTAL molecules (ODE)    = {N_mol_total[-1]:.6e}")
    print()

    # =======================================================
    # 6) MOLECULAR MASS BALANCE CHECK (critical)
    # -------------------------------------------------------
    print("6) MOLECULAR MASS CONSISTENCY")
    print("-------------------------------")

    # Compute molecular moments exactly
    N_att_safe = np.where(N_att > 0, N_att, small_number)
    m1_att = S1_att / N_att_safe
    m2_att = S2_att / N_att_safe

    # attached molecule first and second moments
    mu1_mol_att = f_avg * m1_att
    mu2_mol_att = f_avg * m2_att + E_ff1 * (m1_att ** 2)

    S1_mol_total = N_mol_att * mu1_mol_att + ga1
    S2_mol_total = N_mol_att * mu2_mol_att + ga2

    diff_S1 = np.max(np.abs(S1_mol_total - (la1 + mu1 + ga1)))
    diff_S2 = np.max(np.abs(S2_mol_total - (la2 + mu2 + ga2)))

    print(f"Max |S1_mol_total - chain S1| = {diff_S1:.6e}  (must be near 0)")
    print(f"Max |S2_mol_total - chain S2| = {diff_S2:.6e}  (must be near 0)")
    print()

    # =======================================================
    # 7) ODE vs MC COMPARISON
    # -------------------------------------------------------
    print("7) ODE vs MC COMPARISON")
    print("-------------------------")

    # align times
    closest_idx = [np.argmin(np.abs(t_out - τ)) for τ in t]
    Mn_mc_aligned = Mn_out[closest_idx]
    Mw_mc_aligned = Mw_out[closest_idx]

    Mn_mc_safe = np.where(Mn_mc_aligned > 0, Mn_mc_aligned, small_number)
    Mw_mc_safe = np.where(Mw_mc_aligned > 0, Mw_mc_aligned, small_number)

    # compute deterministic exact Mn and Mw
    DP_n = S1_mol_total / (N_mol_total + small_number)
    DP_w = S2_mol_total / (S1_mol_total + small_number)

    Mn_ode = ModelPars.MW * DP_n
    Mw_ode = ModelPars.MW * DP_w

    ratio_Mn = Mn_ode / Mn_mc_safe
    ratio_Mw = Mw_ode / Mw_mc_safe

    print(f"Final Mn_ODE / Mn_MC = {ratio_Mn[-1]:.4f}")
    print(f"Final Mw_ODE / Mw_MC = {ratio_Mw[-1]:.4f}")
    print("\nIf Mw matches but Mn does not -> molecule counts are wrong.\n")
    print("If Mn matches but Mw does not -> second moment calculation wrong.\n")
    print("If neither matches -> functionality distribution or MC chemistry differs.\n")

    print("Diagnostics complete.\n")

diagnostics_branching(
    la0, la1, la2, mu0, mu1, mu2, ga0, ga1, ga2,
    D0_composition, t, t_out, Mn_out, Mw_out, ModelPars
)



########################################################################
total_time = process_time() - total_time
total_total_time = time() - total_total_time
print(f"Total time taken to run the Monte Carlo simulation: {total_total_time} seconds")
print(f"Total processor time taken to run the whole program: {total_time} seconds")