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

# Plot the results and save them to a .csv and .npy file
# plot_deterministic_results( solution, ModelPars )
