"""
 Batch ring-opening polymerization of L,L-lactide
 Rewritten after Yu Y., Storti G., Morbidelli M., Ind. Eng. Chem. Res,
                 2011, 50, 7927-7940
 by Alexandr Zubov, DTU Chemical Engineering
 Last update: March 2025 by Richard Lego UCT Prague
"""
"""
    Uses:
    model_parts - parameters of the model
    model_eqs - system of ODEs that make up the model
    plot_results - function for plotting results and storing necessary info for the MC model
    measure_time - decorator for measuring time taken by a function
"""

#* Load necessary 3rd party modules and libraries 
from model_pars import ModelPars            # Model parameters
from model_eqs import model_eqs             # Model equations
from plot_results import plot_results       # Plotting function and saving to .csv and .npy files
from scipy.integrate import solve_ivp       # For ODE solving
from measure_time import measure_time       # Decorator for time measurements


#* Apply initial conditions
#? pro rozsireny zmenit 0. momenty?? chaiins = x*molekules
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
print(f"Solving the model...")

@measure_time
def solve_ode_system():
    # Use solve_ivp to solve the ODE system
    solution = solve_ivp(lambda t, y: model_eqs(t, y, ModelPars), t_span, y0, method='BDF') 
    # For a stiff problem, implicit methods might be better than the used RK23 like BDF or Radau?
    return solution

solution = solve_ode_system() # Solve the ODE system

#* Plot the results and save them to a .csv and .npy file
plot_results( solution, ModelPars )