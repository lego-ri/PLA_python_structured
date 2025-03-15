""" 
    Check if the specified initial composition has a corresponding number of elements Nx.
    If so, return that Nx.
    Otherwise find the closest possivle initial composition which has a corresponding Nx and return these.
"""

import numpy as np
from measure_time import measure_time

@measure_time
def find_Nx(D0_composition, Nx_pars, process_pars):

    #* Define the parameters for the optimal Nx search
    min_Nx = Nx_pars[0]                        # minimal Nx
    max_Nx = Nx_pars[1]                        # maximal Nx    
    max_difference_RD_dec_round = Nx_pars[2]   # maximal difference between rounded and non-rounded D,R
    
    #* Unpack the data from the deterministic model 
    D_conc = process_pars[0]                     # Dormant concentration, mol/m3
    R_conc = process_pars[1]                     # Active chain concentration, mol/m3
    ga0 = process_pars[2]                        # Terminated chain concentration, mol/m3

    N_A = 6.022140858e23     # Avogadro number, 1/mol

    #* Find viable Nx values for our system (R-D instanteneous equilibrium)
    difference_RD_dec_round = float('inf')       # Define the starting difference_RD_dec_round between decimal and round numbers of D which will be calculated later on
    Nx_viable = np.array([])        # Define an array for viable Nx values
    D_round_viable = np.array([])   # Define an array for number of dormant chains for each viable Nx 
    R_round_viable = np.array([])   # Define an array for number of active chains for each viable Nx

    # Iterate over Nx from a defined interval to find the viable Nx values
    for Nx in range(min_Nx, max_Nx + 1):
        V = Nx / (N_A*(D_conc + R_conc + ga0[-1]))        # Simulated volume, m3 #! terminated chains incl.
        D_MC    = (D_conc * N_A * V)                      # Number of dormant chains
        R_MC    = (R_conc * N_A * V)                      # Number of active chains
        # D_MC2 = (D_conc/(D_conc+R_conc+ga0[-1])*Nx)   # alternative way of computing D_MC (from the Fortran code)

        #* Compare the counted R,D with the rounded values
        # Recalcualte the number of dormant and active chains acording to Nx, mantaining the ratio between R and D
        D_decimal = (Nx-1)/(R_MC/D_MC + 1)  # Number of dormant chains (decimal value) 
        R_decimal = (Nx-1)/(D_MC/R_MC + 1)  # Number of active chains (decimal value)
        D_round = round(D_decimal)          # Number of dormant chains (integer)
        R_round = round(R_decimal)          # Number of active chains (integer)
        
        difference_RD_dec_round = abs(D_decimal - D_round) + abs(R_decimal - R_round)    # Differences between decimal and whole number (want to minimise)

        # Save the possible Nx and D_round and R_round values
        if difference_RD_dec_round < max_difference_RD_dec_round:
            Nx_viable = np.append(Nx_viable, Nx)
            D_round_viable = np.append(D_round_viable, D_round)
            R_round_viable = np.append(R_round_viable, R_round)        
    print(f"Viable Nx are: {Nx_viable}") # From these we must choose one Nx to match our defined composition

    #* Test if the user defined initial composition has a corresponding viable Nx
    found_initial_Nx = False # True if the user defined initial composition has a corresponding viable Nx
    # Iterate over the viable Nxs
    for i in range(len(Nx_viable)):
        valid = True
        # Iterate over the fractions in the initial composition
        for j in range(len(D0_composition)):
            # check if the number of branches belonging to an x-brancehd molecule type is divisible by x
            if not ( (Nx_viable[i]-1) * D0_composition[j] % (j + 1) == 0): 
                valid = False
                break
        # If a valid Nx is found, break the loop and save the data for future use    
        if valid:
            Nx = int(Nx_viable[i])
            D_round = int(D_round_viable[i])
            R_round = int(R_round_viable[i])
            print("Original composition is valid with Nx:", Nx)
            print(f"D_round is: {D_round}, R_round is: {R_round}, G is set to 1")
            print("Composition:", D0_composition)
            found_initial_Nx = True
            break

    #* If the initial composition cannot be modelled, find the closest modellable compositions 
    if not found_initial_Nx: 
        print("User input D0_composition cannot be modelled with available Nx, searching for a close match...")
        Nx_viable_no_G = Nx_viable - 1  # Substract 1 terminated chain to get the total number of D and R initial
        Inlet_comp_results = {}         # Define a dictionary to save the initial compositions and the Nx they belong to
        Inlet_comp_sorted_results = {}  # Define a dictionary for sorted results with the closest matches at the top
        eps_fraction = 0.001            # Margin for finding close fractions to the defined inlet comp
        
        #* Step 1:
        # Iterate over possible Nx values to find close initial compositions
        for Nx in Nx_viable_no_G:
            valid_compositions = [] # Define a list for saving possible initial compositions for current Nx
        
            # Iterate over possible values for the part divisible by 3 (must be > 0) (3 branched molecules)
            for x in range(0, int(Nx), 3):  
                # Iterate over possible values for the part divisible by 2 (must be > 0) (2 branched molecules)
                for y in range(0, int(Nx) - x, 2):  
                    z = Nx - x - y  # Compute the remaining part (linear molecules)

                    if z > 0:   # Ensure all parts are greater than 0
                        # Calculate the fraction of given cocatalysts D0
                        frac_3 = x/Nx   # 3 branched molecules
                        frac_2 = y/Nx   # 2 branched molecules
                        frac_1 = z/Nx   # linear molecules
                        
                        initial_fractions = [frac_1, frac_2, frac_3] # Pack the fractions into a composition
                        
                        # Check if the initial_fractions are close (+-eps) to the user defined D0_composition
                        is_frac_valid = True
                        for i in range(len(D0_composition)):
                            if not D0_composition[i]-eps_fraction <= initial_fractions[i] <= D0_composition[i]+eps_fraction:
                                is_frac_valid = False
                                break
                        if is_frac_valid:
                            valid_compositions.append((initial_fractions)) # Save compositions close to the wanted one
        
            Inlet_comp_results[Nx+1] = valid_compositions # Save the valid compositions for each Nx to the dictionary
        
        #* Step 2: Sort the compositions by closeness to D0_composition
        for Nx, compositions in Inlet_comp_results.items():
            # Sort compositions by closeness to D0_composition
            sorted_compositions = sorted( compositions, key=lambda comp: sum(abs(comp[i] - D0_composition[i]) for i in range(len(D0_composition))) )
            Inlet_comp_sorted_results[Nx] = sorted_compositions # Save the sorted compositions to the dictionary

        #* Step 3: Find the overall best match across all Nx values
        best_Nx = None
        best_match = None
        best_error = float('inf')  # Initialize with a high error

        # For each Nx iterate over the first (best for that Nx) composition to find the best match to the desired inlet composition
        for Nx, sorted_compositions in Inlet_comp_sorted_results.items():
            if sorted_compositions:  # Ensure there is at least one valid composition
                top_composition = sorted_compositions[0]  # Closest match for this Nx
                error = sum(abs(top_composition[i] - D0_composition[i]) for i in range(3))

                if error < best_error:  # Check if this is the new best overall match
                    best_error = error
                    best_match = top_composition
                    best_Nx = Nx

        #* Save and output the overall best match to the user input D0_composition
        if best_match:
            D0_composition = best_match             # Save the found D0_composition
            Nx = int(best_Nx)                       # Save the found Nx                          
            Nx_index = np.where(Nx_viable == Nx)[0][0] # [0] extracts the indices, secon [0] gets the first indice (there is alway just one, but this is to prevent deprecation warning on next line where we convet it to an int)
            D_round, R_round = int(D_round_viable[Nx_index]), int(R_round_viable[Nx_index]) # Save the found D and R corresponding to Nx
            print(f"Overall Best Match Found:")
            print(f"Nx = {Nx}")
            print(f"D_round is: {D_round}, R_round is: {R_round}, G is set to 1")
            print("Composition:", D0_composition)
            print(f"Total Deviation from Target: {best_error}")
        else:
            print("No valid compositions found.")
            exit()\
                
    return D0_composition, Nx, D_round, R_round
