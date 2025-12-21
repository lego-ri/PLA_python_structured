import numpy as np
from measure_time import measure_time

@measure_time
def generate_decimal_compositions(n_branches, precision_steps=10):
    """
    Generates all possible lists of length n_branches that sum to 1.0
    using steps of 1/precision_steps (e.g., 0.1).
    
    Returns a list of numpy arrays, e.g., [0.2, 0.3, 0.5]
    """
    # We solve this using integers summing to 10 to avoid floating point drift, 
    # then divide by 10 at the end.
    
    def sums(target, n): # Target is the fractoon (*precision_steps) of branches remaining to be distributed, n is number of slots [defined by max_branches at the beginning] left to fill
        if n == 1:
            yield (target,)
        else:
            for i in range(target + 1):
                for tail in sums(target - i, n - 1):
                    yield (i,) + tail

    integer_compositions = list(sums(precision_steps, n_branches))
    
    # Convert back to fractions (e.g., 2 -> 0.2)
    fraction_compositions = []
    for comp in integer_compositions:
        fraction_compositions.append(np.array([x/precision_steps for x in comp]))
        
    return fraction_compositions


@measure_time
def find_viable_compositions_for_Nx(Nx_pars, process_pars):
    """ 
    Finds viable Nx values based on process parameters, and for each viable Nx,
    finds a list of all possible initial compositions (1 decimal point precision)
    that mathematically satisfy the integer constraints.

    Returns:
        viable_dict (dict): Keys are Nx (int), Values are lists of valid composition arrays.
    """

    #* Define the parameters for the optimal Nx search
    min_Nx = Nx_pars[0]                        
    max_Nx = Nx_pars[1]                        
    max_difference_RD_dec_round = Nx_pars[2]   
    # eps_fraction is not needed here as we are generating exact fractions
    max_branches = Nx_pars[4]
    
    #* Unpack the data from the deterministic model 
    D_conc = process_pars[0]                     
    R_conc = process_pars[1]                     
    ga0 = process_pars[2]                        
    G_conc = ga0[-1]

    N_A = 6.022140858e23     

    # ---------------------------------------------------------
    # PART 0: Generate all theoretical compositions (0.1 steps)
    # ---------------------------------------------------------
    # This generates [0.0, 0.0, 1.0], [0.1, 0.0, 0.9], etc.
    candidate_compositions = generate_decimal_compositions(max_branches, precision_steps=10)

    # ---------------------------------------------------------
    # PART 1: Find viable Nx values (Logic from original code)
    # ---------------------------------------------------------
    Nx_viable = [] 
    
    for Nx in range(min_Nx, max_Nx + 1):
        V = Nx / (N_A*(D_conc + R_conc + G_conc))       
        D_MC    = (D_conc * N_A * V)                      
        R_MC    = (R_conc * N_A * V)                      

        D_decimal = (Nx-1)/(R_MC/D_MC + 1)  
        R_decimal = (Nx-1)/(D_MC/R_MC + 1)  
        D_round = round(D_decimal)          
        R_round = round(R_decimal)          
        
        difference_RD_dec_round = abs(D_decimal - D_round) + abs(R_decimal - R_round)

        if difference_RD_dec_round < max_difference_RD_dec_round:
            Nx_viable.append(int(Nx))
            
    Nx_viable = np.array(Nx_viable)
    print(f"Found {len(Nx_viable)} viable Nx values.")

    # ---------------------------------------------------------
    # PART 2: Cross-check Nx with Compositions
    # ---------------------------------------------------------
    results_dict = {}

    for Nx in Nx_viable: 
        valid_compositions_for_this_Nx = []
        
        for D0_comp in candidate_compositions:
            
            # 1. Calculate weighted average chain length (chains per macromolecule)
            # k is number of branches. Chains = k + 1.
            # sum(Fraction_k * (k+1))
            avg_chain_length = np.sum([D0_comp[k] * (k+1) for k in range(max_branches)])
            
            # 2. Calculate Total Number of Macromolecules
            # (Nx - 1) is total chains available for distribution (1 is terminated G)
            tot_number_of_macromol = (Nx - 1) / avg_chain_length # Nx includes the 1 terminated chain!
            
            # CHECK A: Is total number of macromolecules an integer?
            if not (abs(tot_number_of_macromol - np.round(tot_number_of_macromol)) < 1e-9): #TODO: should be smallert tol?
                continue # Skip this composition
            
            # CHECK B: Is the number of macromolecules for EACH branch type an integer?
            # Num_j = Fraction_j * Total_Macromolecules
            all_species_are_integers = True
            for j in range(len(D0_comp)):
                num_macromol_j = D0_comp[j] * tot_number_of_macromol
                if not (abs(num_macromol_j - np.round(num_macromol_j)) < 1e-9):
                    all_species_are_integers = False
                    break
            
            if all_species_are_integers:
                valid_compositions_for_this_Nx.append(D0_comp)
        
        # Only add to dictionary if this Nx has at least one valid composition
        if len(valid_compositions_for_this_Nx) > 0:
            results_dict[Nx] = valid_compositions_for_this_Nx

    return results_dict

def find_Nx_match(D0_composition, Nx_viable):
    """
    Looks through the already calculated Nx_viable array and returns the 
    first Nx that mathematically allows the specific D0_composition.
    """
    # Calculate the average chain length based on the composition input
    # Assumes D0_composition[0] is linear (1 chain), [1] is 1-branched (2 chains), etc.
    avg_chain_length = np.sum([D0_composition[k] * (k+1) for k in range(len(D0_composition))])

    for Nx in Nx_viable:
        # Calculate the theoretical total number of macromolecules
        # (Nx - 1) because 1 chain is the terminated 'G'
        tot_macromol = (Nx - 1) / avg_chain_length

        # Check 1: Is the total number of macromolecules a whole number?
        if abs(tot_macromol - round(tot_macromol)) > 1e-9:
            continue

        # Check 2: Is the number of molecules for EACH species a whole number?
        is_valid = True
        for fraction in D0_composition:
            count = fraction * tot_macromol
            if abs(count - round(count)) > 1e-9:
                is_valid = False
                break
        
        if is_valid:
            return int(Nx) # Found the match!

    return None # No match found


# --- Example Usage ---

# Example Parameters (similar to your setup)
# [min_Nx, max_Nx, max_diff, eps, max_branches]
Nx_parameters = [200, 1000, 0.1, 0.05, 3] 

# [D_conc, R_conc, ga0]
process_parameters = [1.0, 0.1, [0.01]] 

# Run the search
viable_dictionary = find_viable_compositions_for_Nx(Nx_parameters, process_parameters)

# # Print results
# print("\n--- Results ---")
# for Nx_val, comps in viable_dictionary.items():
#     print(f"\nNx: {Nx_val} supports {len(comps)} compositions:")
#     for c in comps:
#         print(f"  {c}")


# Define the composition you are looking for
target_composition = [0.2, 0.3, 0.5] 

# Call the function
matched_Nx = find_Nx_match(target_composition, Nx_viable)

if matched_Nx:
    print(f"Success! The composition {target_composition} works with Nx = {matched_Nx}")
else:
    print(f"No match found for {target_composition} in the current Nx_viable list.")