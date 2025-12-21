import numpy as np
from measure_time import measure_time

def generate_decimal_compositions(n_branches, precision_steps=10):
    """
    Generates all possible lists of fractions that sum to 1.0.
    Returns a list of numpy arrays, e.g., [0.2, 0.3, 0.5]
    """
    def sums(target, n):
        if n == 1:
            yield (target,)
        else:
            for i in range(target + 1):
                for tail in sums(target - i, n - 1):
                    yield (i,) + tail

    integer_compositions = list(sums(precision_steps, n_branches))
    
    fraction_compositions = []
    for comp in integer_compositions:
        fraction_compositions.append(np.array([x/precision_steps for x in comp]))
        
    return fraction_compositions


# --- 1. CORE LOGIC: FIND PROCESS VIABLE Nx ---

@measure_time
def get_process_viable_Nx(Nx_pars, process_pars):
    """
    Step 1: Find all Nx values that satisfy the R/D/G equilibrium constraints.
    Does NOT check compositions.
    
    Returns:
        np.array: A list of viable integer Nx values.
    """
    # Unpack parameters
    min_Nx = Nx_pars[0]                        
    max_Nx = Nx_pars[1]                        
    max_difference_RD_dec_round = Nx_pars[2]   
    
    D_conc = process_pars[0]                     
    R_conc = process_pars[1]                     
    ga0 = process_pars[2]                        
    G_conc = ga0[-1] if isinstance(ga0, (list, np.ndarray)) else ga0

    N_A = 6.022140858e23    

    Nx_viable = [] 
    
    for Nx in range(int(min_Nx), int(max_Nx) + 1):
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
            
    return np.array(Nx_viable)


# --- 2. TARGET SEARCH: MATCH 1 COMPOSITION ---
@measure_time
def find_Nx_match(D0_composition, Nx_viable):
    """
    Step 2a: Look through the valid Nx list and find the FIRST Nx 
    that supports the specific input composition.
    
    Returns:
        int or None: The matching Nx, or None if not found.
    """
    # Calculate average chain length (weighted sum of branches)
    # Assumes D0_composition[k] corresponds to (k+1) chains
    avg_chain_length = np.sum([D0_composition[k] * (k+1) for k in range(len(D0_composition))])

    for Nx in Nx_viable:
        # Calculate theoretical total macromolecules (Nx-1 because of terminated G)
        tot_macromol = (Nx - 1) / avg_chain_length

        # Check 1: Is total macromolecules an integer?
        if abs(tot_macromol - round(tot_macromol)) > 1e-9:
            continue

        # Check 2: Is the number of molecules for EACH species an integer?
        is_valid = True
        for fraction in D0_composition:
            count = fraction * tot_macromol
            if abs(count - round(count)) > 1e-9:
                is_valid = False
                break
        
        if is_valid:
            return int(Nx) # Found the match!

    return None # No match found


# --- 3. REVERSE SEARCH: MAP ALL COMPOSITIONS ---

@measure_time
def map_all_compositions(Nx_viable, max_branches):
    """
    Step 2b: For every valid Nx, find ALL possible compositions that work.
    
    Returns:
        dict: { Nx_int : [ list of valid composition arrays ] }
    """
    # Generate all theoretical decimal compositions (e.g. 0.1 steps)
    candidate_compositions = generate_decimal_compositions(max_branches, precision_steps=10)
    
    results_dict = {}

    for Nx in Nx_viable: 
        valid_compositions_for_this_Nx = []
        
        for D0_comp in candidate_compositions:
            # Re-use logic: Calculate Avg Chain Length
            avg_chain_length = np.sum([D0_comp[k] * (k+1) for k in range(max_branches)])
            
            # Re-use logic: Calculate Total Macromolecules
            tot_number_of_macromol = (Nx - 1) / avg_chain_length 
            
            # Check A: Integer Total?
            if not (abs(tot_number_of_macromol - np.round(tot_number_of_macromol)) < 1e-9):
                continue 
            
            # Check B: Integer Species Counts?
            all_species_are_integers = True
            for j in range(len(D0_comp)):
                num_macromol_j = D0_comp[j] * tot_number_of_macromol
                if not (abs(num_macromol_j - np.round(num_macromol_j)) < 1e-9):
                    all_species_are_integers = False
                    break
            
            if all_species_are_integers:
                valid_compositions_for_this_Nx.append(D0_comp)
        
        if len(valid_compositions_for_this_Nx) > 0:
            results_dict[Nx] = valid_compositions_for_this_Nx

    return results_dict


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # --- A. Setup Parameters ---
    # [min_Nx, max_Nx, max_diff, eps, max_branches]
    Nx_parameters = [200, 1000, 0.1, 0.05, 3] 
    # [D_conc, R_conc, ga0]
    process_parameters = [1.0, 0.1, [0.01]] 

    print("--- 1. Finding Process Viable Nx ---")
    
    # 1. Get the list of Nx values that satisfy R/D equilibrium
    # This is now separated from the composition logic
    viable_Nx_array = get_process_viable_Nx(Nx_parameters, process_parameters)
    print(f"Found {len(viable_Nx_array)} viable Nx values.\n")


    # --- B. Find Match for Specific Composition (User Request) ---
    print("--- 2. Checking Specific Composition ---")
    
    target_composition = [0.2, 0.3, 0.5] 
    matched_Nx = find_Nx_match(target_composition, viable_Nx_array)

    if matched_Nx:
        print(f"SUCCESS: Composition {target_composition} is valid with Nx = {matched_Nx}\n")
    else:
        print(f"FAILURE: Composition {target_composition} not found in current Nx range.\n")


    # --- C. (Optional) Generate the Big Dictionary ---
    print("--- 3. Mapping All Possibilities (Optional) ---")
    
    # We pass the already calculated viable_Nx_array into this function
    full_dictionary = map_all_compositions(viable_Nx_array, max_branches=3)
    
    print(f"Generated map for {len(full_dictionary)} Nx values.")
    # Example print of first found
    first_key = list(full_dictionary.keys())[0]
    print(f"Example: Nx={first_key} supports {len(full_dictionary[first_key])} compositions.")