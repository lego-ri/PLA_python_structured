import numpy as np
from measure_time import measure_time
from scipy import interpolate           # To fit the monomer profile as piecewise linear
from time import time
from select_reaction import SelReac     # For semi-random chosing of the reaction to occure (is not used in the end as an inbuilt function in Python is used isntead)



@measure_time
def monte_carlo_algorithm(mc_pars, process_pars, ModelPars): 
    """
    Runs the hybrid Monte Carlo simulation using the R/D equil from the deterministic model.
    
    Parameters:
        mc_pars (array): 
            Nx (int): The total number of chains (including 1 terminated chain to start up the simulation).
            N_A - Avogadro's number 
            M - monomer concentration 
            C - catalyst concentration
            A - acid concentration
            R - matrix representing the active chains 
            D - matrix representing the dormant chains
            G - array representing the terminated chains
            total_num_D_mol - total  number of molecules at the beginning of the simulation, excluding 1 terminated chain

            
        process_pars (array): 
            D_conc - concentration of dormant chains from the deterministic model main_process
            R_conc - concentration of active chains from the deterministic model main_process
            Gn    - concentration of terminated chains from the deterministic model main_process
            t      - time in simulation from deterministic model

            
            
        ModelPars (class):
    """
    
    #* Unpack parameters
    Nx = mc_pars[0]
    M = mc_pars[1]
    C = mc_pars[2]
    A = mc_pars[3]
    R = mc_pars[4]
    D = mc_pars[5]
    G = mc_pars[6]
    total_num_D_mol = mc_pars[7]
    
    D_conc = process_pars[0]    # Concentration of dormant chains from the deterministic model main_process
    R_conc = process_pars[1]    # Concentration of active chains from the deterministic model main_process
    Gn = process_pars[2]       # Concentration of terminated chains from the deterministic model main_process
    t = process_pars[3]         # Time in simulation, s
    y = process_pars[4]         # Concentration profiles from the deterministic model main_process
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
    
    k_p = ModelPars.k_p
    k_d = ModelPars.k_d
    k_s = ModelPars.k_s
    k_te = ModelPars.k_te
    k_de = ModelPars.k_de    
       
    #* Monte carlo parameters   
    N_A = 6.022140858e23     # Avogadro number, 1/mol
    V = Nx / (N_A*(D_conc + R_conc + Gn[-1]))        # Simulated volume, m3 #! terminated chains incl.
    time_end = t[-1]
    MW = ModelPars.MW        # Molecular weight of lactoyl (monomer) group, kg/mol
    fq = 10                 # frequency for data logging
       
    #* Convert deterministic kinetic rates to stochastic (MC)
    k_p_MC   = k_p  / (V*N_A)        # Propagation, 1/s
    k_d_MC   = k_d                   # Depropagation, 1/s
    k_s_MC   = k_s  / (V*N_A)        # Chain transfer, 1/s
    k_te_MC  = k_te / (V*N_A)        # Transesterification, 1/s
    k_de_MC  = k_de                  # Scission, 1/s

    #* Convert specie concetrations to number values for MC
    M_MC    = M  * N_A * V                          # Number of monomer molecules, -
    C_MC    = C  * N_A * V                          # Number of catalyst molecules, -
    A_MC    = A  * N_A * V                          # Number of acid molecules, -
    la0_MC = la0 * N_A * V  # 0th moment of active chains, mol/m3
    la1_MC = la1 * N_A * V  # 1st moment of active chains, mol/m3
    la2_MC = la2 * N_A * V  # 2nd moment of active chains, mol/m3
    mu0_MC = mu0 * N_A * V  # 0th moment of dormant chains, mol/m3
    mu1_MC = mu1 * N_A * V  # 1st moment of dormant chains, mol/m3
    mu2_MC = mu2 * N_A * V  # 2nd moment of dormant chains, mol/m3
    ga0_MC = ga0 * N_A * V  # 0th moment of terminated chains, mol/m3
    ga1_MC = ga1 * N_A * V  # 1st moment of terminated chains, mol/m3
    ga2_MC = ga2 * N_A * V  # 2nd moment of terminated chains, mol/m3


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
    # C_cur = C_MC[-1]                                        # Catalyst concentration considered constant 
    # A_cur = A_MC[-1]                                        # Acid concentration considered constant 
    M_fit = interpolate.interp1d(t, M_MC, kind='linear')    # Fit monomer profile as piecewise linear, later used in the main loop 

    #* Main simulation loop 
    start_time = time()     # To measure the time taken by the simulation
    time_sim = 0                    # Reaction time in simulation, s (not time taken to simulate)
    Rate = np.zeros(reac_num)       # Initiate the vector of reaction rate
    case_counts = np.zeros(reac_num, dtype=int) # Initialize a counter array for the 12 cases
    
    step_counter = 0 #TODO
      
    while time_sim <= time_end:        
        # Get concentrations of non-polymeric (monomer) species from process ODE simulation
        M_cur = M_fit(time_sim) # (cur~current concentration)

        #* Get concentrations (numbers) of polymer chains
        Rn = np.sum(R>0) # Active chains (branches)
        Dn = np.sum(D>0) # Dormant chains (branches)
        Gn = len(G) # Terminated chains
        
        sumR = np.sum(R)  # Total number of monomer units in active chains
        sumD = np.sum(D)  # Total number of monomer units in dormant chains
        sumG = np.sum(G)  # Total number of monomer units in terminated chains
        R_column_sums = np.sum(R, axis=0) # Lengths of all macromolecules (each column is a macromolecule with x branches(rows)) 
        D_column_sums = np.sum(D, axis=0) # Lengths of all macromolecules (each column is a macromolecule with x branches(rows)) 
        sumR2 = np.sum(R_column_sums**2)  # Sum of squared lengths of all macromolecules (2nd moment)
        sumD2 = np.sum(D_column_sums**2)  # Sum of squared lengths of all macromolecules (2nd moment)
        sumG2 = np.sum(G**2)              # Sum of squared lengths of all terminated chains (2nd moment)
        sumR3 = np.sum(R_column_sums**3)  # Sum of cubed lengths of all macromolecules (3rd moment)
        sumD3 = np.sum(D_column_sums**3)  # Sum of cubed lengths of all macromolecules (3rd moment)
        sumG3 = np.sum(G**3)              # Sum of cubed lengths of all terminated chains (3rd moment)
        
        #* Evaluate individual reaction rates
#TODO
        # Rate[0] = k_p_MC * M_cur * Rn         # Propagation 
        # Rate[1] = k_d_MC * Rn                   # Depropagation 
        # Rate[2] = k_s_MC * Rn * Dn           # Chain transfer 
        # Rate[3] = k_de_MC *sumR           # Random scission (R)
        # Rate[4] = k_de_MC * sumD           # Random scission (D)
        # Rate[5] = k_de_MC * sumG           # Random scission (G)
        # Rate[6] = k_te_MC * Rn * sumD      # "Active" transesterification (R+D)
        # Rate[7] = k_te_MC * Rn * sumD/Dn   # "Passive" transesterification (R+D), if Dn > 0 else 0
        # Rate[8] = 2 * k_te_MC * Rn * sumR # "Active" transesterification (R+R)
        # Rate[9] = 2 * k_te_MC * sumR      # "Passive" transesterification (R+R)
        # Rate[10] = k_te_MC * Rn * sumG     # "Active" transesterification (R+G)
        # Rate[11] = k_te_MC * Rn * sumG/Gn  # "Passive" transesterification (R+G), if Gn > 0 else 0 
        # Find the closest indices in `t` for current time_sim`
        current_t_idx = int(np.argmin(np.abs(t - time_sim)))
        # Current (scalar) state values taken from ODE solution arrays
        M_cur_det   = max(M_MC[current_t_idx], 1e-7)        # Monomer concentration at current time
        C_cur_det   = max(C_MC[current_t_idx], 1e-7)        # Catalyst concentration at current time
        A_cur_det   = max(A_MC[current_t_idx], 1e-7)        # Acid concentration at current time
        la0_cur_det = max(la0_MC[current_t_idx], 1e-7)      # 0th moment of active chains, mol/m3
        la1_cur_det = max(la1_MC[current_t_idx], 1e-7)      # 1st moment of active chains, mol/m3
        la2_cur_det = max(la2_MC[current_t_idx], 1e-7)      # 2nd moment of active chains, mol/m3
        mu0_cur_det = max(mu0_MC[current_t_idx], 1e-7)      # 0th moment of dormant chains, mol/m3
        mu1_cur_det = max(mu1_MC[current_t_idx], 1e-7)      # 1st moment of dormant chains, mol/m3
        mu2_cur_det = max(mu2_MC[current_t_idx], 1e-7)      # 2nd moment of dormant chains, mol/m3
        ga0_cur_det = max(ga0_MC[current_t_idx], 1e-7)      # 0th moment of terminated chains, mol/m3
        ga1_cur_det = max(ga1_MC[current_t_idx], 1e-7)      # 1st moment of terminated chains, mol/m3
        ga2_cur_det = max(ga2_MC[current_t_idx], 1e-7)      # 2nd moment of terminated chains, mol/m3
        
        Rate[0] = k_p_MC * M_cur_det * la0_cur_det         # Propagation 
        Rate[1] = k_d_MC * la0_cur_det                   # Depropagation 
        Rate[2] = k_s_MC * la0_cur_det * mu0_cur_det           # Chain transfer 
        Rate[3] = k_de_MC *la1_cur_det           # Random scission (R)
        Rate[4] = k_de_MC * mu1_cur_det           # Random scission (D)
        Rate[5] = k_de_MC * ga1_cur_det           # Random scission (G)
        Rate[6] = k_te_MC * la0_cur_det * mu1_cur_det      # "Active" transesterification (R+D)
        Rate[7] = k_te_MC * la0_cur_det * mu1_cur_det/mu0_cur_det   # "Passive" transesterification (R+D), if Dn > 0 else 0
        Rate[8] = 2 * k_te_MC * la0_cur_det * la1_cur_det # "Active" transesterification (R+R)
        Rate[9] = 2 * k_te_MC * la1_cur_det      # "Passive" transesterification (R+R)
        Rate[10] = k_te_MC * la0_cur_det * ga1_cur_det     # "Active" transesterification (R+G)
        Rate[11] = k_te_MC * la0_cur_det * ga1_cur_det/ga0_cur_det  # "Passive" transesterification (R+G), if Gn > 0 else 0 
# TODO
        #* Select the reaction to happen (randomly weighted by reaction rates)
        Reac_idx = np.random.choice(reac_num, p=Rate/np.sum(Rate))    # inbuilt function that does the same as our custom sel_reac
        # Reac_idx = SelReac.sel_reac(Rate, reac_num)                     # Handmade function 
        # Increment the counter for the selected case
        case_counts[Reac_idx] += 1
        step_counter = 0

        #* Realize the selected reaction step
        match Reac_idx:
            case 0:  # Propagation
                R_idx = np.argwhere(R>0)   # Indices of positive values
                R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)
                R[R_idx_random[0], R_idx_random[1]] += 2    # Propagate the chain (+2, because each monomer adds 2 units)

            case 1:  # Depropagation 
                R_idx = np.argwhere(R > 2) # Indexes of active chains with length more than 2
                if R_idx.size > 0:
                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)
                    R[R_idx_random[0],R_idx_random[1]] -= 2   # Depropagate the chain (-2, because each monomer removes 2 units)
                    # print(f"Depropagation, R is {R[R_idx_random[0],R_idx_random[1]]}")
                else:
                    print("Depropagation chosen, but no chains with length>2 available to depropagate\nsimulation ongoing...")
                    continue

            case 2:  # Chain transfer
                R_idx, D_idx = np.argwhere(R>0), np.argwhere(D>0)   # Indices of positive values i.e. existing chains
                R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)
                D_idx_random = D_idx[np.random.choice(D_idx.shape[0])] # Select random dormant chain
                # Transfer the chains
                R[R_idx_random[0], R_idx_random[1]], D[D_idx_random[0], D_idx_random[1]] = D[D_idx_random[0], D_idx_random[1]], R[R_idx_random[0], R_idx_random[1]] 


            case 3:  # Random scission (R)
                # Get indexes of active chains with length greater than 1
                R_idx = np.argwhere(R>1)   # Indices of dormant chains available for scission
                if R_idx.size > 0:
                    R_lengths = R[R_idx[:, 0], R_idx[:, 1]] # Lengths of active chains 
                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0], p=R_lengths/np.sum(R_lengths))] # Select random active chain (radical)
                    sub_len = np.random.randint(1, R[R_idx_random[0], R_idx_random[1]])    # Randomly select subchain length (part of the radical)
                    R[R_idx_random[0], R_idx_random[1]] -= sub_len # Remove subchain from radical
                    G = np.append(G, sub_len)                       # Add a new terminated chain (born from the part of the radical)
                else:
                    print("Random scission (R) chosen, but no chains with length>1 available to split.\nsimulation ongoing...")
                    continue

            case 4:  # Random scission (D)
                # Get indexes of dormant chains with length greater than 1
                D_idx = np.argwhere(D>1)   # Indices of dormant chains available for scission
                if D_idx.size > 0:
                    D_lengths = D[D_idx[:, 0], D_idx[:, 1]] # Lengths of active chains 
                    D_probability = D_lengths/np.sum(D_lengths) # Probability of each active chain proportional to chain length
                    D_idx_random = D_idx[np.random.choice(D_idx.shape[0], p=D_probability)] # Select random dormant chain
                    sub_len = np.random.randint(1, D[D_idx_random[0], D_idx_random[1]])    # Randomly select subchain length (part of the radical)
                    D[D_idx_random[0], D_idx_random[1]] -= sub_len # Remove subchain from radical
                    G = np.append(G, sub_len)                       # Add a new terminated chain (born from the part of the radical)
                else:
                    print("Random scission (D) chosen, but no chains with length>1 available to split.\nsimulation ongoing...")
                    continue

            case 5:  # Random scission (G)
                # Get indexes of terminated chains with length greater than 1
                G_idx = np.argwhere(G > 1).flatten()  # Flatten to get a 1D array of indices
                if G_idx.size > 0:
                    probabilities = G[G_idx] / np.sum(G[G_idx])  # Calculate probabilities only for chains with length > 1
                    G_idx_random = np.random.choice(G_idx, p=probabilities)  # Select a random index from G_idx with probability proportional to chain length
                    sub_len = np.random.randint(1, G[G_idx_random])  # Randomly select subchain length (part of the terminated chain) of size 1 up to the chain length minus 1
                    G[G_idx_random] -= sub_len  # Remove subchain from the selected terminated chain
                    G = np.append(G, sub_len)  # Add the new terminated chain (from part of the terminated chain) to G
                else:
                    print("Random scission (G) chosen, but no chains with length>1 available to split.")
                    continue
                
            case 6:  # "Active" transesterification (R+D) 
                D_idx = np.argwhere(D>1)   # Indices of considered dormant chains values
                R_idx = np.argwhere(R>0)   # Indices of all activee chains
                if D_idx.size > 0:
                    D_lengths = D[D_idx[:, 0], D_idx[:, 1]] # Lengths of dormant chains 
                    D_probability = D_lengths/np.sum(D_lengths) # Probability of each dormant chain proportional to chain length
                    D_idx_random = D_idx[np.random.choice(D_idx.shape[0], p=D_probability)] # Select random dormant chain
                    sub_len_D = np.random.randint(1, D[D_idx_random[0], D_idx_random[1]])    # Randomly select part of the chain

                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)
                    R[R_idx_random[0], R_idx_random[1]] += sub_len_D      # Add the selected chain part to the active chain
                    D[D_idx_random[0], D_idx_random[1]] -= sub_len_D      # Remove the selected chain part from the dormant chain
                else:
                    print("Active transesterification (R+D) chosen, but no dormant chain with length>1 available.\nsimulation ongoing...")
                    continue

            case 7:  # "Passive" transesterification (R+D)
                D_idx = np.argwhere(D>1)   # Indices of non NaN values
                R_idx = np.argwhere(R>0)   # Indices of non NaN values
                if D_idx.size > 0:
                    D_idx_random = D_idx[np.random.choice(D_idx.shape[0])] # Select random dormant chain
                    sub_len_D = np.random.randint(1, D[D_idx_random[0], D_idx_random[1]])    # Randomly select part of the chain

                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)
                    R[R_idx_random[0], R_idx_random[1]] += sub_len_D      # Add the selected chain part to the active chain
                    D[D_idx_random[0], D_idx_random[1]] -= sub_len_D      # Remove the selected chain part from the dormant chain
                else:
                    print("Passive transesterification (R+D) chosen, but no dormant chain with length>1 available.\nsimulation ongoing...")
                    continue

            case 8:  # "Active" inter-transesterification (R+R)
                R_idx_1 = np.argwhere(R > 1)
                if R_idx_1.size > 0:
                    R_lengths = R[R_idx_1[:,0], R_idx_1[:,1]] # Lengths of active chains
                    R_probability = R_lengths/np.sum(R_lengths) # Probability of each active chain proportional to chain length
                    R_idx_random_1 = R_idx_1[np.random.choice(R_idx_1.shape[0], p=R_probability)] # Select first active chain based on probability
                    
                    R_idx_2 = np.argwhere(R > 0)
                    R_indexes_2_filt = R_idx_2[~np.all(R_idx_2 == R_idx_random_1, axis=1)]
                    R_idx_random_2 = R_indexes_2_filt[np.random.choice(R_indexes_2_filt.shape[0])] # Select second active chain based on probability                
                    sub_len_R_1 = np.random.randint(1, R[R_idx_random_1[0], R_idx_random_1[1]])    # Randomly select part of the first chain
                    R[R_idx_random_1[0], R_idx_random_1[1]] -= sub_len_R_1      # Remove the selected chain part from the first chain
                    R[R_idx_random_2[0], R_idx_random_2[1]] += sub_len_R_1       # Add the selected chain part to the second chain
                else:
                    print("Active transesterification (R+R) chosen, but no active chains with length>1 available\nsimulation ongoing...")
                    continue

            case 9:  # "Passive" inter-transesterification (R+R)
                R_indexes_1 = np.argwhere(R > 1)
                if R_indexes_1.size > 0:
                    R_idx_random_1 = R_indexes_1[np.random.choice(R_indexes_1.shape[0])] # Select first active chain based on probability
                    
                    R_idx_2 = np.argwhere(R > 0)
                    R_indexes_2_filt = R_idx_2[~np.all(R_idx_2 == R_idx_random_1, axis=1)]
                    R_idx_random_2 = R_indexes_2_filt[np.random.choice(R_indexes_2_filt.shape[0])] # Select second active chain based on probability                
                    sub_len_R_1 = np.random.randint(1, R[R_idx_random_1[0], R_idx_random_1[1]])    # Randomly select part of the first chain
                    R[R_idx_random_1[0], R_idx_random_1[1]] -= sub_len_R_1      # Remove the selected chain part from the first chain
                    R[R_idx_random_2[0], R_idx_random_2[1]] += sub_len_R_1       # Add the selected chain part to the second chain
                else:
                    print("Passive transesterification (R+R) chosen, but no active chains with length>1 available\nsimulation ongoing...")
                    continue

            case 10:  # "Active" transesterification (R+G)
                G_idx = np.argwhere(G > 1).flatten()
                if G_idx.size > 0:
                    G_idx_random = np.random.choice(G_idx, p=G[G_idx]/np.sum(G[G_idx]))
                    DP_G = G[G_idx_random]
                    DP_G_rnd = np.random.randint(1, DP_G)
                    
                    R_idx = np.argwhere(R > 0)
                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)

                    R[R_idx_random[0], R_idx_random[1]] += DP_G_rnd
                    G[G_idx_random] -= DP_G_rnd
                else:
                    print("Active transesterification (R+G) chosen, but no terminated chains with length>1 available\nsimulation ongoing...")
                    continue

            case 11:  # "Passive" transesterification (R+G)
                G_idx = np.argwhere(G > 1).flatten()
                if G_idx.size > 0:
                    G_idx_ranom = np.random.choice(G_idx)
                    DP_G = G[G_idx_ranom]
                    DP_G_rnd = np.random.randint(1, DP_G)
                                    
                    R_idx = np.argwhere(R > 0)
                    R_idx_random = R_idx[np.random.choice(R_idx.shape[0])] # Select random active chain (radical)

                    R[R_idx_random[0], R_idx_random[1]] += DP_G_rnd
                    G[G_idx_ranom] -= DP_G_rnd
                else:
                    print("Passive transesterification (R+G) chosen, but no terminated chains with length>1 available.")
                    continue


        #* Increment the elapsed time 
        # tau = -np.log(np.random.random()) / 477    # np.random.random() generates a random number between 0 and 1
        sumRate = np.sum(Rate)
        step_counter += 1
        if sumRate <1:
            input(f"Pausing since sumRate is very low: {sumRate:.2e}, stepcounter:{step_counter} press Enter to continue...")
            sumRate = 50
        tau = -np.log(np.random.random()) / sumRate    # np.random.random() generates a random number between 0 and 1
        time_sim += tau                                     # Update elapsed time with the time increment
        # print(time_sim, tau, np.sum(Rate))
        
        #* Print the current time in simulation every 20 seconds of real time
        current_time = time() - start_time  # Gets the CPU time in seconds since the start of teh simulation
        if current_time > 20*current_time_counter:
            current_time_counter += 1
            # print(f"Real time elapsed: {current_time:.2f} seconds\nsimulation ongoing...")
            print(f"Current time in simulation: {time_sim:.2f} seconds out of {time_end:.2f} seconds")
            
        #* Data output - for post processing
        if time_sim > out_idx * fq:     # register the data every fq seconds
            out_idx += 1
            t_out = np.append(t_out, time_sim)              # Register the time within the simulation
            Rates_out = np.column_stack((Rates_out, Rate))  # Register the reaction rates
            R_out = np.append(R_out, Rn)                    # Register the number of active chains
            D_out = np.append(D_out, Dn)                    # Register the number of dormant chains
            G_out = np.append(G_out, Gn)                    # Register the number of terminated chains

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Combine the matrixes of dormant and active branches into one matrix of all species except for terminated chains
            R_and_D = np.where(D<1, R, D)
            
            # calculate the number or elements in R_and_D
            #Ntot = R_and_D.shape[1] + Gn            # Total number of branches 
            Ntot = total_num_D_mol + Gn              # Total number of polymer chains (macromolecules, not branches) (i.e. 0th moment)        
            
            RD_column_sums = np.sum(R_and_D, axis=0) # Lengths of all macromolecules (each column is a macromolecule with x branches(rows))            
                    
            Sum1 = np.sum(R_and_D) + np.sum(G)                  # Sum of all polymer chain lengths (1st moment)
            Sum2 = np.sum(RD_column_sums**2) + np.sum(G**2)     # Sum of all squared polymer chain lengths (2nd moment)

            
            # Evaluate Mn, Mw # TODO: chagne the way Mn and Mw are calculated - MW??
            Mn_out = np.append(Mn_out, MW * Sum1 / Ntot)        # Number-average mol. weight, kg/mol
            Mw_out = np.append(Mw_out, MW * Sum2 / Sum1)        # Weight-average mol. weight, kg/mol 
            suma_n_tot = np.append(suma_n_tot, Ntot)            # Save the total number of polymer chains            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
    #* End of simulation, evaluated time taken to simulate the process
    print(f"Total number of chains is Ntot={Ntot}, reactive branches Rn={Rn}, dormant branches Dn={Dn}, terminated chains G={Gn}")
    print(f"Number of times each reaction occured: {case_counts}")
    print(f"Total number of steps in simulation: {step_counter}") #TODO
    
    return t_out, Rates_out, R_out, D_out, G_out, Mn_out, Mw_out, suma_n_tot, RD_column_sums, R, D, G
