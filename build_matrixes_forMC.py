import numpy as np
from measure_time import measure_time

@measure_time
def build_matrixes_forMC(D0_composition, Nx, R_round, max_branches):
    """
    Sorts the system according to the inlet composition into matrixes for dormant and active chains.

    Parameters:
        D0_composition (list or np.array): The initial composition fractions found by the algorithm find_Nx if the user defined is not viable.
        Nx (int): The total number of chains (including 1 terminated chain to start up the simulation).
        R_round (int): The number of active chains.
        max_branches (int): The maximum number of branches in a molecule.

    Returns:
        D - matrix representing the dormant chains, each column is one molecule and each row is one branch in that molecule.
        R - matrix representing the active chains, each column is one molecule and each row is one branch in that molecule.
        G - array representing the terminated chains, each number is a molecule.
    """

    print("Building matrixes for MC simulation...")
    # Initialize arrays
    num_branch_per_mol_type = np.zeros(max_branches)  # Number of branches in each molecule type
    num_mol_per_type = np.zeros(max_branches)  # Number of molecules with (i+1) branches
    
    # Calculate number of molecules and branches per molecule type
    for i in range(max_branches):
        num_branch_per_mol_type[i] = int(D0_composition[i] * (Nx - 1))
        num_mol_per_type[i] = num_branch_per_mol_type[i] / (i + 1)

    total_num_D_mol = int(np.sum(num_mol_per_type))  # Total number of molecules excluding 1 terminated chain
    
    # Define chain matrices
    D = np.ones((max_branches, total_num_D_mol))  # Dormant chains
    R = np.zeros((max_branches, total_num_D_mol))  # Active chains
    G = np.ones(1)  # One terminated chain of length 1

    # Set D=0 for branches that don't exist
    sum_molecules = 0
    for i in range(1, max_branches):
        sum_molecules += int(num_mol_per_type[i - 1])
        D[i, :sum_molecules] = 0
    
    # Convert dormant chains to active based on "instantaneous" equilibrium
    for _ in range(R_round):
        D_idx = np.argwhere(D > 0)  # Get indices of positive values
        if D_idx.size == 0:
            break  # No more available dormant chains
        D_idx_random = D_idx[np.random.choice(D_idx.shape[0])]  # Select a random dormant chain
        R[D_idx_random[0], D_idx_random[1]] = 1  # Set the selected chain to active
        D[D_idx_random[0], D_idx_random[1]] = 0  # Remove from dormant

    # Print information
    print("Matrix building finished.")
    print(f"Numbers of dormant branches from linear to x-branched molecules: {num_branch_per_mol_type}")
    print(f"Number of dormant molecules with x-branches: {num_mol_per_type}")
    print(f"D size: {D.size}, shape: {D.shape}")
    print(f"R size: {R.size}, shape: {R.shape}")
    print(f"G size: {G.size}, shape: {G.shape}")

    return D, R, G
