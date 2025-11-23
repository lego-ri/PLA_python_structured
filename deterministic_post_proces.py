import numpy as np

def compute_molecular_moments_from_solution(solution, D0_composition, I0=None):
    """
    solution: object with .t (1D array) and .y (2D array with shape (n_vars, n_times))
    D0_composition: iterable where index f contains fraction for functionality f.
                    Example for 50% f=1 and 50% f=3: D0_composition = [0.0, 0.5, 0.0, 0.5]
                    (index 0 allowed but typically zero).
    I0: optional known initiator concentration (mol/m3). If provided and you are sure
        molecules remain distinct, N_molecules = I0; otherwise N_molecules = N_chains / avg_f.
    Returns dict of arrays (same length as solution.t).
    """

    t = solution.t
    y = solution.y
    # Unpack chain moments as arrays (shape: n_times)
    la0 = y[3]
    la1 = y[4]
    la2 = y[5]
    mu0 = y[6]
    mu1 = y[7]
    mu2 = y[8]
    ga0 = y[9]
    ga1 = y[10]
    ga2 = y[11]

    # chain totals per time (per volume)
    N_chains = la0 + mu0 + ga0    # shape (n_times,)
    S1_chains = la1 + mu1 + ga1
    S2_chains = la2 + mu2 + ga2

    # Avoid division by zero: set safe mask
    eps = 1e-30
    N_chains_safe = np.where(N_chains > 0, N_chains, eps)

    # per-branch (per-chain) moments
    m1_branch = S1_chains / N_chains_safe
    m2_branch = S2_chains / N_chains_safe

    # process D0_composition
    p = np.array(D0_composition, dtype=float)
    if p.sum() <= 0:
        raise ValueError("D0_composition must sum to > 0")
    p = p / p.sum()
    f = np.arange(len(p))
    avg_f = float(np.sum(p * f))
    E_ff1 = float(np.sum(p * f * (f - 1)))   # E[f(f-1)]

    # number concentration of molecules (per volume)
    if I0 is not None:
        N_molecules = np.full_like(N_chains, I0, dtype=float)
    else:
        if avg_f <= 0:
            raise ValueError("Average functionality <= 0; check D0_composition")
        N_molecules = N_chains / avg_f

    # per-molecule moments (per molecule)
    mu1_mol = avg_f * m1_branch
    mu2_mol = avg_f * m2_branch + (m1_branch**2) * E_ff1

    # DP_n, DP_w and dispersity per molecule (arrays)
    DP_n_mol = mu1_mol
    # avoid division by zero
    DP_w_mol = np.where(mu1_mol > 0, mu2_mol / mu1_mol, 0.0)
    dispersity_mol = np.where(DP_n_mol > 0, DP_w_mol / DP_n_mol, 0.0)

    # per-volume molecular moments (consistency checks)
    S1_from_molecules = N_molecules * mu1_mol   # should equal S1_chains (within numerical error)
    S2_from_molecules = N_molecules * mu2_mol   # this is (sum of molecule (sum branches)^2) per volume

    return {
        "t": t,
        "N_chains": N_chains,
        "S1_chains": S1_chains,
        "S2_chains": S2_chains,
        "m1_branch": m1_branch,
        "m2_branch": m2_branch,
        "avg_f": avg_f,
        "E_f_fminus1": E_ff1,
        "N_molecules": N_molecules,
        "mu1_mol": mu1_mol,
        "mu2_mol": mu2_mol,
        "DP_n_mol": DP_n_mol,
        "DP_w_mol": DP_w_mol,
        "dispersity_mol": dispersity_mol,
        "S1_from_molecules": S1_from_molecules,
        "S2_from_molecules": S2_from_molecules,
        "D0_composition_normalized": p
    }

# Example use for your 50/50 case:
# D0_composition = [0.0, 0.5, 0.0, 0.5]   # index: 0,1,2,3 -> p(1)=0.5, p(3)=0.5
# results = compute_molecular_moments_from_solution(solution, D0_composition)
# Now results['DP_n_mol'] etc. are arrays indexed by time.
