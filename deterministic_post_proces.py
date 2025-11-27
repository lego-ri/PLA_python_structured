import numpy as np
from model_pars import ModelPars            # Model parameters
from measure_time import measure_time

@measure_time
def deterministic_post_proces(
    y,
    D0_composition,
    mode='chain',   # 'chain' if D0_composition gives fraction of chains (branches)
                    # 'molecule' if it gives fraction of molecules (per-functionality)
    eps = 1e-7
):
    """
    Compute Mn and Mw; D0_composition[i] corresponds to functionality f = i+1.
    mode:
      - 'chain': D0_composition are fractions of chains (branches): p_f
      - 'molecule': D0_composition are fractions of molecules: q_f
    """
    
    #* Unpack input
    small_number = eps
    p_raw = np.array(D0_composition, dtype=float)
    
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
    
    
    
    # Basic checks
    if p_raw.size == 0:
        raise ValueError("D0_composition cannot be empty")
    if np.any(p_raw < 0):
        raise ValueError("D0_composition must be non-negative")
    p_sum = p_raw.sum()
    if p_sum <= 0:
        raise ValueError("D0_composition must sum to > 0")

    # normalize input (so we work with fractions)
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

    return Mn_ODE_4, Mw_ODE_4