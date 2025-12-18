""" 
    Model equations of batch L,L-lactide ROP (Set of ODEs)
    After Yu Y., Storti G., Morbidelli M., Ind. Eng. Chem. Res, 2011, 50, 7927-7940
"""

""" 
    Used in: 
    main_process
"""

import numpy as np # Python numerical library


def model_eqs(t, y, pars):
    #* Unpack State variables
    M = y[0]        # Monomer concentration, mol/m3
    C = y[1]        # Catalyst concentration, mol/m3
    A = y[2]        # Acid concentration, mol/m3
    la0 = y[3]      # 0th moment of active chains, mol/m3
    la1 = y[4]      # 1st moment of active chains, mol/m3 
    la2 = y[5]      # 2nd moment of active chains, mol/m3
    mu0 = y[6]      # 0th moment of dormant chains, mol/m3
    mu1 = y[7]      # 1st moment of dormant chains, mol/m3
    mu2 = y[8]      # 2nd moment of dormant chains, mol/m3
    ga0 = y[9]      # 0th moment of terminated chains, mol/m3
    ga1 = y[10]     # 1st moment of terminated chains, mol/m3
    ga2 = y[11]     # 2nd moment of terminated chains, mol/m3

    #* Unpack Model parameters
    k_a1 = pars.k_a1        # Activation rate coefficient, m3/mol/s
    k_a2 = pars.k_a2        # Deactivation rate coefficient, m3/mol/s
    k_p = pars.k_p          # Propagation rate coefficient, m3/mol/s
    k_d = pars.k_d          # Depropagation rate coefficient, 1/s
    k_s = pars.k_s          # Chain-transfer rate coefficient m3/mol/s
    k_te = pars.k_te        # Intermolecular transesterification rate coefficient, m3/mol/s
    k_de = pars.k_de        # Nonradical random chain scission rate coefficient, 1/s    

    #* Moment closure equations - 3rd momenets
    la3 = la2 * (2 * la2 * la0 - la1**2) / (la1 * la0)
    mu3 = mu2 * (2 * mu2 * mu0 - mu1**2) / (mu1 * mu0)
    ga3 = ga2 * (2 * ga2 * ga0 - ga1**2) / (ga1 * ga0)

    #****************************************************************
    #* Model equations
    # Initialize derivatives vector
    dy = np.zeros(12)

    # Balance of monomer (M) - lactide
    dy[0] = - k_p * M * la0 + k_d * la0 

    # Balance of catalyst (C) - tin(II) octoate
    dy[1] = - k_a1 * C * mu0 + k_a2 * A * la0

    # Balance of octanoic acid (A)
    dy[2] = k_a1 * C * mu0 - k_a2 * A * la0

    # Balance of 0th moment (concentration) of active chains 
    dy[3] = k_a1 * mu0 * C - k_a2 * la0 * A

    # Balance of 1st moment of active chains 
    dy[4] = (k_a1 * mu1 * C - k_a2 * la1 * A + 2 * k_p * M * la0 - 2 * k_d * la0
             - k_s * la1 * mu0 + k_s * mu1 * la0 - k_te * la1 * (mu1 - mu0)
             + (1/2) * k_te * la0 * (mu2 - mu1) - k_te * la1 * (ga1 - ga0)
             + (1/2) * k_te * la0 * (ga2 - ga1) - (1/2) * k_de * (la2 - la1))

    # Balance of 2nd moment of active chains
    dy[5] = (k_a1 * mu2 * C - k_a2 * la2 * A + 4 * k_p * M * (la1 + la0)
             + 4 * k_d * (la0 - la1) - k_s * la2 * mu0 + k_s * mu2 * la0
             + (1/3) * k_te * la0 * (la1 - la3) + k_te * la1 * (la2 - la1)
             - k_te * la2 * (mu1 - mu0) + (1/6) * k_te * la0 * (2 * mu3 - 3 * mu2 + mu1)
             - k_te * la2 * (ga1 - ga0) + (1/6) * k_te * la0 * (2 * ga3 - 3 * ga2 + ga1)
             - (1/6) * k_de * (4 * la3 - 3 * la2 - la1))

    # Balance of 0th moment (concentration) of dormant (all OH-bearing) species
    dy[6] = - k_a1 * mu0 * C + k_a2 * la0 * A

    # Balance of 1st moment of dormant chains
    dy[7] = (- k_a1 * mu1 * C + k_a2 * la1 * A + k_s * la1 * mu0 - k_s * mu1 * la0
             + k_te * la1 * (mu1 - mu0) - (1/2) * k_te * la0 * (mu2 - mu1)
             - (1/2) * k_de * (mu2 - mu1))

    # Balance of 2nd moment of dormant chains
    dy[8] = (- k_a1 * mu2 * C + k_a2 * la2 * A + k_s * la2 * mu0 - k_s * mu2 * la0
             + k_te * la2 * (mu1 - mu0) + k_te * la1 * (mu2 - mu1)
             + (1/6) * k_te * la0 * (-4 * mu3 + 3 * mu2 + mu1)
             - (1/6) * k_de * (4 * mu3 - 3 * mu2 - mu1))

    # Balance of 0th moment (concentration) of terminated chains
    dy[9] = k_de * (la1 - la0) + k_de * (mu1 - mu0)

    # Balance of 1st moment of terminated chains
    dy[10] = (k_te * la1 * (ga1 - ga0) - (1/2) * k_te * la0 * (ga2 - ga1)
              - k_de * (ga2 - ga1) + (1/2) * k_de * (la2 - la1)
              + (1/2) * k_de * (mu2 - mu1))

    # Balance of 2nd moment of terminated chains
    dy[11] = (k_te * la2 * (ga1 - ga0) + k_te * la1 * (ga2 - ga1)
              + (1/6) * k_te * la0 * (-4 * ga3 + 3 * ga2 + ga1)
              - (1/3) * k_de * (4 * ga3 - 3 * ga2 - ga1)
              + (1/6) * k_de * (2 * la3 - 3 * la2 + la1)
              + (1/6) * k_de * (2 * mu3 - 3 * mu2 + mu1))

    return dy