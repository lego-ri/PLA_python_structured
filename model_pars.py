""" 
    Parameters of batch L,L-lactide ROP
    After Yu Y., Storti G., Morbidelli M., Ind. Eng. Chem. Res, 2011, 50, 7927-7940
"""

""" 
    Used in: 
    main_process
"""
import numpy as np

class ModelPars:
    #* General physical-chemical constants
    R = 8.314                     # Universal gas constant, J/K/mol
    MW = 72.06e-3                 # Molecular weight of lactoyl (monomer) group, kg/mol

    #*  Reaction conditions + initial concentrations
    T = 155                      # Reaction temperature, deg. C
    T = T + 273.15               # Temperature conversion to K
    M0 = 1e4                     # Init. conc. of monomer, mol/m3
    C0 = M0 / 10000              # Init. conc. of catalyst, mol/m3
    D0 = C0 * 60                 # Init. conc. of cocatalyst ~ , mol/m3
    A0 = C0 * 0.36               # Init. conc. of acid, mol/m3 (impurity, side product of catalytic activation)

    #* Kinetic parameters
    # Activation & Deactivation (for all temperatures = temperature independant)
    k_a1 = 1e6 * 1e-3 / 3600                    # Activation rate coefficient, m3/mol/s
    Keq_a = 0.256                               # Activation equilibrium constant, -
    Keq_a_article = np.exp(-6029.3/T + 11.943)  # at 155 deg.C should be 0.256??? but isnt
    k_a2 = k_a1 / Keq_a                         # Deactivation rate coefficient, m3/mol/s

    # Propagation & Depropagation
    k_p0 = 7.4e11 * 1e-3 / 3600         # Preexponential factor, m3/mol/s
    Ea_p = 63.3 * 1e3                   # Activation energy, J/mol
    k_p = k_p0 * np.exp(-Ea_p / (R*T))  # Propagation rate coefficient, m3/mol/s
    Meq = 0.225 * 1e3                   # Monomer equilibirum constant, mol/m3
    k_d = k_p * Meq                     # Depropagation rate coefficient, 1/s

    # Chain-transfer (for all temperatures)
    k_s = 1e6 * 1e-3 / 3600       # Chain-transfer rate coefficient m3/mol/s
   

    # Intermolecular transesterification
    k_te0 = 3.38e11 * 1e-3 / 3600           # Preexponential factor, m3/mol/s
    Ea_te = 83.3 * 1e3                      # Activation energy, J/mol
    k_te = k_te0 * np.exp(-Ea_te / (R*T))    # Intermolecular transesterification rate coefficient, m3/mol/s
   

    # Nonradical random chain scission
    k_de0 = 1.69e8 / 3600                   # Preexponential factor, 1/s
    Ea_de = 101.5 * 1e3                     # Activation energy, J/mol
    k_de = k_de0 * np.exp(-Ea_de / (R*T))   # Nonradical random chain scission rate coefficient, 1/s