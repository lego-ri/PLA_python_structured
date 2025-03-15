import numpy as np

class SelReac:
    """
    Class to semi-randomly select a chemical reaction to happen using a list
    of individual reaction rates stored in Rvec and calculating the probabilities of each reaction.
    Output: r_idx = index of selected reaction
    """

    @classmethod
    def sel_reac(cls, Rvec, reac_num):
        """ Evaluate reaction probabilities """
        Rvec = np.array(Rvec)                      # Vector of reaction rates (ensure it is a NumPy array)
        Pvec = Rvec / np.sum(Rvec)                 # Calculate the probability of each reaction from its reac. rate

        """ Generate a random number between 0 and 1 """
        rnd = np.random.rand()

        """ Scan the probabilities """
        P_sum = 0                                  # Initiate the cumulative probability
        for i in range(reac_num):
            P_sum += Pvec[i]                       # Add the prob. of each reac. to the cumulative probability
            if P_sum > rnd:                        # Condition which chooses the reaction
                r_idx = i                          # Store the reaction index
                break
        else:
            r_idx = -1                             # Default value if no reaction is chosen

        if np.isnan(Pvec).any():                   # Alarm if the code is wrong, by handling the r_idx in the code
            r_idx = -1

        return r_idx