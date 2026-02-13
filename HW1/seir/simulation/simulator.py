from typing import Tuple
from __future__ import annotations
import numpy as np
import numpy.typing as npt

Parameters = Tuple[float, float, float] # (beta, sigma, gamma)
InitialConditions = Tuple[float, float, float, float] # (S0, E0, I0, R0)
SEIRSeries = Tuple[npt.NDArray[np.float_], 
                   npt.NDArray[np.float_], 
                   npt.NDArray[np.float_], 
                   npt.NDArray[np.float_]] # (S, E, I, R)
def simulate_seir(
    parameters: Parameters,
    init_conditions: InitialConditions,
    days: int = 51,
) -> SEIRSeries:

    # Extract parameters and initial conditions (Your code here)
    beta, sigma, gamma = parameters
    S0, E0, I0, R0 = init_conditions
    N = S0 + E0 + I0 + R0
    S = np.empty(days, dtype=np.float64)
    E = np.empty(days, dtype=np.float64)
    I = np.empty(days, dtype=np.float64)
    R = np.empty(days, dtype=np.float64)

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # For each day, perform SEIR update
    for t in range(1, days):

        # Compute new cases and update equations
        E_new = (beta*S[t-1]*I[t-1]) / N
        I_new = sigma*E[t-1]
        R_new = gamma*I[t-1]
      
        # Update equations
        S[t] = S[t-1] - E_new
        E[t] = E[t-1] + E_new - I_new
        I[t] = I[t-1] + I_new - R_new
        R[t] = R[t-1] + R_new

    return (S, E, I, R)
