import math

import numpy as np
import scipy


# Reduce the number of iterations for faster optimization
def reconstruction_error_faster(params):
    alpha, = params

    errors = []
    for _ in range(1000):  # Reduced iterations for faster computation
        # Generate random skew-symmetric matrix S
        S = np.array([
            [0, np.random.random(), np.random.random(), np.random.random()],
            [0, 0, np.random.random(), np.random.random()],
            [0, 0, 0, np.random.random()],
            [0, 0, 0, 0]
        ])
        S = S - S.T
        S = S * math.pi / 4

        # Define deltaS
        deltaS = np.array([
            [0, alpha, alpha, -1],
            [0, 0, 0, -alpha],
            [0, 0, 0, -alpha],
            [0, 0, 0, 0]
        ]) / 100
        deltaS = deltaS - deltaS.T

        # Compute S_prime
        S_prime = S + deltaS*S

        # Compute matrix exponentials
        exp_S = scipy.linalg.expm(S)
        exp_S_new = scipy.linalg.expm(S_prime)

        # Compute reconstruction error
        error = ((exp_S - exp_S_new)**2).sum()
        errors.append(error)

    return np.mean(errors)

# Initial guess for the parameters
initial_guess = [1/4]
bounds = [(0.1, 1/2)]

# Run optimization with reduced complexity
result = scipy.optimize.minimize(reconstruction_error_faster, x0=initial_guess, method='Powell', bounds=bounds)

# Optimal parameters
print(result.x, result.fun)
