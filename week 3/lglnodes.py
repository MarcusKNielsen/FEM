import numpy as np
from numpy import finfo

def lglnodes(N):
    # Truncation + 1
    N1 = N + 1
    
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(N + 1) / N)
    
    # The Legendre Vandermonde Matrix
    P = np.zeros((N1, N1))
    
    # Tolerance for the Newton-Raphson method
    eps = finfo(float).eps
    
    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2
    while np.max(np.abs(x - xold)) > eps:
        xold = x

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(2, N1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        x = xold - (x * P[:, N] - P[:, N - 1]) / (N1 * P[:, N1 - 1])

    w = 2 / (N * N1 * P[:, N1 - 1] ** 2)

    return x, w, P



x, w, V = lglnodes(5)

print(x)
#print(w)
print(V)