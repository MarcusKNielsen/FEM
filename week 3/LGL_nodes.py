from scipy.special import legendre
import numpy as np

def legendre_gauss_lobatto_nodes(n, a, b):
    """
    Compute the Legendre-Gauss-Lobatto nodes for the interval [a, b].

    Parameters:
    n (int): The order of the Legendre polynomial (number of nodes).
    a, b (float): The endpoints of the interval.

    Returns:
    np.ndarray: The Legendre-Gauss-Lobatto nodes in the interval [a, b].
    """
    # Get the Legendre polynomial of degree n
    Pn = legendre(n)

    # Find its derivative
    Pn_deriv = np.polyder(Pn)

    # Compute the roots of the derivative
    roots = np.roots(Pn_deriv)

    # Include the endpoints, which are also LGL nodes
    roots = np.append(roots, [-1, 1])

    # Remove any complex parts due to numerical errors (should be very small)
    roots = np.real(roots)

    # Sort the roots
    roots.sort()

    # Transform the roots to the interval [a, b]
    return 0.5 * (a + b) + 0.5 * (b - a) * roots

# Example usage
n = 5  # Degree of the Legendre polynomial
a, b = -1, 1  # Interval [a, b]
lgl_nodes = legendre_gauss_lobatto_nodes(n, a, b)
lgl_nodes
