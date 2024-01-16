from scipy.special import legendre, eval_legendre
import numpy as np
n = 2
a = -1
b = 1

def legendre_gauss_lobatto_nodes(n,a,b):
    """
    Compute the Legendre-Gauss-Lobatto nodes for the interval [-1, 1].

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
    x = np.sort(roots)
    x = 0.5*(b-a)*x + 0.5*(b+a)

    V = np.zeros((n+1,n+1))
    Vr = np.zeros((n+1,n+1))

    for i in range(n+1):
        Pi = legendre(i)
        Pid = Pi.deriv()
        V[:,i] = Pi(x)
        Vr[:,i] = Pid(x)

   
    return V,Vr,x
