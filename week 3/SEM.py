import numpy as np
from LGL_nodes import legendre_gauss_lobatto_nodes as lglnodes


def conelmtab(x0, L, noelms):
    VX = np.linspace(x0, x0+L, noelms+1)
    idxs = np.arange(noelms+1)
    EToV = np.vstack((idxs[:-1], idxs[1:])).T

    return VX, EToV



p = 1

x0 = 0
L = 1
noelms = 2

VX, EToV = conelmtab(x0, L, noelms)

def new_assembly(VX, EToV, D=1, qt=None, t=None,p=1):
    M = len(EToV[:,1])
    N = M+1
    A = np.zeros((N,N))
    C = np.zeros((N,N))
    b = np.zeros(N)

    V,Vr,x = lglnodes(p,-1,1)
    M = np.linalg.inv(V@V.T)        #(3.49)
    Dr = Vr@np.linalg.inv(V)        #(3.56)
    DrMDr = Dr@M@Dr


    for elm in EToV:
        i = elm[0]
        j = elm[1]
        xi = VX[i]
        xj = VX[j]

        h = xj-xi


        Mn = h/2*M            #(3.50)
        Kn = (2/h)*DrMDr      #(3.59)

    return M,Mn, Kn

M,Mn, Kn = new_assembly(VX, EToV, D=1, qt=None, t=None,p=1)
