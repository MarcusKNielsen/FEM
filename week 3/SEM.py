import numpy as np
from LGL_nodes import legendre_gauss_lobatto_nodes as lglnodes

def construct_c(N,p, x0, L):
    C = np.zeros((N,p+1), dtype=int)
    gidx = 2
    for n in range(N):
        gidx -= 1
        for i in range(p+1):
            C[n,i] = gidx
            gidx += 1
    VX = np.linspace(x0, x0+L, N+1)
    return VX, C

#def construct_new_VX(VX,x):

def new_assembly(VX, C, D=1, qt=None, t=None,p=1):
    M = len(VX)
    R = np.zeros((M,M))
    S = np.zeros((M,M))
    b = np.zeros(M)

    V,Vr,x = lglnodes(p)
    M_cal = np.linalg.inv(V@V.T)        #(3.49)
    Dr = Vr@np.linalg.inv(V)        #(3.56)
    DrMDr = Dr.T@M_cal@Dr
    h = (VX[-1] - VX[0])/(M-1)

    for n in range(M-1):        
        Mn = h/2*M_cal            #(3.50)
        Kn = (2/h)*DrMDr          #(3.59)

        for j in range(p+1):
            for i in range(j):
                idx1 = C[n,i]
                idx2 = C[n,j]
                xi = VX[idx1]
                xj = VX[idx2]
                hi = xj - xi

                qtilde = hi/4*(qt(t,xi)+qt(t,xj))
                b[[i,j]] += qtilde

                R[idx1, idx2] += Kn[i,j]
                S[idx1, idx2] += Mn[i,j]

    return R,S,b

## Testcase:
n = 5
p=1
VX, C = construct_c(n,p,x0 = 0, L = np.pi)
D = 1
qt = lambda t,x: 0
t0 = 0
f = [0,0]
un = np.sin(VX)
theta = 1.0
dt = 0.1


A,C,b = new_assembly(VX, C, D, qt, t0, p)

C