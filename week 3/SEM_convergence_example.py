import numpy as np
from LGL_nodes import legendre_gauss_lobatto_nodes as lglnodes
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import numpy as np

def construct_c(N,p, x0, L):
    C = np.zeros((N,p+1), dtype=int)
    gidx = 1
    for n in range(N):
        gidx -= 1
        for i in range(p+1):
            C[n,i] = gidx
            gidx += 1
    VX = np.linspace(x0, x0+L, N+1)
    
    VX_fine = [VX[0]] 
    _,_,x = lglnodes(p)
    for n in range(N):
        VX_fine.extend(VX[n] + (x[1:] + 1) * (VX[n+1] - VX[n]) / 2)

    return VX, VX_fine, C

#def construct_new_VX(VX,x):

def new_assembly(VX, VX_fine, C, qt=None,p=1):
    M = len(C[:,0])*p+1
    N = len(C[:,0])
    A = np.zeros((M,M))
    b = np.zeros(M)

    V,Vr,_ = lglnodes(p)
    h = (VX[-1] - VX[0])/N
    M_cal = np.linalg.inv(V@V.T)    #(3.49)
    Dr = Vr@np.linalg.inv(V)        #(3.56)
    Kn = (2/h)*Dr.T@M_cal@Dr        #(3.59)
    Mn = h/2*M_cal                  #(3.50)

    for n in range(N):
        for j in range(p+1):
            for i in range(p+1):
                idx1 = C[n,i]
                idx2 = C[n,j]
                xi = VX_fine[idx1]
                xj = VX_fine[idx2]
                if i < j:
                    hi = xj - xi

                    qtilde = hi/4*(qt(xi)+qt(xj))
                    b[[i,j]] += qtilde

                A[idx1, idx2] += Kn[i,j] + Mn[i,j]

    return A,b

def dirbc1D(f, A, b):
    # Boundary conditions
    temp = A[1:p+1,0]*f[0]
    
    b[0] = f[0]

    # Modify R and S
    b[1:p+1] -= temp

    A[:p+1,0] = A[0,:p+1] = 0
    A[0,0] = 1
    
    temp = A[-1,-(p+1):-1]*f[-1]

    b[-1] = f[-1]

    b[-(p+1):-1] -= temp
    
    A[-1,-(p+1):-1] = A[-(p+1):-1,-1] = 0
    A[-1,-1] = 1
    

    return A,b

## Testcase:
N = 2
p=2
L = 2
x0 = 0
VX, VX_fine, C = construct_c(N,p,x0, L)
D = 1
qt = lambda x: 0
f = [1,np.exp(2)]


A, b = new_assembly(VX, VX_fine, C, qt,p)
A, b = dirbc1D(f, A, b)

u = np.linalg.solve(A,b)

# A,C,b, VX_fine = new_assembly(VX, C, D, qt, t0, p)

x_grid = np.linspace(x0, x0+L, 1000)
plt.figure()
plt.plot(VX_fine,u,label="u0")
plt.plot(x_grid,np.exp(x_grid), label="True")
plt.legend()
plt.show()


def convergence_test(N,p,x0,L,qt,f, u):
    VX, VX_fine, C = construct_c(N,p,x0, L)
    A, b = new_assembly(VX, VX_fine, C, qt,p)
    A, b = dirbc1D(f, A, b)

    uhat = np.linalg.solve(A,b)

    return np.linalg.norm(uhat-u(VX_fine),np.inf)

# Define the parameters
L = 2
x0 = 0
D = 1
qt = lambda x: 0
f = [1, np.exp(2)]
u = lambda x: np.exp(x)

Ne = list(range(2, 15, 5))

p_values = [1, 2, 3, 4, 5, 6]

plt.figure()

for p in p_values:
    H = L / (np.array(Ne)*p+1)
    error = np.zeros(len(Ne))
    for i, n in enumerate(Ne):
        error[i] = convergence_test(n, p, x0, L, qt, f, u)
        
    DGF = np.array(Ne) * p + 1

    a, b = np.polyfit(np.log10(DGF), np.log10(error), 1)

    plt.plot(np.log10(DGF), np.log10(error), label=f"p = {p}, a = {a:.2f}")

plt.title("Convergence Test")
plt.ylabel("Error (log scale)")
plt.xlabel("Degrees of Freedom (log scale)")
# plt.axhline(y=np.log(1e-6), color='r', linestyle='--', label="Error bound")
plt.legend()
plt.show()

# Test CPU time for error bound
L = 2
x0 = 0
D = 1
qt = lambda x: 0
f = [1, np.exp(2)]
u = lambda x: np.exp(x)

Ne = list(range(2, 1000))
H = L / np.array(Ne)
p_values = [1, 2, 3, 4, 5, 6]
times = np.zeros(len(p_values))
elements = np.zeros(len(p_values))
DFG = np.zeros(len(p_values))
errors = np.zeros(len(p_values))
for k, p in enumerate(p_values):
    
    #for i, n in enumerate(Ne):
    start_time = time.time()
    error = 1
    i = 0
    while error > (1e-6):
        error = convergence_test(Ne[i], p, x0, L, qt, f, u)
        i += 1
    errors[k] = error
    end_time = time.time()
    elements[k] = i
    times[k] = end_time - start_time
    DFG[k] = i*p + 1

P = 63 #W
C = 0.285 #kgCO2eq/kWh

CO2eq = times/3600*P/1000*C

