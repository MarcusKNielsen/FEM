import numpy as np
from LGL_nodes import legendre_gauss_lobatto_nodes as lglnodes
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

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

def new_assembly(VX, VX_fine, C, D=1, qt=None, t=None,p=1):
    M = len(C[:,0])*p+1
    N = len(C[:,0])
    R = np.zeros((M,M))
    S = np.zeros((M,M))
    b = np.zeros(M)

    V,Vr,x = lglnodes(p)
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

                    qtilde = hi/4*(qt(t,xi)+qt(t,xj))
                    b[[i,j]] += qtilde

                R[idx1, idx2] += Kn[i,j]
                S[idx1, idx2] += Mn[i,j]

    return R,S,b

def dirbc1D(f, R,S, b):
    d = np.zeros(len(b))

    # Boundary conditions
    temp = R[1:p+1,0]*f[0]
    
    b[0] = d[0] = f[0]

    # Modify R and S
    b[1:p+1] -= temp
    d[1:p+1] += temp

    R[0,0] = R[:p+1,0] = R[0,:p+1] = 0

    

    S[:p+1,0] = S[0,:p+1] = 0
    S[0,0] = 1
    
    temp = R[-1,-(p+1):-1]*f[-1]

    b[-1] = d[-1] = f[-1]

    b[-(p+1):-1] -= temp
    d[-(p+1):-1] += temp

    R[-1,-(p+1):] = R[-(p+1):,-1]=  0
    
    S[-1,-(p+1):-1] = S[-(p+1):-1,-1] = 0
    S[-1,-1] = 1
    

    return R,S,b,d
   
def RS_1D(R,S,dt,d2):
    S -= d2*R
    R = S + dt*R 

    return R,S


# Step 4 Factor but we don't.

def construct_e(S,b,un,d2):
    return S@un + d2*b


# Step 7 + 8
def advance_b(d, VX_fine, C, qt,tnext, p):
    bnext = np.zeros(len(d))
    N = len(C[:,0])

    for n in range(N):
        for j in range(p+1):
            for i in range(j):
                idx1 = C[n,i]
                idx2 = C[n,j]
                xi = VX_fine[idx1]
                xj = VX_fine[idx2]
                hi = xj - xi

                qtilde = hi/4*(qt(tnext,xi)+qt(tnext,xj))
                bnext[[i,j]] += qtilde

    bnext[0] = bnext[-1] = 0
    bnext -= d

    return bnext

# Step 9
def update_e(e,bnext,d1):
    return e + d1*bnext

def oneit(VX, VX_fine, C, f, D, qt, t0, dt, un, theta, p):

    d1, d2 = dt*theta, dt*(1-theta)

    # Step 1: Assemble R, S and b
    R,S,b = new_assembly(VX, VX_fine, C, D, qt, t0, p)

    # Step 2: Dirichlet BC
    R,S, b, d = dirbc1D(f, R, S, b)

    # Compute d1, d2

    # Step 3: Overwrite R and S
    R,S = RS_1D(R,S,dt,d2)

    # Step 5 + 6: Construct e

    e = construct_e(S,b,un,d2)

    # Step 7 + 8: Advance b
    b = advance_b(d,VX_fine, C, qt,t0+dt, p)

    # Step 9: compute e
    e = update_e(e,b,d1)


    # Step 10: Solve!
    unext = np.linalg.solve(R,e)

    return unext

## Testcase:
N = 5
p=3
L = np.pi
VX, VX_fine, C = construct_c(N,p,x0 = 0, L = L)
D = 1
qt = lambda t,x: 0
t0 = 0
f = [0,0]
theta = 1.0
dt = 0.1
u0 = np.sin(VX_fine)


# A,C,b, VX_fine = new_assembly(VX, C, D, qt, t0, p)



unext = oneit(VX, VX_fine, C, f, D, qt, t0,dt,u0,theta, p)

plt.figure()
plt.plot(VX_fine,u0,label="u0")
plt.plot(VX_fine,unext,label="unext")
plt.plot(VX_fine,u0*np.exp(-dt), label="True")
plt.legend()
plt.show()


# Time Convergence Test

Nt = list(range(2,50))
error = np.zeros(len(Nt))
def u_true(x,t):
    return np.sin(x)*np.exp(-D*t)


for i,n in enumerate(Nt):

    unext = u0
    
    t = 0
    T = 0.5
    dt = T/n
    while t < T:
        unext = oneit(VX, VX_fine, C, f, D, qt, t,dt,unext,theta, p)
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

h = T/np.array(Nt)

plt.figure()
plt.plot(np.log(h),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Step Size: dt")
plt.show()

a,b = np.polyfit(np.log(h)[-20:],np.log(error)[-20:],1)

#Spatial Convergence Test
L = np.pi
D = 1
qt = lambda t,x: 0
t0 = 0
f = [0,0]
theta = 1.0
dt = 0.1


p = 1
Ns = list(range(2,20))
error = np.zeros(len(Ns))
times = np.zeros(len(Ns))

def u_true(x,t):
    return np.sin(x)*np.exp(-D*t)


for i,ns in enumerate(Ns):

    VX, VX_fine, C = construct_c(ns,p,x0 = 0, L = np.pi)
    unext = u_true(VX_fine,0)
    
    t = 0
    T = dt
    while t < T:
        start_time = time.time()  # start timer
        unext = oneit(VX, VX_fine, C, f, D, qt, t,dt,unext,theta, p)
        end_time = time.time()  # end timer
        t += dt
    times[i] = end_time - start_time  # store computation time
    error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

print()
DGF = np.array(Ns)*p+1

plt.figure()
plt.plot(np.log(DGF),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Degrees of Freedom")
plt.show()

a,b = np.polyfit(np.log(DGF),np.log(error),1)