import numpy as np
from matplotlib import pyplot as plt
from SEM import *

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