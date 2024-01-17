from FEM import *
import numpy as np
from matplotlib import pyplot as plt
import time

## Testcase:
N  = 10
#p  = 3
L  = np.pi
D  = 1
f  = [0,0]
x0 = 0
theta = 1.0
t = 0


VX, EToV = conelmtab(x0, L, N)


def u_true(x,t):
    return np.sin(x)*np.exp(-np.pi**2 * t) + 0.1*x*(np.pi - x)*np.exp(x)*t

def qt(t,x):
    k = 0.1
    term1 = (1-np.pi**2) * np.sin(x)*np.exp(- np.pi**2 * t)
    term2 = k*x*( (np.pi - x) - (np.pi-x-1)*t ) * np.exp(x)
    term3 = k*(2 - 2*(np.pi-x) + x)*np.exp(x)*t
    return term1 + term2 + term3

u0 = u_true(VX,0)

R, S, b = assembly1D(VX, EToV, D, qt, t)

#%%

unext = u0
dt = 0.1
T = 1
t = 0
while t < T:
    unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)
    t += dt


plt.figure()
plt.plot(VX,unext,label="unext")
plt.plot(VX,u_true(VX,t), label="True")
plt.legend()
plt.show()

#%% Time Convergence Test

Nt = list(range(25,50))
error = np.zeros(len(Nt))

for i,n in enumerate(Nt):

    unext = u0
    
    t = 0
    T = 0.5
    dt = T/n
    while t < T:
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

dt = T/np.array(Nt)

plt.figure()
plt.plot(np.log(dt),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Step Size: dt")
plt.show()

a,b = np.polyfit(np.log(dt),np.log(error),1)

#%% Spatial Convergence Test

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

    VX, EToV = conelmtab(x0, L, ns)
    unext = u_true(VX,0)
    
    t = 0
    T = dt
    while t < T:
        start_time = time.time()  # start timer
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)
        end_time = time.time()  # end timer
        t += dt
    times[i] = end_time - start_time  # store computation time
    error[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

print()
DGF = np.array(Ns)*p+1

plt.figure()
plt.plot(np.log(DGF),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Degrees of Freedom")
plt.show()

a,b = np.polyfit(np.log(DGF),np.log(error),1)


