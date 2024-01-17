import numpy as np
from matplotlib import pyplot as plt
from SEM import *

## Testcase:
N  = 10
p  = 1
L  = np.pi
D  = 1
f  = [0,0]

t = 0

theta = 1.0

VX, VX_fine, C = construct_c(N,p,x0 = 0, L = L)

VX_fine = np.array(VX_fine)


def u_true(x,t):
    return np.sin(x)*np.exp(-np.pi**2 * t) + 0.1*x*(np.pi - x)*np.exp(x)*t

def qt(t,x):
    k = 0.1
    term1 = (1-np.pi**2) * np.sin(x)*np.exp(- np.pi**2 * t)
    term2 = k*x*( (np.pi - x) - (np.pi-x-1)*t ) * np.exp(x)
    term3 = k*(2 - 2*(np.pi-x) + x)*np.exp(x)*t
    return term1 + term2 + term3

R,S,b = new_assembly(VX, VX_fine, C, D, qt, t, p)

#%%

u0 = u_true(VX_fine,0)

unext = u_true(VX_fine,0)
dt = 0.1
T = 1
t = 0
while t < T:
    unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)
    t += dt


plt.figure()
plt.plot(VX_fine,unext,label="unext")
plt.plot(VX_fine,u_true(VX_fine,t), label="True")
plt.legend()
plt.show()


#%% Time Convergence Test

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

