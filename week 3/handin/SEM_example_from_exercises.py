#%%
from SEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 15
p = 3
x0 = 0
L = 1.5
VX,VX_fine, C = construct_c(n, p, x0, L)
D = 1
t0 = 0
f = [0,0]

VX_fine = np.array(VX_fine)


def u_true(x,t):
    return (np.exp(-800*(x - 0.4)**2) + 0.25*np.exp(-40*(x - 0.8)**2))*np.exp(-t)

def qt(t,x):
    return -(np.exp(-800*(x - 0.4)**2) + 0.25*np.exp(-40*(x - 0.8)**2))*np.exp(-t) - (1600.0*(x - 0.8)**2*np.exp(-40*(x - 0.8)**2) + 2560000*(x - 0.4)**2*np.exp(-800*(x - 0.4)**2) - 1600*np.exp(-800*(x - 0.4)**2) - 20.0*np.exp(-40*(x - 0.8)**2))*np.exp(-t)

un = u_true(VX_fine,0)

theta = 1.0
dt = 0.001

unext = oneit(VX, VX_fine, C, f, D, qt, t0, dt, un, theta, p)
x = np.linspace(x0,L,100)
plt.figure()
plt.plot(VX_fine,un,label="u0")
plt.plot(VX_fine,unext,'-o',label="unext")
plt.plot(x,u_true(x,dt), label="True")
plt.legend()
plt.show()

#%%

unext = u_true(VX_fine,0)
n = 200
plt.figure()
t = 0
T = 0.5
dt = T/n
k = 0
while t < T:
    if k % 40 == 0:
        plt.plot(VX_fine,unext,label=f"t = {np.round(t,2)}")
        plt.plot(VX_fine,u_true(VX_fine,t),"--",label=f"t = {np.round(t,2)}")
    unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)
    t += dt
    k += 1
plt.legend()
plt.show()


#%% Time plots

N = list(range(2,50))
error = np.zeros(len(N))


for i,n in enumerate(N):

    unext = un
    
    t = 0
    T = 0.5
    dt = T/n
    while t < T:
        unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

h = T/np.array(N)

plt.figure()
plt.plot(np.log(h),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Step Size: dt")
plt.show()

a,b = np.polyfit(np.log(h)[-20:],np.log(error)[-20:],1)

#%% Time convergence test

N = list(range(1,20))
error = np.zeros(len(N))

dt = 0.00001

for i,n in enumerate(N):
    
    VX,VX_fine, C = construct_c(n, p, x0, L)
    
    unext = np.sin(VX_fine)
    
    t = 0
    T = dt
    while t < T:
        unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 


h = L/np.array(N)


a,b = np.polyfit(np.log10(h)[:],np.log10(error)[:],1)

plt.figure()
plt.plot(np.log(h),error,label="error")
plt.plot(np.log(h),np.log(h),label=r"$O(h^2)$")
plt.ylabel("log(Error)")
plt.xlabel("log(h)")
plt.title(f"Convergence test for p = {p}. a = {a:.2f}")
plt.legend()
plt.show()




# %%
