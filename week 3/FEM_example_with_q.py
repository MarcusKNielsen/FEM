from FEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 150
VX, EToV = conelmtab(0,np.pi,n)
L = np.pi
D = 1

t0 = 0
f = [0,0]

def u_true(x,t):
    return np.sin(x)*np.exp(-np.pi**2 * t) + 0.1*x*(np.pi - x)*np.exp(x)*t

def qt(t,x):
    k = 0.1
    term1 = (1-np.pi**2) * np.sin(x)*np.exp(- np.pi**2 * t)
    term2 = k*x*( (np.pi - x) - (np.pi-x-1)*t ) * np.exp(x)
    term3 = k*(2 - 2*(np.pi-x) + x)*np.exp(x)*t
    return term1 + term2 + term3

un = u_true(VX,0)

theta = 1.0
dt = 0.01

unext = oneit(VX, EToV, f, D, qt, t0,dt,un,theta)

plt.figure()
plt.plot(VX,un,label="u0")
plt.plot(VX,unext,label="unext")
plt.plot(VX,u_true(VX,dt), label="True")
plt.legend()
plt.show()

#%%
unext = u_true(VX,0)
n = 200
plt.figure()
t = 0
T = 0.5
dt = T/n
k = 0
while t < T:
    if k % 40 == 0:
        plt.plot(VX,unext,label=f"t = {np.round(t,2)}")
        plt.plot(VX,u_true(VX,t),"--",label=f"t = {np.round(t,2)}")
    unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)
    t += dt
    k += 1
plt.legend()
plt.show()


#%% Time Convergence test


N = list(range(2,50))
error = np.zeros(len(N))


for i,n in enumerate(N):

    unext = un
    
    t = 0
    T = 0.5
    dt = T/n
    while t < T:
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)    
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

h = T/np.array(N)

plt.figure()
plt.plot(np.log(h),np.log(error),label="error")
plt.ylabel("Error")
plt.xlabel("Step Size: dt")
plt.show()

a,b = np.polyfit(np.log(h)[-20:],np.log(error)[-20:],1)

#%%

N = list(range(10,150,10))
error = np.zeros(len(N))

dt = 0.00001

for i,n in enumerate(N):
    
    VX, EToV = conelmtab(0,np.pi,n)
    
    unext = np.sin(VX)
    
    t = 0
    T = dt
    while t < T:
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)    
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

h = L/np.array(N)

plt.figure()
plt.plot(np.log(h),np.log(error),label="error")
plt.plot(np.log(h),2*np.log(h),label=r"$O(h^2)$")
plt.ylabel("log(Error)")
plt.xlabel("log(h)")
plt.legend()
plt.show()

a,b = np.polyfit(np.log(h),np.log(error)[:],1)














