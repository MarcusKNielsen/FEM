from FEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 150
VX, EToV = conelmtab(0,np.pi,n)
L = np.pi
D = 1
qt = lambda t,x: 0

t0 = 0
f = [0,0]

un = np.sin(VX)

theta = 1.0
dt = 0.1

unext = oneit(VX, EToV, f, D, qt, t0,dt,un,theta)

plt.figure()
plt.plot(VX,un,label="u0")
plt.plot(VX,unext,label="unext")
plt.plot(VX,un*np.exp(-dt), label="True")
plt.legend()
plt.show()


#%% Time Convergence test


def u_true(x,t):
    return np.sin(x)*np.exp(-D*t)


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
plt.xlabel("Error")
plt.ylabel("Step Size: dt")
plt.show()

a,b = np.polyfit(np.log(h)[-20:],np.log(error)[-20:],1)

#%%

N = list(range(2,20))
error = np.zeros(len(N))

dt = 0.01

for i,n in enumerate(N):
    
    VX, EToV = conelmtab(0,np.pi,n)
    
    unext = np.sin(VX)
    
    t = 0
    T = 0.5
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

a,b = np.polyfit(np.log(h)[:10],np.log(error)[:10],1)














