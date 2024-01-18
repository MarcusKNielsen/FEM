from FEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 100
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

def u_true(x,t):
    return np.sin(x)*np.exp(-D*t)


#%% Time Convergence test


H = np.logspace(-5, -1,10)
error = np.zeros(len(H))

for i,dt in enumerate(H):

    unext = un
    
    t = 0
    T = 0.1
    while t < T:
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)    
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

plt.figure()
plt.plot(np.log10(H),np.log10(error),"-o",label=r"$\Vert u - u_h \Vert_\infty$")
plt.plot(np.log10(H),np.log10(H), label=r"O(dt)")
plt.ylabel(r"Error: $\varepsilon$")
plt.xlabel("Step Size: dt")
plt.title("log-log plot of error vs time step (FEM)")
plt.legend()
plt.show()

a,b = np.polyfit(np.log(H),np.log(error),1)

#%%

N = list(range(2,20))
error_s = np.zeros(len(N))

dt = 0.01

for i,n in enumerate(N):
    
    VX, EToV = conelmtab(0,np.pi,n)
    
    unext = np.sin(VX)
    
    t = 0
    T = dt
    while t < T:
        unext = oneit(VX, EToV, f, D, qt, t,dt,unext,theta)    
        t += dt
    
    error_s[i] = np.linalg.norm(unext - u_true(VX,t),np.inf) 

h = L/np.array(N)

plt.figure()
plt.plot(np.log10(h),np.log10(error_s),"-o",label=r"$\Vert u - u_h \Vert_\infty$")
plt.plot(np.log10(h),2*np.log10(h)-2, label=r"O(h²)")
plt.ylabel(r"Error: $\varepsilon$")
plt.xlabel("Element lenght: h")
plt.title("log-log plot of error vs Element length (FEM)")
plt.legend()
plt.show()

a,b = np.polyfit(np.log(h),np.log(error_s),1)

#%%


# Your data (assuming H, h, error, and error_s are defined)
# H, error = ...
# h, error_s = ...

plt.figure(figsize=(8, 4))

size = 12

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(np.log10(H), np.log10(error), "-o", label=r"$\Vert u - u_h \Vert_\infty$")
plt.plot(np.log10(H), np.log10(H), label=r"O(dt)")
plt.ylabel(r"Error: $\varepsilon$",fontsize=size)
plt.xlabel("Step Size: dt",fontsize=size)
plt.title("log-log plot: error vs time step (FEM)", fontsize=size+1)
plt.legend()

# Second subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(np.log10(h), np.log10(error_s), "-o", label=r"$\Vert u - u_h \Vert_\infty$")
plt.plot(np.log10(h), 2 * np.log10(h) - 2, label=r"O(h²)")
plt.ylabel(r"Error: $\varepsilon$",fontsize=size)
plt.xlabel("Element length: h",fontsize=size)
plt.title("log-log plot: error vs element length (FEM)", fontsize=size+1)
plt.legend()

# Show the combined figure with subplots
plt.tight_layout()
plt.show()















