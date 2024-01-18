from SEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 10
p = 3
x0 = 0
L = np.pi
VX,VX_fine, C = construct_c(n, p, x0, L)
D = 1
t0 = 0
f = [0,0]

VX_fine = np.array(VX_fine)

def u_true(x,t):
    return np.sin(x)*np.exp(-np.pi**2 * t) + 0.1*x*(np.pi - x)*np.exp(x)*t

def qt(t,x):
    k = 0.1
    term1 = (1-np.pi**2) * np.sin(x)*np.exp(- np.pi**2 * t)
    term2 = k*x*( (np.pi - x) - (np.pi-x-1)*t ) * np.exp(x)
    term3 = k*(2 - 2*(np.pi-x) + x)*np.exp(x)*t
    return term1 + term2 + term3

un = u_true(VX_fine,0)

theta = 1.0
dt = 0.001

unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, un, theta, p)

# plt.figure()
# plt.plot(VX_fine,un,label="u0")
# plt.plot(VX_fine,unext,label="unext")
# plt.plot(VX_fine,u_true(VX_fine,dt), label="True")
# plt.legend()
# plt.show()

#%%
fig = plt.figure('convergence tests for order p', figsize=(12,6))
axgrid = fig.add_gridspec(2,4)

unext = u_true(VX_fine,0)
n = 201

t = 0
T = 0.5
dt = T/n
k = 0
# ax0 = fig.add_subplot(axgrid[0:2,:])

# while t < T:
#     if k % 40 == 0:
#         ax0.plot(VX_fine,unext,label=f"t = {np.round(t,2)}")
#         ax0.plot(VX_fine,u_true(VX_fine,t),"--",label=f"t = {np.round(t,2)}")
#     unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)
#     t += dt
#     k += 1
# ax0.set_title(f"Solution at different times for p = {p}")
# ax0.legend()


#%% Time Convergence test
# '
# N = list(range(2,50))
# error = np.zeros(len(N))


# for i,n in enumerate(N):

#     unext = un
    
#     t = 0
#     T = 0.5
#     dt = T/n
#     while t < T:
#         unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
#         t += dt
    
#     error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

# h = T/np.array(N)

# plt.figure()
# plt.plot(np.log(h),np.log(error),label="error")
# plt.ylabel("Error")
# plt.xlabel("Step Size: dt")
# plt.show()

# a,b = np.polyfit(np.log(h)[-20:],np.log(error)[-20:],1)'

#%%

N = list(range(2,50))
dt = 10**(-8)
p_values = [1,2,3,4,5] 
ax1 = fig.add_subplot(axgrid[:,:2])
for p in p_values:
    error = np.zeros(len(N))
    for i,n in enumerate(N):
    
        VX,VX_fine, C = construct_c(n, p, x0, L)
        
        unext = np.sin(VX_fine)
        
        t = 0
        T = dt
        while t < T:
            unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
            t += dt
    
        error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

    DGF = np.array(N)*p+1
    a,b = np.polyfit(np.log(DGF)[:],np.log(error)[:],1)
    ax1.loglog(DGF,error,label=f"p = {p}, a = {a:.2f}")

ax1.set_title("Convergence Test at t = 1e-8 and timestep = 1e-8")
ax1.set_ylabel("Error")
ax1.set_xlabel("degrees of freedom")
ax1.legend()


N = list(range(2,50))
dt = 0.01
p_values = [1,2,3,4,5] 

ax2 = fig.add_subplot(axgrid[:,2:])
for p in p_values:
    error = np.zeros(len(N))
    for i,n in enumerate(N):
    
        VX,VX_fine, C = construct_c(n, p, x0, L)
        
        unext = np.sin(VX_fine)
        
        t = 0
        T = 0.5
        while t < T:
            unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
            t += dt
    
        error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 

    DGF = np.array(N)*p+1
    a,b = np.polyfit(np.log(DGF)[:],np.log(error)[:],1)
    ax2.loglog(DGF,error,label=f"p = {p}, a = {a:.2f}")

ax2.set_title("Convergence Test at t = 0.5 and timestep = 0.01")
ax2.set_ylabel("Error")
ax2.set_xlabel("degrees of freedom")
ax2.legend()

plt.tight_layout()

plt.show()











# %%
