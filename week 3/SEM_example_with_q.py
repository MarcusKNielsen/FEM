from SEM import *
import numpy as np
import matplotlib.pyplot as plt


## Testcase:
n = 50
p = 3
x0 = 0
L = np.pi
VX,VX_fine, C = construct_c(n, p, x0, L)
D = 1
t = 0
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

plt.figure()
plt.plot(VX_fine,un,label="u0")
plt.plot(VX_fine,unext,label="unext")
plt.plot(VX_fine,u_true(VX_fine,dt), label="True")
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

#%% 3d plot

unext = u_true(VX_fine,0)
N = 50
n = 100
t = 0
T = 0.5
dt = T/n
U = np.zeros([N*p + 1, n + 1])
U[:,0] = unext
k = 1
t_arr = [0]
while t < T:
    unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)
    U[:,k] = unext
    t += dt
    t_arr.append(t)
    k += 1
    
#%% 3d plot continued
from mpl_toolkits.mplot3d import Axes3D

# Getting the unique values and sorting them
x = VX_fine

# Creating the meshgrid
T, X = np.meshgrid(t_arr,x)

size = 12

# Create a figure and a 3D axis
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121, projection='3d')

# Surface plot
surf = ax.plot_surface(T, X, U, cmap='plasma')
ax.set_xlabel('Time: t',fontsize=size)
ax.set_ylabel('Space: x',fontsize=size)
ax.set_zlabel('Temperature: u(x,t)',fontsize=size)
ax.set_title("Numerical Solution (SEM)",fontsize=size+1)

# Second surface plot
U1 = u_true(X,T)
ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd subplot
surf2 = ax2.plot_surface(T, X, U1, cmap='plasma')
ax2.set_xlabel('Time: t',fontsize=size)
ax2.set_ylabel('Space: x',fontsize=size)
ax2.set_zlabel('Temperature: u(x,t)',fontsize=size)
ax2.set_title("Analytical Solution",fontsize=size+1)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=-0.3)
#plt.tight_layout()

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

#%%

N = list(range(1,20))
error = np.zeros(len(N))

dt = 0.0001

for i,n in enumerate(N):
    
    VX,VX_fine, C = construct_c(n, p, x0, L)
    
    unext = np.sin(VX_fine)
    
    t = 0
    T = dt
    while t < T:
        unext = oneit(VX, VX_fine, C, f, D, qt, t, dt, unext, theta, p)  
        t += dt
    
    error[i] = np.linalg.norm(unext - u_true(VX_fine,t),np.inf) 
    break
h = L/np.array(N)


a,b = np.polyfit(np.log(h)[:],np.log(error)[:],1)

plt.figure()
plt.plot(np.log(h),np.log(error),label="error")
plt.plot(np.log(h),(p+1)*np.log(h),label=r"$O(h^2)$")
plt.ylabel("log(Error)")
plt.xlabel("log(h)")
plt.legend()
plt.show()












