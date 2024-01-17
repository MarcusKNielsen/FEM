from FEM import *
import numpy as np
from matplotlib import pyplot as plt


## Testcase:
N  = 100
#p  = 3
L  = np.pi
D  = 1
f  = [0,0]
x0 = 0
theta = 1.0

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

unext = u_true(VX,0)
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