#%%
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.Symbol('x')
t = sym.Symbol('t')

u = (sym.exp(-800*(x-0.4)**2)+0.25*sym.exp(-40*(x-0.8)**2))*sym.exp(-t/2)
print(u)

u_x = sym.diff(u, x)
u_xx = sym.diff(u_x, x)
u_t = sym.diff(u, t)
D = 1

qt = u_t - D*u_xx
print(qt)

x_vals = np.linspace(0, 2, 100)
for T in range(0, 5):
    f_vals = [u.subs([(x, val), (t, T)]) for val in x_vals]
    plt.plot(x_vals, f_vals)

plt.xlabel('x')
plt.ylabel('f(x, 1)')
plt.title('Plot of f(x, 1)')
plt.show()

# %%
