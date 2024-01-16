#%%
import numpy as np

p = 3
x0 = 0
L = 1
x = np.linspace(x0,x0+L,p+1)
print(np.polynomial.legendre.legvander(x,3))

# %%
