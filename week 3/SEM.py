import numpy as np
from LGL_nodes import legendre_gauss_lobatto_nodes
from Assignment3 import conelmtab


p = 1

x0 = 0
L = 1
noelms = 2

VX, EToV = conelmtab(x0, L, noelms)
