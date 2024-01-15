import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np

def conelmtab(x0, L, noelms):
    VX = np.linspace(x0, x0+L, noelms+1)
    idxs = np.arange(noelms+1)
    EToV = np.vstack((idxs[:-1], idxs[1:])).T

    return VX, EToV

def assembly1D(VX, EToV, D, qt, t):
    N = len(EToV[:,1])
    M = N+1

    nnzmax = 4*N
    ii = np.zeros(nnzmax, dtype=int)
    jj = np.zeros(nnzmax, dtype=int)
    r = np.zeros(nnzmax)    # Corresponds to A
    s = np.zeros(nnzmax)    # Corresponds to C
    b = np.zeros(M)
    count = 0
    
    for nn in range(N):
        # Save element information
        i = EToV[nn,0]
        j = EToV[nn,1]

        xi = VX[i]
        xj = VX[j]

        h = xj - xi

        qtilde = h/4*(qt(t,xi)+qt(t,xj))

        b[[i,j]] += qtilde


        ii[count:count + 4] = [i, i, j, j]
        jj[count:count + 4] = [i, j, j, i]
        r[count:count + 4] = list(np.array([D/h,-D/h,D/h,-D/h]))
        s[count:count + 4] = list(np.array([h/3,-h/6,h/3,-h/6]))
        
        count += 4


    R = sp.csr_matrix((r[:count], (ii[:count], jj[:count])), shape=(M, M))
    S = sp.csr_matrix((s[:count], (ii[:count], jj[:count])), shape=(M, M))


    return R,S,b

def dirbc1D(f, R,S, b):
    d = np.zeros(len(b))

    # Boundary conditions
    # i = 0, j = 1
    temp = R[0,1]*f[0]
    
    b[0] = d[0] = 0

    b[1] -= temp
    d[1] += temp

    # Modify R and S
    R[0,0] = R[1,0] = R[0,1] = 0

    S[0,0] = 1
    S[1,0] = S[0,1] = 0
    
    # i = N-1, j = N-2
    temp = R[-2,-1]*f[-1]

    b[-1] = d[-1] = 0

    b[-2] -= temp
    d[-2] += temp

    R[-1,-1] = R[-1,-2] = R[-2,-1] = 0
    
    S[-1,-1] = 1
    S[-1,-2] = S[-2,-1] = 0

    return S,R,b,d

   
def RS_1D(R,S,dt,theta):
    S -= dt*(1-theta)*R
    R = S + dt*R 

    return R,S


# Step 4 Factor but we don't.

def construct_e(S,b,un,dt,theta):
    return S@un + dt*theta*b


# Step 7 + 8
def advance_b(d,EToV, qt,tnext):
    bnext = np.zeros(len(EToV[:,0])+1)

    for elm in EToV:
        # Save element information
        i = elm[0]
        j = elm[1]

        xi = VX[i]
        xj = VX[j]

        h = xj - xi

        qtilde = h/4*(qt(tnext,xi)+qt(tnext,xj))

        bnext[[i,j]] += qtilde

    bnext[0] = bnext[-1] = 0
    bnext -= d

# Step 9
def update_e(e,bnext,dt,theta):
    return e + dt*theta*bnext    
    

## Testcase:
n = 5
VX, EToV = conelmtab(0,n,n)
D = 1
qt = lambda t,x: 1
t = 0
f = [0,0]

R, S, b = assembly1D(VX, EToV, D, qt, t)
R,S, b, d = dirbc1D(f, R,S, b)

print('A:')
print(R.todense())

print(':C')
print(S.todense())

dt = 1
theta = 0

R,S = RS_1D(R,S,dt,theta)

print(R.todense())

un = np.ones(len(b))
e = construct_e(S,b,un,dt,theta)

b = advance_b(d,EToV, qt,1)

e = update_e(e,b,dt,theta)


# Step 10: Solve!
u = sp.linalg.spsolve(R,e)