import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

def refine_marked(EToVcoarse, xcoarse, idxMarked):
    N = len(EToVcoarse[:,0]) + 1
    Old2New = np.zeros((len(idxMarked),3),dtype=int)

    EToVfine = EToVcoarse.copy()
    xfine = xcoarse.copy()

    for i,idx in enumerate(idxMarked):

        xi   = xfine[EToVfine[idx][0]]
        xip1 = xfine[EToVfine[idx][1]]
        xih  =  xi + (xip1 - xi)/2

        xfine = np.hstack((xfine,[xih]))

        M = EToVfine[idx][1]
        EToVfine[idx][1] = N 
        EToVfine = np.vstack((EToVfine,[N,M]))

        Old2New[i] = [EToVfine[idx][0], N, M]

        N += 1

    return EToVfine, xfine, Old2New

K = lambda h: np.array([[1/h + h/3, -1/h + h/6], [-1/h + h/6, 1/h + h/3]])

def GlobalAssembly(x,c,d,func):
    M = len(x)
    nnzmax = 4 * M
    ii = np.ones(nnzmax, dtype=int)
    jj = np.ones(nnzmax, dtype=int)
    ss = np.zeros(nnzmax)
    b = np.zeros(M)
    count = 0

    for i in range(M - 1):
        h = x[i+1] - x[i]
 
        fval = func(x[i])

        if i > 0:
            b[i-1] += h*fval/6        
        b[i] += 2*h*fval/3
        b[i+1] += h*fval/6

        Ki = K(h)

        ii[count:count + 4] = [i, i, i + 1, i + 1]
        jj[count:count + 4] = [i, i + 1, i + 1, i]
        ss[count:count + 4] = [
        Ki[0, 0],
        Ki[0, 1],
        Ki[1, 1],
        Ki[1, 0]
        ]
        count += 4
    
    A = csr_matrix((ss[:count], (ii[:count], jj[:count])), shape=(M, M))
    b = -b
    
    # Boundary conditions
    b[0] = c
    b[1] -= A[0,1]*c

    A[0,0] = 1
    A[0,1] = 0
    A[1,0] = 0
    
    b[M-1] = d
    b[M-2] -= A[M-1,M-2]*d

    A[M-1,M-1] = 1
    A[M-1,M-2] = 0
    A[M-2,M-1] = 0

    

    return A, b

def BVP1D(L, x, c, d,func, plot=True):
    
    if type(x) == int:
        x = np.linspace(0, L, x)

    A,b = GlobalAssembly(x,c,d,func)

    u = sparse.linalg.spsolve(A, b)
    
    if plot:
        plt.plot(x, u, '.',label="FEM solution")
        plt.show()

    return u



def prep_grid(L,c,d,VXc,EToVc,func):
    idxMarked = np.arange(len(VXc)-1)

    # sorter her efter value
    sort_indices_c = np.argsort(VXc)
    uc = BVP1D(L, VXc[sort_indices_c], c, d,func, plot=False)



    EToVf, VXf,Old2New = refine_marked(EToVc,VXc,idxMarked)

    sort_indices_f = np.argsort(VXf)
    uf = BVP1D(L, VXf[sort_indices_f], c, d,func, plot=False)
    
    # sorter her efter index
    uc = uc[np.argsort(sort_indices_c)]
    uf = uf[np.argsort(sort_indices_f)]

    return uc,uf, VXc, VXf, EToVc, EToVf, Old2New



def compute_error_decrease(uc,uf,VXc,VXf,Old2New):
    
    N = len(VXc)-1
    err = np.zeros(N)

    for n, triple in enumerate(Old2New):

        i = triple[0]
        j = triple[2]
        k = triple[1]

        xi   = VXc[i]
        xj = VXc[j]
        xk  =  xi + (xj - xi)/2

        uci  = uc[i]
        ucj =  uc[j]

        ufi = uf[i]
        ufk = uf[k]
        ufj = uf[j]

        a  = (ucj - uci) / (xj - xi)
        a1 = (ufk - ufi) / (xk - xi)
        a2 = (ufj - ufk) / (xj - xk)

        b = uci - a * xi
        b1 = ufi - a1 * xi
        b2 = ufj - a2 * xj


        int1 = ((a-a1)**2/3) * (xk**3 - xi**3) + (b-b1)**2 * (xk - xi) + (a-a1)*(b-b1) * (xk**2 - xi**2)
        int2 = ((a-a2)**2/3) * (xj**3 - xk**3) + (b-b2)**2 * (xj - xk) + (a-a2)*(b-b2) * (xj**2 - xk**2)
        
        err[n] = np.sqrt(int1 + int2)

    return err

def DriverAMR17(L,c,d,VXc,func,tol, maxit):
    it = 0
    idxMarked = [1]
    
    idxs = np.arange(len(VXc))
    EToVc = np.vstack((idxs[:-1], idxs[1:])).T
    
    while len(idxMarked)>0 and it < maxit:
        
        uc,uf, VXc, VXf, EToVc, EToVf, Old2New = prep_grid(L,c,d,VXc,EToVc,f)

        err = compute_error_decrease(uc,uf,VXc,VXf,Old2New)
        
        # Change strategy here
        
        # Old
        #idxMarked = np.where(err > tol)[0]
        
        # New
        m = np.max(err)
        if m > tol:
            idxMarked = np.where(err > 0.91*m)[0]
        else:
            idxMarked = []
    
        EToVc, VXc, Old2New = refine_marked(EToVc,VXc,idxMarked)

        it +=1

    VXc = np.sort(VXc)
    uc = BVP1D(L, VXc, c, d,func, plot=False)
    
    
    return VXc, uc, it,Old2New



#%%
u = lambda x: np.exp(-800*(x-0.4)**2) + 0.25 * np.exp(-40*(x-0.8)**2)
L = 1
c = u(0)
d = u(1)
VXc = np.linspace(0,1,4)
idxs = np.arange(len(VXc))
EToVc = np.vstack((idxs[:-1], idxs[1:])).T

#func = lambda x: 1*x
f = lambda x: (np.exp(-800*(x-0.4)**2)*((-1600*(x-0.4))**2-1601) + 0.25*np.exp(-40*(x-0.8)**2)*((-80*(x-0.8))**2-81))

u_init = BVP1D(L, VXc, c, d,f, plot=True)

#uc,uf, VXc, VXf, EToVc, EToVf, Old2New = prep_grid(L,c,d,VXc,EToVc,f)

#uc,uf, VXc, VXf, EToVc, EToVf, Old2New = prep_grid(L,c,d,VXf,EToVf,f)


#%%

#plt.plot(VXf,uf,".")
#plt.plot(VXc,uc,".")

#%%

VXc, uc, it,Old2New = DriverAMR17(L,c,d,VXc,f,tol=10**(-4), maxit=100)

print(f"Number of points: {len(VXc)}")

plt.plot(VXc,uc,".")





