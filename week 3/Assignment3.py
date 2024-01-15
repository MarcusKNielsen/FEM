import scipy.sparse as sp
import numpy as np

def conelmtab(x):

    ...

    return VX, EToV

def assembly(VX, EToV, D, qt, ...):
    N = len(EToV[:,1])
    M = len(VX)

    nnzmax = 9*N
    ii = np.zeros(nnzmax, dtype=int)
    jj = np.zeros(nnzmax, dtype=int)
    ss = np.zeros(nnzmax)
    b = np.zeros(M)
    count = 0
    
    for nn in range(N):
        # Save element information
        i = EToV[nn,0]
        j = EToV[nn,1]    

        x1 = VX[i]
        x2 = VX[j]

        points = [i,j]


        for rr in range(2):
            # Compute qtilde
            qtilde = 1/2*(qt(x1)+qt(x2))

            # compute qn ???
            
            b[points[rr]] += qtilde

    #         for s in range(3):
                
    #             k_rs = 1/(4*np.abs(delta))*(lam1*abc[r,1]*abc[s,1]+lam2*abc[r,2]*abc[s,2])
                
    #             ii[count] = triplet[r]
    #             jj[count] = triplet[s]
    #             ss[count] = k_rs
    #             count += 1


    # A = sp.csr_matrix((ss[:count], (ii[:count], jj[:count])), shape=(M, M))
    # return A,C,b

def dirbc(bnodes, f, A, b):
    M = len(b)
    for n,i in enumerate(bnodes):

        A[i,i] = 1
        b[i] = f[n]

        for j in range(M):
            if j != i and A[i,j] != 0:
                A[i,j] = 0
                if j not in bnodes:
                    b[j] -= A[j,i]*f[n]
                    A[j,i] = 0
    
    return A,b
