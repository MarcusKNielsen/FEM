import numpy as np
def xy(x0, y0, L1, L2, noelms1, noelms2):
    lx = L1 / noelms1
    ly = L2 / noelms2

    VX = np.repeat([x0 + i * lx for i in range(noelms1 + 1)], noelms2 + 1)
    VY = [y0 + j * ly for j in range(noelms2,-1,-1)] * (noelms1 + 1)

    return VX, VY




#%%

def conelmtab(noelms1,noelms2):
    EToV = []
    for j in range(noelms1):
        for i in range(noelms2):
            EToV.append([i+noelms2+1+noelms1*j,i+noelms1*j, i+noelms2+2+noelms1*j])
            EToV.append([i+1+noelms1*j,i+noelms2+2+noelms1*j,i+noelms1*j])

    return np.array(EToV)


#%%



from matplotlib import pyplot as plt

def basfun(n,VX,VY,EToV):
    idx1 = EToV[n,0]
    idx2 = EToV[n,1]    
    idx3 = EToV[n,2]

    x1 = VX[idx1]
    x2 = VX[idx2]
    x3 = VX[idx3]

    y1 = VY[idx1]
    y2 = VY[idx2]
    y3 = VY[idx3]

    delta = 1/2*(x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1)

    abc = np.zeros((3,3))

    abc[0,0] = x2*y3 - x3*y2
    abc[0,1] = y2 - y3
    abc[0,2] = x3 - x2

    abc[1,0] = x3*y1 - x1*y3
    abc[1,1] = y3 - y1
    abc[1,2] = x1 - x3

    abc[2,0] = x1*y2 - x2*y1
    abc[2,1] = y1 - y2
    abc[2,2] = x2 - x1


    return delta, abc


    # plt.plot(VX,VY,'o')
    # plt.plot([x1,x2,x3],[y1,y2,y3],'o')

    

#%%

import numpy as np
from scipy.sparse import csr_matrix

def assembly(VX, VY, EToV, lam1,lam2, qt):
    N = len(EToV[:,1])
    M = len(VX)
    # nnzmax = 6*M
    # ii = np.ones(nnzmax, dtype=int)
    # jj = np.ones(nnzmax, dtype=int)
    # ss = np.zeros(nnzmax)
    # b = np.zeros(M)
    # count = 0

    A = np.zeros((M,M))
    b = np.zeros(M)
    
    for n in range(N):
        delta, abc = basfun(n,VX,VY,EToV)
        print(abc)

        i = EToV[n,0]
        j = EToV[n,1]    
        k = EToV[n,2]
            
        triplet = [i,j,k]

        x1 = VX[i]
        x2 = VX[j]
        x3 = VX[k]

        y1 = VY[i]
        y2 = VY[j]
        y3 = VY[k]
        
        for r in range(3):
            
            qtilde = 1/3*(qt(x1,y1)+qt(x2,y2)+qt(x3,y3))
            b[triplet[r]] += (np.abs(delta)/3)*qtilde

            for s in range(3):
                
                k_rs = 1/(4*np.abs(delta))*(lam1*abc[r,1]*abc[s,1]+lam2*abc[r,2]*abc[s,2])
                # print(k_rs, i, j)
                A[triplet[r],triplet[s]] += k_rs
                # print(A)
    return A,b

#%%


#(x0, y0) = (0,0) 
#L1 = 1
#L2 = 1
#noelms1 = 1
#noelms2 = 1
#q = lambda x,y: 0
#lam1 = 1
#lam2 = 1

(x0, y0) = (-2.5, -4.8) 
L1 = 7.6
L2 = 5.9
noelms1 = 4
noelms2 = 3
q = lambda x,y: -6*x+2*y-2
lam1 = 1
lam2 = 2

VX,VY = xy(x0,y0,L1,L2,noelms1,noelms2)
EToV = conelmtab(noelms1,noelms2)

A,b = assembly(VX, VY, EToV, lam1,lam2, q)









