import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def xy(x0, y0, L1, L2, noelms1, noelms2):
    lx = L1 / noelms1
    ly = L2 / noelms2

    VX = np.repeat([x0 + i * lx for i in range(noelms1 + 1)], noelms2 + 1)
    VY = [y0 + j * ly for j in range(noelms2,-1,-1)] * (noelms1 + 1)

    return np.array(VX), np.array(VY)


def conelmtab(noelms1,noelms2):
    EToV = []
    for j in range(noelms1):
        for i in range(noelms2):
            EToV.append([i+noelms2+1+(noelms2+1)*j,i+(noelms2+1)*j, i+noelms2+2+(noelms2+1)*j])
            EToV.append([i+1+(noelms2+1)*j,i+noelms2+2+(noelms2+1)*j,i+(noelms2+1)*j])

    return np.array(EToV)


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



def assembly(VX, VY, EToV, lam1,lam2, qt):
    N = len(EToV[:,1])
    M = len(VX)

    nnzmax = 9*N
    ii = np.zeros(nnzmax, dtype=int)
    jj = np.zeros(nnzmax, dtype=int)
    ss = np.zeros(nnzmax)
    b = np.zeros(M)
    count = 0
    
    for n in range(N):
        delta, abc = basfun(n,VX,VY,EToV)

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
                
                ii[count] = triplet[r]
                jj[count] = triplet[s]
                ss[count] = k_rs
                count += 1


    A = sp.csr_matrix((ss[:count], (ii[:count], jj[:count])), shape=(M, M))
    return A,b



def dirbc(bnodes, f, A, b):
    M = len(b)
    for n,i in enumerate(bnodes):

        if f[n] == None:
            continue

        A[i,i] = 1
        b[i] = f[n]

        for j in range(M):
            if j != i and A[i,j] != 0:
                A[i,j] = 0
                if j not in bnodes:
                    b[j] -= A[j,i]*f[n]
                    A[j,i] = 0
    
    return A,b


def find_bnodes(noelms1,noelms2):
    bnodes = np.arange(noelms2+1)
    for j in range(1,noelms1):
        bnodes = np.append(bnodes, j*(noelms2+1))
        bnodes = np.append(bnodes, j*(noelms2+1) + noelms2)

    bnodes = np.append(bnodes, np.arange((noelms2+1)*noelms1, (noelms2+1)*(noelms1+1)))

    return bnodes


def ConstructBeds(VX,VY,EToV,x0,y0,L1,L2):

    beds = []

    for n in range(len(EToV[:,1])):
        for k in range(3):
            (i,j) = (EToV[n,(k,((k+1) % 3))])

            mid = np.array([(VX[i]+VX[j])/2, (VY[i]+VY[j])/2])

            if mid[0] <= x0 or mid[0]  >= x0 + L1 or mid[1] <= y0 or mid[1] >= y0 + L2:
                beds.append([n,k])

    return np.array(beds)


def neubc(VX,VY,EToV,beds,q,b):
    for n,r in beds:
        i, j = EToV[n,(r,((r+1) % 3))]

        xi = VX[i]
        xj = VX[j]
        yi = VY[i]
        yj = VY[j]


        qhat = q( (xi+xj)/2, (yi+yj)/2 )
        
        if qhat == None:
            continue

        qp = qhat/2*np.sqrt(((xj-xi)*(xj-xi) + (yj-yi)*(yj-yi)))

        b[i] -= qp
        b[j] -= qp
    
    return b


def BVP2D(x0,y0,L1,L2,noelms1,noelms2,qt,lam1,lam2,f,q=None):
    VX,VY = xy(x0,y0,L1,L2,noelms1,noelms2)
    EToV = conelmtab(noelms1,noelms2)
    bnodes = find_bnodes(noelms1,noelms2)

    A,b = assembly(VX, VY, EToV, lam1,lam2, qt)
    
    if q != None:
        beds = ConstructBeds(VX,VY,EToV,x0,y0,L1,L2)
        b = neubc(VX,VY,EToV,beds,q,b)
    
    farr = [f(VX[bnodes[i]], VY[bnodes[i]]) for i in range(len(bnodes))]
    A, b = dirbc(bnodes, farr, A, b)


    u = sp.linalg.spsolve(A,b)

    return u, VX, VY


def plot_heatmap(u,VX,VY,ufun=None):

    # Get the unique values in x and y
    unique_x = np.unique(VX)
    unique_y = np.unique(VY)

    # Create a meshgrid
    X, Y = np.meshgrid(unique_x, unique_y)

    # Reshape u to the shape of X and Y
    reshaped_u = np.flip(u.reshape(X.shape),axis=1)

    # Create the heatmap
    heatmap = plt.pcolormesh(X, Y, reshaped_u, shading='auto')

    # Add a color bar
    plt.colorbar(heatmap)  # This adds the color bar

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmap of u(x, y)')
    plt.show()

    if ufun != None:
        error = np.abs(u - ufun(VX,VY))
        error_reshaped = np.flip(error.reshape(X.shape),axis=1)
        heatmap = plt.pcolormesh(X, Y, error_reshaped, shading='auto',cmap='hot')

        plt.colorbar(heatmap)  # This adds the color bar

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Heatmap of error')
        plt.show()