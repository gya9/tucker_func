import numpy as np
# import tensorly as tl
# from tensorly.decomposition import parafac, tucker
# from tensorly.tenalg import mode_dot, multi_mode_dot
# from tensorly import tucker_to_tensor
from scipy import linalg


def printT(X):
    """3階テンソル用のprint関数。
       i*j*k テンソルに対し、ij:を正面スライスとして描画する。"""
    if str(type(X)) != "<class 'numpy.ndarray'>":
        print("type error!",type(X),"is not allowed")
    else:
        XX = X.transpose(2,0,1)
        print(XX)


def printfac(X):
    """3階テンソル用のprint関数。
       各部屋の要素を確認できる。"""
    a,b,c = X.shape
    for k in range(c):
        for j in range(b):
            for i in range(a):
                print((i+1,j+1,k+1), "=", X[i][j][k])


def modep(X,U,mode):
    """モード積の関数。"""
    mode = int(mode)
    I,J,K = X.shape
    L,P = U.shape
    if mode == 1:
        if P != I:
            print("dimension error, mode=1")
            print("dim of tensor:",X.shape)
            print("dim of matrix:",U.shape)
        else:
            Y = np.zeros(L*J*K).reshape(L,J,K)

            for k in range(K):
                for j in range(J):
                    for i in range(L):
                        for l in range(I):
                            Y[i][j][k] += X[l][j][k] * U[i][l]
            return(Y)

    elif mode == 2:
        if P != J:
            print("dimension error, mode=2")
            print("dim of tensor:",X.shape)
            print("dim of matrix:",U.shape)
        else:
            Y = np.zeros(I*L*K).reshape(I,L,K)

            for k in range(K):
                for j in range(L):
                    for i in range(I):
                        for l in range(J):
                            Y[i][j][k] += X[i][l][k] * U[j][l]
            return(Y)

    elif mode == 3:
        if P != K:
            print("!dimension error, mode=3")
            print("dim of tensor:",X.shape)
            print("dim of matrix:",U.shape)
        else:
            Y = np.zeros(I*J*L).reshape(I,J,L)

            for k in range(L):
                for j in range(J):
                    for i in range(I):
                        for l in range(K):
                            Y[i][j][k] += X[i][j][l] * U[k][l]
            return(Y)

    else:
        print("mode > 3 is not defined yet")


def core2tensor(g,A1,A2,A3):
    Y = modep(g,A1,1)
    Y = modep(Y,A2,2)
    Y = modep(Y,A3,3)
    return(Y)


def unfold(X,mode):
    mode = int(mode)
    I,J,K = X.shape

    if mode == 1:
        V = X.transpose(0,2,1).reshape(I,J*K)
        return(V)

    elif mode == 2:
        V = X.transpose(1,2,0).reshape(J,I*K)
        return(V)

    elif mode == 3:
        V = X.transpose(2,1,0).reshape(K,I*J)
        return(V)

    else:
        print("mode > 3 is not defined yet")


def unfold_old(X,mode):
    mode = int(mode)
    I,J,K = X.shape

    if mode == 1:
        V = []
        for k in range(K):
            for j in range(J):
                    Vi = np.c_[np.array(X[:,j,k])]
                    if V == []:
                        V = Vi
                    else:
                        V = np.concatenate((V, Vi), axis=1)
        return(V)

    if mode == 2:
        V = []
        for k in range(K):
            for i in range(I):
                    Vi = np.c_[np.array(X[i,:,k])]
                    if V == []:
                        V = Vi
                    else:
                        V = np.concatenate((V, Vi), axis=1)
        return(V)

    if mode == 3:
        V = []
        for j in range(J):
            for i in range(I):
                    Vi = np.c_[np.array(X[i,j,:])]
                    if V == []:
                        V = Vi
                    else:
                        V = np.concatenate((V, Vi), axis=1)
        return(V)

    else:
        print("mode > 3 is not defined yet")


def tucker(X,R1,R2,R3):
    I,J,K = X.shape

    X1 = unfold(X,1)
    X2 = unfold(X,2)
    X3 = unfold(X,3)

    A1 = linalg.svd(X1, full_matrices=False)[0]
    A2 = linalg.svd(X2, full_matrices=False)[0]
    A3 = linalg.svd(X3, full_matrices=False)[0]

    A1 = A1[:,:R1]
    A2 = A2[:,:R2]
    A3 = A3[:,:R3]

    g = modep(X,A1.T,1)
    g = modep(g,A2.T,2)
    g = modep(g,A3.T,3)

    return(g,A1,A2,A3)


def tucker2d(X,R1,R2):
    I,J,K = X.shape

    X1 = unfold(X,1)
    X2 = unfold(X,2)
    # X3 = unfold(X,3)

    A1 = linalg.svd(X1)[0]
    A2 = linalg.svd(X2)[0]
    # A3 = linalg.svd(X3)[0]

    A1 = A1[:,:R1]
    A2 = A2[:,:R2]
    # A3 = A3[:,:R3]

    g = modep(X,A1.T,1)
    g = modep(g,A2.T,2)
    # g = modep(g,A3.T,3)

    return(g,A1,A2)


def hooi(X,R1,R2,R3,itr):
    itr = int(itr)

    X0,A1,A2,A3 = tucker(X,R1,R2,R3)

    Y = modep(X0,A1,1)
    Y = modep(Y,A2,2)
    Y = modep(Y,A3,3)

    print(linalg.norm(X-Y) / linalg.norm(X))

    for i in range(itr):
        y = modep(X,A2.T,2)
        y = modep(y,A3.T,3)
        Y = unfold(y,1)
        A1,foo,bar = linalg.svd(Y)
        A1 = A1[:,:R1]


        y = modep(X,A1.T,1)
        y = modep(y,A3.T,3)
        Y = unfold(y,2)
        A2,foo,bar = linalg.svd(Y)
        A2 = A2[:,:R2]


        y = modep(X,A1.T,1)
        y = modep(y,A2.T,2)
        Y = unfold(y,3)
        A3,foo,bar = linalg.svd(Y)
        A3 = A3[:,:R3]


        g = modep(X,A1.T,1)
        g = modep(g,A2.T,2)
        g = modep(g,A3.T,3)

        Y = modep(g,A1,1)
        Y = modep(Y,A2,2)
        Y = modep(Y,A3,3)

        print(linalg.norm(X-Y) / linalg.norm(X))

