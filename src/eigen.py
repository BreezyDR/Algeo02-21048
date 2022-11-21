import cmath
import numpy as np
from typing import List
from scipy.linalg import hessenberg
from matrix import Matrix

# @squareOnly()
def getEigenValues(A:np.ndarray, iteration=50, mode='gs') -> List[int]:
    """ 
    deprecated, too slow
    =======

    Menghasilkan list berupa eigen values yang sudah terurut secara descending.
    Akar-akar imajiner akan diabaikan."""
    H, Q = hessenberg(A, calc_q=True)
    Ak = np.copy(H)
    n = Ak.shape[0]
    QQ = np.eye(n)

    if(mode == 'gs'):
        qr_func = Matrix.qr_decomposititon_gs
    elif(mode == 'sr'):
        qr_func = Matrix.qr_gs_modsr
    elif(mode == 'np'):
        qr_func = np.linalg.qr

    for k in range(iteration):
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        Q, R = qr_func(np.subtract(Ak, smult))
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q
    # print(tabulate(Ak))

    n = np.shape(Ak)[1]                    
    if n >= 2:                            
        es = np.zeros(n, dtype=complex)   
        i = 0
        while i < n - 1:
            mu = (Ak[i][i] + Ak[i+1][i+1]) / 2
            det = Ak[i][i] * Ak[i+1][i+1] - Ak[i][i+1] * Ak[i+1][i]
            dt_root = cmath.sqrt(mu**2 - det)
            e1 = mu + dt_root; e2 = mu - dt_root

            if np.iscomplex(e1):
               if e1.real == e2.real:
                  e2 = np.conj(e2)
               es[i] = e1; es[i+1] = e2
               if es[i] < es[i+1]:
                  es[[i]] = es[[i+1]]
               i += 1
            i += 1
    Ak = np.diag(Ak)                         
    es = [Ak[i] if i in np.argwhere(es == 0)[:,0] 
            else es[i] for i in range(n)]

    return es


# def getEigenVectors() -> List[List[int]]:
#     """Menghasilkan basis ruang eigen dalam bentuk matrix dan sudah terurut menurut eigen value-nya."""
#     eigenValues = self.getEigenValues()
#     eigenVectors = []

#     A = sp.Matrix(self.buffer)
#     lamda = sp.Symbol("lamda", real=True)
#     larr = sp.eye(len(self.buffer))*lamda
#     B = larr-A

#     temprow = []
#     zeroMat = sp.zeros(B.rank(),1)
#     for e in eigenValues:
#         temp = B.copy().subs(lamda,e)
#         sol, params = temp.gauss_jordan_solve(zeroMat)
#         for param in params:
#             taus = {tau:0 for tau in params}
#             taus.update({param: 1})
#             temprow = [tau[0] for tau in sol.xreplace(taus).tolist()]
#             eigenVectors.append(temprow)

#     eigenVectors = sp.Matrix(eigenVectors).T.tolist()
#     return eigenVectors


# @staticmethod
# def qr_decomposititon_gs(mat: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
#     """Dekomposisi QR dengan algoritma Gram-Schmidt Orthogonalization
#     Too slow, still. O(2mn²).
    
#     Must try Schwarz-Rutishauser Algorithm O(mn²)"""
#     mat = np.array(mat)
#     length = len(mat)

#     e = np.empty((length,length))
#     a = mat.T
#     for i in range(length):
#         u = np.copy(a[i])
#         for j in range(i):
#             u -= (a[i] @ e[j]) * e[j]
#         e[i] = u/np.linalg.norm(u)

#     R = np.zeros((length,length))
#     for i in range(length):
#         for j in range(i, length):
#             R[i][j] = a[j]@e[i]
#     Q = e.T

#     return (Q,R)


# @staticmethod
# def qr_gs_modsr(A):
#     A = np.array(A)
#     (m,n) = np.shape(A)

#     Q = np.array(A)      
#     R = np.zeros((n, n))

#     for k in range(n):
#         for i in range(k):
#             R[i,k] = Q[:,i].T@(Q[:,k])
#             Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]

#         R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
    
#     return -Q, -R
