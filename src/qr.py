from typing import List
import numpy as np
import threading

def qr_decomposititon_gs(mat: List[List[int]]) -> tuple[List[List[int]], List[List[int]]]:
    """Dekomposisi QR dengan algoritma Gram-Schmidt Orthogonalization
    Too slow, still. O(2mn²).
    
    Must try Schwarz-Rutishauser Algorithm O(mn²)"""
    mat = np.array(mat)
    length = len(mat)

    e = np.empty((length,length))
    a = mat.T
    for i in range(length):
        u = np.copy(a[i])
        for j in range(i):
            u = u - (a[i] @ e[j]) * e[j]
        e[i] = u/np.linalg.norm(u)

    R = np.zeros((length,length))
    for i in range(length):
        for j in range(i, length):
            R[i][j] = a[j]@e[i]
    Q = e.T

    return (Q,R)


def qr_gs_modsr(A):
    A = np.array(A)
    (m,n) = np.shape(A)

    Q = np.array(A)      
    R = np.zeros((n, n))

    for k in range(n):
        for i in range(k):
            R[i,k] = Q[:,i].T@(Q[:,k])
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]

        R[k,k] = np.linalg.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
    
    return -Q, -R
