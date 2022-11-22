import cmath
import numpy as np
import sympy as sp
from typing import List
from scipy.linalg import hessenberg

import multiprocessing
import src.qr as qr
import src.utility as util
import os
import datetime

# @squareOnly()
def getEigenValues(A:np.ndarray, iteration=50, mode='gs', h_opt=False, ignore_complex:bool = True) -> List[float]:
    """ 
    deprecated, too slow
    =======

    Menghasilkan list berupa eigen values yang sudah terurut secara descending.
    Akar-akar imajiner akan diabaikan."""
    if(h_opt):
        H, Q = hessenberg(A, calc_q=True)
    else:
        H = A
    Ak = np.copy(H)
    n = Ak.shape[0]
    QQ = np.eye(n)

    if(mode == 'gs'):
        qr_func = qr.qr_decomposititon_gs
    elif(mode == 'sr'):
        qr_func = qr.qr_gs_modsr
    elif(mode == 'np'):
        qr_func = np.linalg.qr
    elif(mode == 'hh'):
        qr_func = qr.qr_hh

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

    if ignore_complex:
        for i in range(len(es)-1,-1,-1):
            if np.iscomplex(es[i]):
                es[i] = np.real(es[i])

    return es

def rot_complex(val):

    v_rr = val.real * cmath.cos(cmath.pi / 2) + val.imag * cmath.sin(cmath.pi / 2)
    v_ri = val.real * cmath.sin(cmath.pi / 2) + val.imag * cmath.cos(cmath.pi / 2)

    return v_rr + 1.j * v_ri

def z_process(A, e, z, records):
    Av = np.array(A, dtype=complex)
    m, n = np.shape(Av)
    eigenV = np.eye(n, dtype=complex)

    for j in range(m):
        Av[j,j] -= e

    for i in range(m-1):
        alpha = Av[i,i]
        if alpha not in [0, 1]:
            for j in range(m):
                Av[i,j] = Av[i,j] / alpha
        if alpha == 0:
            Av[[i, i+1]] = Av[[i+1, i]]

        for k in range(m):
            if i != k:
                theta = Av[k,i]
                for j in range(i,m):
                    Av[k,j] = Av[k,j] - Av[i,j] * theta
    eigenV[:,z] = Av[:,m-1]
    eigenV[m-1][z] = 1.0
    nV = util.getEuclidDistance(eigenV[:,z])
    eigenV[:,z] = np.array([v / nV for v in eigenV[:,z]], dtype=complex)
    eigenV[:,z] = [rot_complex(v) for v in eigenV[:,z]] if e.imag != 0 else eigenV[:,z]
    records.append((z, eigenV[:,z]))

def getEigenVectors(A: np.ndarray, eigenValues:List[float], ignore_complex:bool = True) -> np.ndarray:
    """Menghasilkan basis ruang eigen dalam bentuk matrix dan sudah terurut menurut eigen value-nya."""
    if(multiprocessing.get_start_method(allow_none=True) == None):
        multiprocessing.set_start_method('spawn')
    m, n = np.shape(A)
    processes = []
    eigenV = np.identity(n, dtype=complex)
    now = datetime.datetime.now()
    process_count = os.cpu_count()

    with multiprocessing.Manager() as manager:

        records = manager.list()

        for (e,z) in zip(eigenValues, range(m)):
            process = multiprocessing.Process(target = z_process, args=(A, e, z, records))
            processes.append(process)
            process.start()
            z = z + 1
            if(len(processes) == process_count):
                process = processes.pop(0)
                process.join()
                # print(processes)

        for process in processes:
            process.join()

        for record in records:
            z, x = record
            eigenV[:, z] = x
        
        if(ignore_complex):
            eigenV = eigenV.real.astype(float)

        return eigenValues, eigenV