# For your convenience:
# Paste the required functions from previous assignments here.
import numpy as np
def estimate_F_DLT(x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    '''
    # Your code here
    u1,v1=x1s[0,:], x1s[1,:]
    u2,v2=x2s[0,:], x2s[1,:]
    A = np.column_stack([
    u2*u1,
    u2*v1,
    u2,
    v2*u1,
    v2*v1,
    v2,
    u1,
    v1,
    np.ones_like(u1)
    ])
    U, S, Vt = np.linalg.svd(A)
    f=Vt[-1,:]
    F = f.reshape(3, 3)
    return F
def enforce_fundamental(F_approx):
    '''
    F_approx - Approximate Fundamental matrix (3x3)
    '''
    # Your code here
    U, S, Vt = np.linalg.svd(F_approx)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    return F

def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    # Your code here
    U, S, Vt = np.linalg.svd(E_approx)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    Sigma = np.diag([1, 1, 0])
    E = U @ Sigma @ Vt
    return E

def convert_E_to_F(E,K1,K2):
    '''
    A function that gives you a fundamental matrix from an essential matrix and the two calibration matrices
    E - Essential matrix (3x3)
    K1 - Calibration matrix for the first image (3x3)
    K2 - Calibration matrix for the second image (3x3)
    '''
    # Your code here
    F=np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F - Fundamental matrix (3x3)
    '''
    # Your code here
    l2=F @ x1s
    d2=np.abs(np.sum(l2 * x2s, axis=0)) / np.sqrt(l2[0,:]**2 + l2[1,:]**2)
    return d2

def extract_P_from_E(E):
     '''
    A function that extract the four P2 solutions given above
    E - Essential matrix (3x3)
    P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution) 
    '''
    # Your code here
     P=np.zeros((4, 3, 4))
     U, S, Vt = np.linalg.svd(E)
     if np.linalg.det(U) < 0:
          U[:, -1] *= -1
     if np.linalg.det(Vt) < 0:
          Vt[-1, :] *= -1
     W = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])
     t = U[:, 2]
     P[0, :, :] = np.hstack((U @ W @ Vt, t.reshape(3, 1)))
     P[1, :, :] = np.hstack((U @ W @ Vt, -t.reshape(3, 1)))
     P[2, :, :] = np.hstack((U @ W.T @ Vt, t.reshape(3, 1)))
     P[3, :, :] = np.hstack((U @ W.T @ Vt, -t.reshape(3, 1)))
     return P

def triangulate_3D_point_DLT(x1_n, x2_n, P1, P2):
   
    A = np.zeros((4, 4))
    A[0, :] = x1_n[0] * P1[2, :] - P1[0, :]
    A[1, :] = x1_n[1] * P1[2, :] - P1[1, :]
    A[2, :] = x2_n[0] * P2[2, :] - P2[0, :]
    A[3, :] = x2_n[1] * P2[2, :] - P2[1, :]
    
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :].reshape(4, 1)
    return X