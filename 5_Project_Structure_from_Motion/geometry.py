import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import least_squares
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

def compute_h_errors(H, x1s, x2s):
    """
    compute error distances given homography H
    impout:
        H: (3, 3) 
        x1s, x2s: (3, N) 
    output:
        d2: (N,) 
    """
    # 1.  x2_pred = H * x1
    x2_pred = H @ x1s
    
    # 2.  (x/z, y/z)
    x2_pred = x2_pred / (x2_pred[2, :] + 1e-9)
    x2_target = x2s / (x2s[2, :] + 1e-9)
    
    # 3. ||x2_target - x2_pred||
    diff = x2_target[:2, :] - x2_pred[:2, :]
    d2 = np.sqrt(np.sum(diff**2, axis=0))
    
    return d2

def model_selection(pts1, pts2, K):
    """
    input:
        pts1, pts2: (N, 2) pixel coordinates of matching points
        
    output:
        best_model: string 'H' or 'F'
        best_M: coresponding model matrix
        best_mask: inlier mask for the best model
    """
    N = pts1.shape[0]
    # transform to homogeneous coordinates
    x1s = np.vstack((pts1.T, np.ones(N)))
    x2s = np.vstack((pts2.T, np.ones(N)))

    # GRIC parameters
    lambda1 = 2.0  
    lambda2 = 4.0 
    limit = 4.0**2 

    # ============================
    # option A: matrix H
    # ============================
    H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    
    if H is None:
        GRIC_H = np.inf 
    else:
        d_h = compute_h_errors(H, x1s, x2s)
        
        # Residual Sum of Squares
        #  sum( min(d^2, limit) )
        res_sum_H = np.sum(np.minimum(d_h**2, limit))
        
        # 3. calculate GRIC score
        GRIC_H = res_sum_H + (lambda1 * 2 * N) + (lambda2 * 8)

    # ============================
    # option B: matrix F
    # ============================
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    
    if F is None:
        GRIC_F = np.inf
    else:
        
        d_f = compute_epipolar_errors(F, x1s, x2s)
        
        res_sum_F = np.sum(np.minimum(d_f**2, limit))
        
        GRIC_F = res_sum_F + (lambda1 * 3 * N) + (lambda2 * 7)

    # ============================
    # Compare GRIC scores
    # ============================
    print(f"GRIC Scores -> H: {GRIC_H:.2f} vs F: {GRIC_F:.2f}")
    
    if GRIC_H < GRIC_F:
        print(">>> result: Planar ->  H")
        return 'H', H, mask_H
    else:
        print(">>>  result: 3D ->  F")
        return 'F', F, mask_F
def estimate_E_robust(x1, x2, eps, seed=None):
    """
    RANSAC estimate of essential matrix using normalized correspondences x1 and x2 and a normalized threshold.
    Note: Make sure to normalize things before using it in this function!
    -------------------------------------------
    x1: Normalized keypoints in image 1 - 3xN np.array or 2xN np.array, as you desire 
    x2: Normalized keypoints in image 2 - 3xN np.array or 2xN np.array, as you desire 
    eps: Normalized inlier threshold - float

    Returns:
    E: 3x3 essential matrix
    inliers: The inlier points
    errs: The epipolar errors
    iters: How many iterations it took
    """
    # TIPS: 
    # * You can use the already created functions, enforce_essential, estimate_F_DLT, and compute_epipolar_errors
    # * Normalizing the pixel threshold can be done by e.g. eps = threshold_px / K[0,0]
    # * To create an estimate for E using DLT for a random subset of calibrated correspondences...
    # ...you can chain your functions like: E = enforce_essential(estimate_F_DLT(x1[:, randind], x2[:, randind]))

    # * Pseudo code for computing inliers:
    # e1 = compute_epipolar_errors(E, x1, x2)**2 
    # e2 = compute_epipolar_errors(E.T, x2, x1)**2
    # inliers = (1/2)*(e1+e2) < eps**2
    
    # ------ Your code here ------
    if seed is not None:
        np.random.seed(seed)    

    m = x1.shape[1]
    max_iterations = 1000  
    confidence = 0.99  
    sample_size = 8
    iters = 0

    best_inliers = None
    best_E = None
    best_inlier_count = 0

    while iters < max_iterations:
        randind = np.random.choice(m, size=sample_size, replace=False)
        
        try:
            E_candidate = enforce_essential(estimate_F_DLT(x1[:, randind], x2[:, randind]))
        except:
            iters += 1
            continue
        
        e1 = compute_epipolar_errors(E_candidate, x1, x2)**2 
        e2 = compute_epipolar_errors(E_candidate.T, x2, x1)**2
        errors = 0.5 * (e1 + e2)  
        
        inliers_mask = errors < eps**2
        num_inliers = np.sum(inliers_mask)
        
        if num_inliers > best_inlier_count:
            best_inlier_count = num_inliers
            best_inliers = inliers_mask
            best_E = E_candidate
            
            if num_inliers > 0:
                inlier_ratio = num_inliers / m
                required_iterations = int(np.ceil(np.log(1-confidence) / np.log(1-inlier_ratio**sample_size)))
                max_iterations = min(max_iterations, required_iterations)
        
        iters += 1

    print(f"RANSAC: {iters} iterations, {best_inlier_count}/{m} inliers")

    if best_inlier_count >= sample_size:
        x1_inliers = x1[:, best_inliers]
        x2_inliers = x2[:, best_inliers]
        best_E = enforce_essential(estimate_F_DLT(x1_inliers, x2_inliers))
        
        e1_refined = compute_epipolar_errors(best_E, x1_inliers, x2_inliers)**2
        e2_refined = compute_epipolar_errors(best_E.T, x2_inliers, x1_inliers)**2
        errs = np.sqrt(0.5 * (e1_refined + e2_refined))  
    else:
        errs = np.array([]) 

    E = best_E
    inliers = best_inliers
   
    return E, inliers, errs, iters
def extract_P_from_E(E):
    '''
    A function that extract the four P2 solutions given above
    E - Essential matrix (3x3)
    P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution) 
    '''
    # Your code here
    P = np.zeros((4, 3, 4))
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


def triangulate_3D_point_DLT(x1, x2, P1, P2):
    # Your code here
  A = np.array([
    x1[0] * P1[2,:] - P1[0,:],
    x1[1] * P1[2,:] - P1[1,:],
    x2[0] * P2[2,:] - P2[0,:],
    x2[1] * P2[2,:] - P2[1,:]
])
  _,_,Vt = np.linalg.svd(A)
  X = Vt[-1]
  X = X / X[-1]
  return X
  
def normalize_points(x):
    uc,vc=np.mean(x[:2,:],axis=1)
    d = np.mean(np.sqrt((x[0,:]-uc)**2 + (x[1,:]-vc)**2))
    s = np.sqrt(2) / d
    N=np.array([[s,0,-s*uc],
            [0,s,-s*vc],
            [0,0,1]])
    # Normalize the image points
    x_n =pflat(N@x)
    return x_n,N
def pflat(x):
    """ Convert points from homogeneous to Cartesian coordinates. """
    return x / x[-1, :]
def check_cheirality(P2s, x1, x2):
    '''
    P2s: 4x3x4 
    x1, x2:  (3, N)normalized points
    '''
    P1 = np.hstack((np.eye(3), np.zeros((3, 1)))) 
    
    best_ind = -1
    max_in_front = -1
    
    num_test = min(x1.shape[1], 20)
    
    for i in range(4):
        P2 = P2s[i, :, :] 
        
        valid_count = 0
        for j in range(num_test):
            X = triangulate_3D_point_DLT(x1[:, j], x2[:, j], P1, P2)
            
            x1_proj = P1 @ X
            x2_proj = P2 @ X
            
            if x1_proj[2] > 0 and x2_proj[2] > 0:
                valid_count += 1
        
        if valid_count > max_in_front:
            max_in_front = valid_count
            best_ind = i
            
    return best_ind,P2s[best_ind,:,:]
def estimate_camera_DLT(x, X):

    """
    x: 2D image points (2xN or 3xN) -
    X: 3D world points (3xN or 4xN) 
    """
    # X : (4, N)
    if X.shape[0] == 3:
        X = np.vstack([X, np.ones((1, X.shape[1]))])
        
    N = X.shape[1]
    M = []

    for i in range(N):
        Xi = X[:, i] # 3D
        xi = x[:, i] # 2D

        row1 = np.hstack([np.zeros(4), -xi[2]*Xi, xi[1]*Xi])
        row2 = np.hstack([xi[2]*Xi, np.zeros(4), -xi[0]*Xi])
        M.append(row1)
        M.append(row2)

    M = np.array(M)
    
    U, S, Vt = np.linalg.svd(M)
    v = Vt[-1]
    
    P = v.reshape(3, 4)
    
    return P  
def camera_center_and_axis(P):
    # The camera center can be found by taking the null space of the camera matrix
    camera_center = pflat(sp.linalg.null_space(P))[:3]

    principal_axis = P[-1, :3]
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    return camera_center, principal_axis

def plot_camera(camera_matrix, scale, ax=None):
    if ax is None:
        ax = plt.axes(projection='3d')
    (camera_center, principal_direction) = camera_center_and_axis(camera_matrix)

    ax.scatter3D(camera_center[0], camera_center[1], camera_center[2], c='g')
    dir = principal_direction * scale
    ax.quiver(camera_center[0], camera_center[1], camera_center[2], dir[0], dir[1], dir[2], color='r')

def estimate_camera_pose_robust(x,X,K,threshold=5.0,iters=2000):
    x_norm=np.linalg.inv(K)@x
    num_x=x.shape[1]
    best_P=None
    max_inliers=0
    best_inliers_mask = None

    for i in range(iters):
        idx=np.random.choice(num_x,6,replace=False)

        P_curr=estimate_camera_DLT(x_norm[:,idx],X[:,idx])
        x_proj=K@P_curr@X
        x_proj=pflat(x_proj)

        errors=np.linalg.norm(x_proj - x, axis=0)
        current_mask=errors<threshold
        num_inliers=np.sum(current_mask)
        if num_inliers>max_inliers:
            max_inliers=num_inliers
            best_P=P_curr
            best_inliers_mask=current_mask
    if best_P is not None:
        inlier_x=x[:,best_inliers_mask]
        inlier_X=X[:,best_inliers_mask]
        best_P=estimate_camera_DLT(np.linalg.inv(K)@inlier_x,inlier_X)
    test_X = inlier_X[:, 0] # (4,)
        
    test_X_cam = best_P @ test_X
        
    if test_X_cam[2] < 0:
        best_P = -best_P
    return best_P, best_inliers_mask

def clean_point_cloud(X, distance_threshold_scale=2.0):
    """
    X: point cloud, shape (>=3, N) or (4, N)
    distance_threshold_scale: preserve points within (median + scale * std) distance from center
    """
    #extract 3D points 
    points = X[:3, :]
    
    # calculate center
    center = np.median(points, axis=1).reshape(3, 1)
    
    # calculate every point's distance to center
    distances = np.linalg.norm(points - center, axis=0)
    
    # compute median and std of distances
    median_dist = np.median(distances)
    std_dist = np.std(distances)
    
    # set limit
    limit = median_dist + distance_threshold_scale * std_dist
    
    mask = distances < limit
    
    X_clean = X[:, mask]
    
    print(f"🧹 cloud cleaning: original points{X.shape[1]} -> {X_clean.shape[1]} after cleaning (threshold: {limit:.2f})")
    return X_clean

def extract_camera_params(all_cameras):
    """

      camera_params: (n_cameras, 6) e.s. [r1, r2, r3, t1, t2, t3]
      valid_camera_indices: img 0, 1, 2...
    """
    camera_params = []
    valid_camera_indices = []
    
    for i, P in enumerate(all_cameras):
        if P is None:
            continue
            

        R = P[:, :3]
        t = P[:, 3]
        
        #  Rodrigues
        r_vec, _ = cv2.Rodrigues(R)
        r_vec = r_vec.flatten() 

        params = np.hstack((r_vec, t))
        
        camera_params.append(params)
        valid_camera_indices.append(i)
        
    return np.array(camera_params), valid_camera_indices


def untwist(params):
    """
    Converts a 6D vector [r_vec, t_vec] back to R (3x3) and t (3,).
    """
    r_vec = params[:3]
    t_vec = params[3:]
    R, _ = cv2.Rodrigues(r_vec)
    return R, t_vec

def project_point(X,R,t):
    X_cam=R @ X + t.reshape(3,1)
    x_proj = X_cam[:2, :] / (X_cam[2, :] + 1e-9)
    return x_proj

def run_motion_ba(X_recon,all_cameras,image_map,K,kp_list):
    camera_params, valid_indices = extract_camera_params(all_cameras)
    points_3d = X_recon[:3, :]
    observations = []
    for i, real_cam_idx in enumerate(valid_indices):
        kp_curr = kp_list[real_cam_idx]
        map_cur = image_map[real_cam_idx]
        for feat_idx, point_3d_idx in map_cur.items():  
            if point_3d_idx < X_recon.shape[1]:  
                pt_pixel=kp_curr[feat_idx].pt
                pt_h= np.array([pt_pixel[0], pt_pixel[1], 1.0])
                pt_norm=np.linalg.inv(K) @ pt_h
                observations.append((i, point_3d_idx, pt_norm[0], pt_norm[1]))
    observations = np.array(observations)
    print(f"Observation data prepared: {len(observations)} entries ")
    def fun(camera_params):
        
        residuals=[]
        for i in range(len(valid_indices)):
            start=i*6
            end=start+6
            camera_params_i= camera_params[start:end]
            R_i, t_i = untwist(camera_params_i)
            mask= observations[:,0]==i#Take column 0 of rows where camera index matches i

            #skip if there are no observations for this camera
            obs_i=observations[mask] #Extract observations for camera i
            points_indices=obs_i[:,1].astype(int)
            obs_2d=obs_i[:,2:].T #Extract observed 2D points
            points_3d_i=points_3d[:,points_indices]
            proj_2d=project_point(points_3d_i,R_i,t_i)
            res= (proj_2d - obs_2d).flatten()
            residuals.extend(res)
        return residuals
    #optimize
    res=least_squares(fun, camera_params.flatten(), verbose=2, xtol=1e-2, ftol=1e-2, method='trf', max_nfev=10)
        #update cameras
    print("Optimization in progress...")
    optimized_params=res.x
    new_cameras=list(all_cameras)
    for i in range(len(valid_indices)):
        real_idx=valid_indices[i]
        start=i*6
        end=start+6

        R_new,t_new=untwist(optimized_params[start:end])
        new_cameras[real_idx]=np.hstack([R_new,t_new.reshape(3,1)])
    print("-" * 30)
    print(f"Optimization successful: {res.success}")#optimization success flag
    print(f"Termination reason: {res.message}")#optimization end reason
    print(f"Number of iterations: {res.nfev}") # iteration count
    print(f"final cost: {res.cost:.4f}")#final cost
    print("-" * 30)
    return new_cameras