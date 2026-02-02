import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from mpl_toolkits.mplot3d import Axes3D

from project_helpers import get_dataset_info

from geometry import (
    enforce_essential, 
    model_selection, 
    triangulate_3D_point_DLT, 
    run_motion_ba, 
    plot_camera,
    clean_point_cloud,
    estimate_camera_pose_robust,
    extract_P_from_E, 
    check_cheirality
)

def run_sfm(dataset_id):
    # =========================================================
    # 1. Setup and Load Data
    # =========================================================
    print(f"Running structure-from-motion on dataset {dataset_id}")
    K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset_id)
    num_images = len(img_names)

    # =========================================================
    # 2. Feature Extraction
    # =========================================================
    print("Detecting SIFT features...")
    kp_list = []
    des_list = []
    sift = cv2.SIFT_create(nfeatures=8000)

    for img_name in img_names:
        img = cv2.imread(img_name)
        kp, des = sift.detectAndCompute(img, None)
        kp_list.append(kp)
        des_list.append(des)

    
    all_cameras = [None] * num_images          # Stores P matrices (3x4)
    image_map = [{} for _ in range(num_images)] # Stores 2D-3D mapping {feat_idx: 3d_idx}
    X_recon = np.zeros((4, 0))                 # 3D Point Cloud (4, N)
    done_indices = []                          # Registered image indices

    # =========================================================
    # 3. Initialization (Image pair idx0 & idx8)
    # =========================================================
    idx0 = init_pair[0]
    idx8 = init_pair[1]
    print(f"Initializing with images {idx0} and {idx8}...")

    # 3.1 Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_list[idx0], des_list[idx8], k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp_list[idx0][m.queryIdx].pt for m in good_matches]) # (N, 2)
    pts2 = np.float32([kp_list[idx8][m.trainIdx].pt for m in good_matches]) # (N, 2)

    # 3.2 Model Selection (H vs F)
    model_type, M, mask = model_selection(pts1, pts2, K)

    R_final, t_final = None, None
    final_mask = None

    if model_type == 'F':
        print(">>> [Init] Stereo scene detected (Fundamental Matrix)")
        E = K.T @ M @ K
        E = enforce_essential(E)
        # Recover pose
        P_candidates = extract_P_from_E(E)

        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T # (3, N)
        pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T # (3, N)
        x1_n= np.linalg.inv(K) @ pts1_h
        x2_n= np.linalg.inv(K) @ pts2_h

        inliers_F=mask.ravel()>0
        best_ind,P_best=check_cheirality(P_candidates, x1_n[:,inliers_F], x2_n[:,inliers_F])
        R_final = P_best[:, :3]
        t_final = P_best[:, 3]
        t_final = t_final.reshape(3, 1)
        mask_pose= mask
        final_mask = (mask_pose.ravel() > 0)
        #_, R_final, t_final, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        #final_mask = (mask_pose.ravel() > 0)
    else:
        print(">>> [Init] Planar scene detected (Homography Matrix)")
        H = M
        num_sol, Rs, ts, normals = cv2.decomposeHomographyMat(H, K)
        R_final = Rs[0]
        t_final = ts[0]
        t_final = t_final / (np.linalg.norm(t_final) + 1e-9)
        # For H mode, use mask returned by model_selection
        final_mask = (mask.ravel() > 0)

    # 3.3 Store Cameras
    all_cameras[idx0] = np.eye(3, 4)
    all_cameras[idx8] = np.hstack((R_final, t_final))
    done_indices = [idx0, idx8]

    # 3.4 Initial Triangulation
    pts1_in = pts1[final_mask].T # (2, M)
    pts2_in = pts2[final_mask].T # (2, M)

    # Normalize coordinates
    K_inv = np.linalg.inv(K)
    x0_h = np.vstack([pts1_in, np.ones((1, pts1_in.shape[1]))])
    x8_h = np.vstack([pts2_in, np.ones((1, pts2_in.shape[1]))])
    x0n = K_inv @ x0_h
    x8n = K_inv @ x8_h

    P0 = np.eye(3, 4)
    P8 = np.hstack((R_final, t_final))

    X_init = np.zeros((4, x0n.shape[1]))
    for i in range(x0n.shape[1]):
        X_init[:, i] = triangulate_3D_point_DLT(x0n[:, i], x8n[:, i], P0, P8)

    X_recon = X_init

    # 3.5 Build Initial Image Map
    inlier_indices = np.where(final_mask)[0]
    for col_idx, original_idx in enumerate(inlier_indices):
        m = good_matches[original_idx]
        image_map[idx0][m.queryIdx] = col_idx
        image_map[idx8][m.trainIdx] = col_idx

    print(f"Initialization complete. Generated {X_recon.shape[1]} 3D points.")

    # =========================================================
    # 4. Incremental Reconstruction Loop
    # =========================================================
    print(f"\n>>> Processing remaining {num_images - 2} images...")

    while len(done_indices) < num_images:
        
        # -----------------------------------------------------
        # Step 1: Find Best Next Image
        # -----------------------------------------------------
        best_image_idx = -1
        max_matches = 0
        
        best_2d = []         
        best_3d = []         
        best_2d_indices = [] # Original feature indices
        best_3d_indices = [] # 3D point indices
        
        best_ref_img = -1    # For triangulation
        
        for i in range(num_images):
            if i in done_indices: continue 

            curr_2d, curr_3d = [], []
            curr_2d_idx, curr_3d_idx = [], []
            match_counts = {} 

            for j in done_indices:
                bf = cv2.BFMatcher()
                raw_matches = bf.knnMatch(des_list[i], des_list[j], k=2)
                
                count_j = 0
                for m, n in raw_matches:
                    if m.distance < 0.75 * n.distance:
                        if m.trainIdx in image_map[j]:
                            p3d_idx = image_map[j][m.trainIdx]
                            
                            if m.queryIdx not in curr_2d_idx:
                                curr_2d.append(kp_list[i][m.queryIdx].pt)
                                curr_3d.append(X_recon[:3, p3d_idx])
                                curr_2d_idx.append(m.queryIdx)
                                curr_3d_idx.append(p3d_idx)
                                count_j += 1
                
                if count_j > 0:
                    match_counts[j] = count_j

            if len(curr_2d) > max_matches:
                max_matches = len(curr_2d)
                best_image_idx = i
                best_2d = np.array(curr_2d, dtype=np.float32)
                best_3d = np.array(curr_3d, dtype=np.float32)
                best_2d_indices = curr_2d_idx
                best_3d_indices = curr_3d_idx
                
                if match_counts:
                    best_ref_img = max(match_counts, key=match_counts.get)

        if max_matches < 6:
            print("Warning: Not enough matches for PnP, stopping reconstruction.")
            break
            
        print(f"\n>>> Selected image {best_image_idx} (Ref: {best_ref_img}, PnP matches: {max_matches})")

        # -----------------------------------------------------
        # Step 2: PnP Registration
        # -----------------------------------------------------
        pts2d_in = best_2d.T
        pts3d_in = best_3d.T
        pts2d_in_h = np.vstack([pts2d_in, np.ones((1, pts2d_in.shape[1]))])
        pts3d_in_h = np.vstack([pts3d_in, np.ones((1, pts3d_in.shape[1]))])
        best_P, inliers_mask = estimate_camera_pose_robust(pts2d_in_h, pts3d_in_h, K, threshold=2.0, iters=2000)

        if best_P is None:
            print("Manual PnP failed, skipping.")
            continue

        all_cameras[best_image_idx] = best_P
        done_indices.append(best_image_idx)

        
        if inliers_mask is not None:
        
            inlier_indices = np.where(inliers_mask)[0]
            for k in inlier_indices:

                image_map[best_image_idx][best_2d_indices[k]] = best_3d_indices[k]

        print(f"Manual PnP Success! Inliers: {np.sum(inliers_mask)}")
        

        # -----------------------------------------------------
        # Step 3: Triangulate New Points
        # -----------------------------------------------------
        if best_ref_img != -1:
            print(f"Triangulating new points with reference image {best_ref_img}...")
            
            bf = cv2.BFMatcher()
            new_matches = bf.knnMatch(des_list[best_image_idx], des_list[best_ref_img], k=2)
            
            pts_new_curr = []
            pts_new_ref = []
            matches_obj = []
            
            for m, n in new_matches:
                if m.distance < 0.70 * n.distance: # Ratio Test
                    idx_curr = m.queryIdx
                    idx_ref = m.trainIdx
                    
                    if (idx_curr not in image_map[best_image_idx]) and \
                       (idx_ref not in image_map[best_ref_img]):
                        
                        pts_new_curr.append(kp_list[best_image_idx][idx_curr].pt)
                        pts_new_ref.append(kp_list[best_ref_img][idx_ref].pt)
                        matches_obj.append(m)
            
            if len(pts_new_curr) > 0:
                # 1. Prepare data
                pts_new_curr = np.array(pts_new_curr).T 
                pts_new_ref = np.array(pts_new_ref).T   
                
                # 2. Triangulate
                P_curr = all_cameras[best_image_idx]
                P_ref = all_cameras[best_ref_img]
                
                x_curr_n = K_inv @ np.vstack([pts_new_curr, np.ones((1, pts_new_curr.shape[1]))])
                x_ref_n = K_inv @ np.vstack([pts_new_ref, np.ones((1, pts_new_ref.shape[1]))])
                
                X_candidates = np.zeros((4, pts_new_curr.shape[1]))
                for k in range(pts_new_curr.shape[1]):
                    X_candidates[:, k] = triangulate_3D_point_DLT(x_curr_n[:, k], x_ref_n[:, k], P_curr, P_ref)
                
                # Filter Outliers
                # A. Depth Check
                X_cam = P_curr @ X_candidates
                mask_depth = X_cam[2, :] > 0 
                X_ref = P_ref @ X_candidates
                mask_depth &= (X_ref[2, :] > 0)

                # B. Reprojection Error
                proj_curr = X_cam[:2, :] / (X_cam[2, :] + 1e-9)
                proj_ref = X_ref[:2, :] / (X_ref[2, :] + 1e-9)
                
                dist_curr = np.sum((proj_curr - x_curr_n[:2, :])**2, axis=0)
                dist_ref = np.sum((proj_ref - x_ref_n[:2, :])**2, axis=0)
                
                limit = 0.005 
                mask_error = (dist_curr < limit) & (dist_ref < limit)
                
                valid_mask = mask_depth & mask_error
                
                # Keep good points
                X_good = X_candidates[:, valid_mask]
                
                if X_good.shape[1] > 0:
                    start_idx = X_recon.shape[1]
                    X_recon = np.hstack([X_recon, X_good])
                    
                    count_added = 0
                    for i, is_good in enumerate(valid_mask):
                        if is_good:
                            m = matches_obj[i]
                            new_3d_idx = start_idx + count_added
                            image_map[best_image_idx][m.queryIdx] = new_3d_idx
                            image_map[best_ref_img][m.trainIdx] = new_3d_idx
                            count_added += 1
                    
                    print(f"Triangulation: Candidates {X_candidates.shape[1]} -> Retained {count_added} good points")
                else:
                    print("No valid new points added.")
            
        # -----------------------------------------------------
        # Step 4: Bundle Adjustment
        # -----------------------------------------------------
        print(f"Running Bundle Adjustment (Cameras: {len(done_indices)})...")
        all_cameras = run_motion_ba(X_recon, all_cameras, image_map, K, kp_list)

    print("\nReconstruction complete!")

    # =========================================================
    # 5. Visualization and Saving
    # =========================================================
    
    # Clean point cloud 
    # Using 1.5 threshold as requested
    X_final = clean_point_cloud(X_recon, distance_threshold_scale=1.5 )  

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Reconstructed 3D Points (Dataset {dataset_id})')
    
    # Scatter plot
    ax.scatter(X_final[0, :], X_final[1, :], X_final[2, :], s=1, c='k', marker='.')

    # Plot cameras
    for i in done_indices:
        plot_camera(all_cameras[i], scale=0.5, ax=ax)

    # Save visualization
    ax.view_init(elev=-90, azim=-90)
    output_png = f"result_dataset_{dataset_id}.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_png}")
    
    plt.show()

if __name__ == "__main__":
    
    dataset_id = 9
    run_sfm(dataset_id)