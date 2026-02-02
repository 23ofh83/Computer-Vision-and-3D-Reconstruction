# Computer Vision & 3D Reconstruction Portfolio

This repository contains a comprehensive collection of Computer Vision projects implemented using **Python**. It covers the pipeline from fundamental projective geometry to a complete Structure from Motion (SfM) system.

## 📂 Project Structure

### 🚀 Capstone Project: Structure from Motion (SfM)
**Folder:** `5_Project_Structure_from_Motion`
A complete pipeline to reconstruct 3D scene geometry from 2D image sequences.
* **Key Algorithms:**
    * Calculated relative orientations ($R_{i,i+1}$, $T_{i,i+1}$) between image pairs.
    * Upgraded relative rotations to absolute rotations ($R_i$).
    * **3D Reconstruction:** Initialized point clouds from image pairs.
    * **Camera Resectioning:** Robustly calculated camera centers ($C_i$) and translations.
    * **Optimization:** Refined camera parameters using **Levenberg-Marquardt** method.
    * **Visualization:** Triangulated points and visualized 3D structures + camera poses.

### 📚 Core Assignments

#### 1. Projective Geometry
**Folder:** `1_Projective_Geometry`
* Implementation of mathematical foundations for CV.
* Representations of points, lines, and planes.
* Homogeneous transformations and camera matrix operations.

#### 2. Camera Calibration & DLT
**Folder:** `2_Camera_Calibration_DLT`
* **Camera Calibration:** Solved resection and triangulation problems using the **Direct Linear Transform (DLT)** method.
* Computed inner parameters using **RQ factorization**.
* **Feature Matching:** Implemented **SIFT** for robust feature detection and matching.

#### 3. Epipolar Geometry
**Folder:** `3_Epipolar_Geometry`
* 3D structure recovery from stereo pairs.
* Estimation of the **Fundamental Matrix** and **Essential Matrix**.
* Simultaneous reconstruction of 3D structure and camera motion.

#### 4. Model Fitting & Local Optimization
**Folder:** `4_RANSAC_Optimization`
* Robust estimation of camera parameters in the presence of outliers.
* Implemented **RANSAC** (Random Sample Consensus) for robust fitting.
* Joint optimization of scene geometry using **Bundle Adjustment**.

## 🛠 Tech Stack
* **Languages:** Python
* **Libraries:** NumPy, Matplotlib, SciPy
* **Concepts:** Linear Algebra, Optimization, 3D Mathematics
