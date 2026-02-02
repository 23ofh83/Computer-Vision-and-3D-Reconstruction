import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Helper functions
def pflat(x):
    # Normalizes (m,n)-array x of n homogeneous points
    # to last coordinate 1.
    y = x / x[-1,:]
    return y

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