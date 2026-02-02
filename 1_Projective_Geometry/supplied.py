import numpy as np
import matplotlib.pyplot as plt

def psphere(x):
    """ Normalization of projective points. Do a normalization on the projective
        points in x. Each column is considered to be a point in homogeneous coordinates.
         INPUT:
              x     - matrix in which each column is a point.
         OUTPUT:
              y     - result after normalization.
              alpha - depth
    """
    alpha = np.sqrt(np.sum(x**2, axis=0))       
    y = x / alpha                             
    return y, alpha
    
    

def rital(lines, st='r-'):
    """ 
    Takes a nx3 matrix "lines" as input where each row represents the homogeneous coordinates of a 
    2D line. The function then plots the lines in the last opened figure.
    
    Args:
        lines (np.ndarray): A NumPy array of shape (n, 3) where each row represents the homogeneous 
            coordinates of a 2D line.
        st (str): Optional argument that controls the line style of the plot.
    """
    
    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    
    nn = lines.shape[0]
    for i in range(nn):
        line = lines[i]
        rikt,_ = psphere(np.array([[line[1]], [-line[0]], [0]]))
        punkter = np.cross(rikt.T, line).reshape(-1, 1)
        punkter = punkter / punkter[-1,:]
    
        x_coords = [punkter[0, 0] - 2000 * rikt[0, 0], punkter[0, 0] + 2000 * rikt[0, 0]]
        y_coords = [punkter[1, 0] - 2000 * rikt[1, 0], punkter[1, 0] + 2000 * rikt[1, 0]]
        ax.plot(x_coords, y_coords, st)
        
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def axis3D_equal():
    """Sets equal aspect ratio in a 3D matplotlib plot by adjusting axis limits."""
    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    max_range = max(x_limits[1] - x_limits[0], 
                    y_limits[1] - y_limits[0], 
                    z_limits[1] - z_limits[0])

    mid_x = (x_limits[0] + x_limits[1]) / 2
    mid_y = (y_limits[0] + y_limits[1]) / 2
    mid_z = (z_limits[0] + z_limits[1]) / 2

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    
    