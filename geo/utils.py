import plotly.graph_objects as go
import numpy as np

def rotation_matrix(yaw, pitch, roll):
    """
    Creates a rotation matrix from Euler angles (yaw, pitch, roll).
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def plot_ellipsoid(center, radii, R):
    """
    Plots a 3D ellipsoid using Plotly with an arbitrary orientation.

    Parameters:
    center (tuple): The (x, y, z) coordinates of the center of the ellipsoid.
    radii (tuple): The radii (a, b, c) of the ellipsoid along the x, y, and z axes respectively.
    orientation (tuple): The orientation of the ellipsoid in terms of Euler angles (yaw, pitch, roll).
    """
    x0, y0, z0 = center
    a, b, c = radii
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Parametric equations for the ellipsoid
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)

    # Apply the rotation
    xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    rotated_xyz = R @ xyz
    x_rot = rotated_xyz[0, :].reshape(x.shape) + x0
    y_rot = rotated_xyz[1, :].reshape(y.shape) + y0
    z_rot = rotated_xyz[2, :].reshape(z.shape) + z0
    return go.Surface(x=x_rot, y=y_rot, z=z_rot, colorscale='Viridis')
    
