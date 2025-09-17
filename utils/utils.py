# function for suppress print
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
# from torch_geometric.data import Data
def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 
        
# Rotation stuff
def axis_angle_to_matrix(axis_angle):
    axis = axis_angle[:3]
    angle = np.linalg.norm(axis)
    if angle > 0:
        axis = axis / angle  # Normalize the axis
    return R.from_rotvec(axis * angle).as_matrix()

def get_heading_direction(theta):
    """
    Get the heading direction from the SMPL theta parameters.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.

    Returns:
        heading_direction: (3,) array representing the heading direction.
    """
    theta = theta.reshape(-1, 3)
    # Extract the root joint's rotation (usually the first joint in SMPL)
    root_rotation_axis_angle = theta[0]
    
    # Convert the root joint's axis-angle representation to a rotation matrix
    root_rotation_matrix = axis_angle_to_matrix(root_rotation_axis_angle)
    
    # Define the forward direction vector (assume [1, 0, 0] for X-forward)
    forward_vector = np.array([1, 0, 0])
    
    # Apply the root joint's rotation to the forward vector
    heading_direction = root_rotation_matrix @ forward_vector
    
    return heading_direction.reshape(-1)

def matrix_to_axis_angle(matrix):
    rot = R.from_matrix(matrix)
    return rot.as_rotvec()

def apply_z_rotation_on_theta(theta, z_rotation_angle):
    """
    Apply a rotation around the Z-axis to the SMPL theta parameter.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.
        z_rotation_angle: rotation angle around the Z-axis in radians.

    Returns:
        (N, 3) array of updated axis-angle representations.
    """
    theta = theta.reshape(-1, 3)
    z_rotation_matrix = R.from_euler('z', z_rotation_angle).as_matrix()
    
    updated_theta = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        rotation_matrix = axis_angle_to_matrix(theta[i])
        new_rotation_matrix = z_rotation_matrix @ rotation_matrix
        new_axis_angle = matrix_to_axis_angle(new_rotation_matrix)
        updated_theta[i] = new_axis_angle
    
    return updated_theta.reshape(-1)

