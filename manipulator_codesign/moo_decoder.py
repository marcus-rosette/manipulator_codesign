import numpy as np


def decode_decision_vector(x, min_joints=2, max_joints=5):
    """
    Decode a decision vector into kinematic chain parameters.
    
    - x[0]: Number of joints (integer in [min_joints, max_joints])
    - For each potential joint (i = 0,...,max_joints-1):
         x[3*i + 1]: Joint type (0 or 1)
         x[3*i + 2]: Joint axis (0, 1, or 2 corresponding to 'x', 'y', or 'z')
         x[3*i + 3]: Link length (continuous value in [0.1, 0.75])
    
    Only the first 'num_joints' are used.
    """
    num_joints = int(np.rint(x[0]))
    num_joints = max(min_joints, min(num_joints, max_joints))
    
    joint_types, joint_axes, link_lengths = [], [], []
    axis_map = {0: 'x', 1: 'y', 2: 'z'}
    
    for i in range(num_joints):
        idx = 3 * i + 1
        joint_type = int(np.rint(x[idx]))
        joint_type = min(1, max(0, joint_type))
        joint_types.append(joint_type)
        
        joint_axis = int(np.rint(x[idx + 1]))
        joint_axis = min(2, max(0, joint_axis))
        joint_axes.append(axis_map[joint_axis])
        
        length = np.clip(x[idx + 2], 0.1, 0.75)
        link_lengths.append(length)
    
    return num_joints, joint_types, joint_axes, link_lengths