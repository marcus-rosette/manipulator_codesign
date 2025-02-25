import numpy as np
import pytest

from manipulator_codesign.kinematic_chain import KinematicChainBase, KinematicChainPyBullet, KinematicChainRTB


def test_compute_pose_error():
    # Define test cases
    test_cases = [
        # Exact match, should return 0
        ((([0, 0, 0], [0, 0, 0, 1]), ([0, 0, 0], [0, 0, 0, 1])), 0.0),
        
        # Pure position error
        ((([1, 0, 0], [0, 0, 0, 1]), ([0, 0, 0], [0, 0, 0, 1])), 1.0),
        
        # Pure orientation error (90-degree rotation about z-axis)
        ((([0, 0, 0], [0, 0, np.sqrt(2)/2, np.sqrt(2)/2]), ([0, 0, 0], [0, 0, 0, 1])), np.pi / 2),
        
        # Combined error (position + orientation)
        ((([1, 0, 0], [0, 0, np.sqrt(2)/2, np.sqrt(2)/2]), ([0, 0, 0], [0, 0, 0, 1])), 1.0 + np.pi / 2),
    ]
    
    for (poses, expected_error) in test_cases:
        target_pose, actual_pose = poses
        computed_error = KinematicChainPyBullet.compute_pose_error(target_pose, actual_pose)
        assert np.isclose(computed_error, expected_error, atol=1e-6), f"Failed for poses: {poses}"
