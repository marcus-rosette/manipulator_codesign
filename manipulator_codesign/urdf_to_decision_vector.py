import numpy as np
import xml.etree.ElementTree as ET

from manipulator_codesign.pyb_utils import PybUtils
from manipulator_codesign.load_robot import LoadRobot


def parse_cylinder_lengths(urdf_path):
    """
    Returns a dict mapping link-name → true cylinder length (float)
    """
    tree = ET.parse(urdf_path)
    cyls = {}
    for link in tree.getroot().findall('link'):
        cyl = link.find('visual/geometry/cylinder')
        if cyl is not None:
            cyls[link.attrib['name']] = float(cyl.attrib['length'])
    return cyls

def get_link_length(robot, joint_idx, cylinder_lengths):
    """
    Returns the true cylinder length for the child link of `joint_idx`, 
    or 2x the distance between joint and child link if not present.
    """
    joint_pos = robot.con.getJointState(robot.robotId, joint_idx)[0]
    child_pos = robot.con.getJointState(robot.robotId, joint_idx if robot.con.getJointInfo(robot.robotId, joint_idx)[2] == robot.con.JOINT_FIXED else joint_idx)[0]
    child_name = robot.con.getJointInfo(robot.robotId, joint_idx)[12].decode()

    if child_name in cylinder_lengths:
        return cylinder_lengths[child_name]
    else:
        return 2.0 * np.linalg.norm(np.array(child_pos) - np.array(joint_pos))

def get_logical_joints(robot):
    """
    Returns a list of either
        - int          (rev/prism joints)
        - [ix,iy,iz]  (spherical triplets)
    in the order of controllable joints.
    """
    seen = set()
    logical = []
    for j in robot.controllable_joint_idx:
        if j in seen:
            continue

        if j in robot.spherical_groups:
            trip = next(t for t in robot.spherical_joint_idx if t[0] == j)
            logical.append(trip)
            seen.update(trip)
        else:
            logical.append(j)
            seen.add(j)
    return logical

def urdf_to_decision_vector(urdf_path):
    # 1) load robot and let it detect spherical groups
    pyb = PybUtils(renders=False)
    robot = LoadRobot(
        pyb.con,
        urdf_path,
        [0, 0, 0],
        pyb.con.getQuaternionFromEuler([0, 0, 0]),
        home_config=None,
        collision_objects=[],
        ee_link_name='end_effector',
    )

    # 2) parse URDF once for true cylinder lengths
    cylinder_lengths = parse_cylinder_lengths(urdf_path)

    # 3) build logical sequence in URDF/joint order
    logical_joints = get_logical_joints(robot)

    # 3) extract properties for each logical joint
    types, axes, lengths = [], [], []
    for item in logical_joints:
        if isinstance(item, list):
            # spherical
            types.append('spherical')
            # collect each axis vector
            axes.append([tuple(pyb.con.getJointInfo(robot.robotId, j)[13]) for j in item])
            # length is on the fixed joint after the triplet
            fixed = robot.spherical_groups[item[0]]
            lengths.append(get_link_length(robot, fixed, cylinder_lengths))

        else:
            # revolute or prismatic
            info = pyb.con.getJointInfo(robot.robotId, item)
            types.append('revolute' if info[2] == pyb.con.JOINT_REVOLUTE else 'prismatic')
            axes.append(tuple(info[13]))
            lengths.append(get_link_length(robot, item, cylinder_lengths))

    return len(types), types, axes, lengths

def encode_seed(dec_vec, max_joints=7, min_j=2):
    """
    Turn the output of urdf_to_decision_vector:
       (n_joints, types, axes, lengths)
    into a 1D numpy array [n_joints, type0, axis0, len0, …],
    padded out to max_joints with a safe “fallback” design.
    """
    n_joints, types, axes, lengths = dec_vec

    # 1) mapping from your string‐types to integer codes
    type_map = {
       'revolute': 0,
       'prismatic': 1,
       'spherical': 2
    }

    # 2) mapping from axis‐vectors to the integer codes your problem uses,
    #    plus a special code for spherical
    axis_map = {
      (1, 0, 0): 0,
      (0, 1, 0): 1,
      (0, 0, 1): 2,
      'spherical': 3
    }

    vec = []
    # first variable: how many joints
    vec.append(float(n_joints))

    # for each actual joint, append [type_id, axis_id, length]
    for t, a, L in zip(types, axes, lengths):
        # map joint‐type string → integer
        t_id = type_map[t]

        # special‐case spherical joints:
        if t == 'spherical':
            a_key = 'spherical'
        else:
            # for revolute/prismatic: a is a tuple of floats
            a_key = tuple(int(round(x)) for x in a)

        # look up our integer code
        try:
            a_id = axis_map[a_key]
        except KeyError:
            raise ValueError(f"Axis vector {a!r} not one of principal axes")

        vec.extend([float(t_id), float(a_id), float(L)])

    # pad out to exactly max_joints
    while len(vec) < 1 + 3*max_joints:
        vec.extend([float(min_j), 0.0, 0.1])

    return np.asarray(vec, dtype=float)


if __name__ == "__main__":
    urdf_path = "manipulator_codesign/urdf/robots/test_robot_16.urdf"
    total_joint_count, joint_types, joint_axes, link_lengths = urdf_to_decision_vector(
        urdf_path
    )
    print("Total joints in the system:", total_joint_count)
    print("Joint types:", joint_types)
    print("Joint axes:", joint_axes)
    print("Link lengths:", link_lengths)
