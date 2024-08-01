import xml.etree.ElementTree as ET

def create_link(name, type='cylinder', shape_dim=[1, 0.1], origin=[0, 0, 0, 0, 0, 0], material='gray', color=[0.5, 0.5, 0.5, 1], collision=True, inertial=False):
    """
    Creates a new link element.

    :param name: Name of the link.
    :param type: Type of link shape ('cylinder' or 'box')
    :param origin: Origin of the joint (x y z rx ry rz)
    :param length: Length of the link.
    :param radius: Radius of the link.
    :return: A new link element.
    """
    link = ET.Element('link', name=name)
    
    if inertial:
        inertial = ET.SubElement(link, 'inertial')
        mass = ET.SubElement(inertial, 'mass', value='1.0')
        inertia = ET.SubElement(inertial, 'inertia',
                                ixx='0.0', ixy='0.0', ixz='0.0',
                                iyy='0.0', iyz='0.0', izz='0.0')
    
    visual = ET.SubElement(link, 'visual')
    visual_origin = ET.SubElement(visual, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
    geometry = ET.SubElement(visual, 'geometry')
    material = ET.SubElement(visual, 'material', name=material)
    color = ET.SubElement(material, 'color', rgba=' '.join(map(str, color)))

    if collision:
        collision = ET.SubElement(link, 'collision')
        collision_origin = ET.SubElement(collision, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        collision_geometry = ET.SubElement(collision, 'geometry')

    if type == 'cylinder':
        cylinder = ET.SubElement(geometry, 'cylinder', length=str(shape_dim[0]), radius=str(shape_dim[1]))

        if collision:
            collision_cylinder = ET.SubElement(collision_geometry, 'cylinder', length=str(shape_dim[0]), radius=str(shape_dim[1]))

    if type == 'box':
        box = ET.SubElement(geometry, 'box', size=' '.join(map(str, shape_dim)))
        if collision:
            collision_box = ET.SubElement(collision_geometry, 'box', size=' '.join(map(str, shape_dim)))

    if type == 'sphere':
        sphere = ET.SubElement(geometry, 'sphere', radius=' '.join(map(str, shape_dim)))
        if collision:
            collision_sphere = ET.SubElement(collision_geometry, 'sphere', radius=' '.join(map(str, shape_dim)))
    
    return link

def create_joint(name, parent, child, origin=[0, 0, 0, 0, 0, 0], joint_type='revolute', axis='0 0 1', limit=[-1.57, 1.57], effort=10, velocity=1):
    """
    Creates a new joint element.

    :param name: Name of the joint.
    :param parent: Name of the parent link.
    :param child: Name of the child link.
    :param origin: Origin of the joint (x y z rx ry rz)
    :param joint_type: Type of the joint (default is 'revolute').
    :param axis: Axis of the joint (default is '0 0 1').
    :param limit: Lower and upper limit of the joint [lower, upper]
    :param effort: Effort of the joint
    :param velocity: Velocity of the joint
    :return: A new joint element.
    """
    joint = ET.Element('joint', name=name, type=joint_type)
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=child)
    ET.SubElement(joint, 'axis', xyz=axis)
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
    
    if joint_type in ['revolute', 'prismatic']:
        ET.SubElement(joint, 'limit', lower=str(limit[0]), upper=str(limit[1]), effort=str(effort), velocity=str(velocity))
    
    return joint

def custom_gripper(robot, parent, last_link_length):
    # Create a new joint and link for the wrist
    wrist_joint = create_joint('wrist_joint', joint_type='fixed', parent=parent, child='wrist', origin=[0, 0, last_link_length + 0.025, 0, 0, 0])
    wrist = create_link('wrist', type='box', shape_dim=[0.05, 0.25, 0.05], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

    # Create a new joint and link for the left finger
    left_finger_joint = create_joint('left_finger_joint', joint_type='fixed', parent='wrist', child='left_finger', origin=[0, -0.1, 0.075, 0, 0, 0])
    left_finger = create_link('left_finger', type='box', shape_dim=[0.05, 0.05, 0.15], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

    # Create a new joint and link for the right finger
    right_finger_joint = create_joint('right_finger_joint', joint_type='fixed', parent='wrist', child='right_finger', origin=[0, 0.1, 0.075, 0, 0, 0])
    right_finger = create_link('right_finger', type='box', shape_dim=[0.05, 0.05, 0.15], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

    # Create a new link for the center of the gripper
    gripper_center_joint = create_joint('gripper_center_joint', joint_type='fixed', parent='wrist', child='gripper_center', origin=[0, 0, 0.15, 0, 0, 0])
    gripper_center = create_link('gripper_center', type='sphere', shape_dim=[0.02], origin=[0, 0, 0, 0, 0, 0], material='red', color=[1, 0, 0, 1], collision=False)

    # Attach gripper to end of robot
    robot.append(wrist_joint)
    robot.append(wrist)
    robot.append(left_finger_joint)
    robot.append(left_finger)
    robot.append(right_finger_joint)
    robot.append(right_finger)
    robot.append(gripper_center)
    robot.append(gripper_center_joint)

def save_urdf(root, file_path):
    """
    Saves the URDF tree to a file.

    :param root: The root element of the URDF tree.
    :param file_path: The file path to save the URDF.
    """
    tree = ET.ElementTree(root)
    tree.write(file_path, xml_declaration=True, encoding='utf-8', method="xml")

def create_planar_manipulator(filename, robot_name, shape_dims, prismatic_axis='0 1 0', link_shape='cylinder'):
    """
    Creates a manipulator chain URDF.

    :param filename: Name of the URDF file to be saved
    :param robot_name: Name of the robot within URDF
    :param shape_dims: List of lists -> [(link_length (float), shape_radii (float))]
    :param prismatic_axis: String -> Axis of translation for prismatic joint (default '0 1 0')
    :param link_shape: Shape of link ('cylinder', 'box', etc.)
    """

    # Initialize urdf
    robot = ET.Element('robot', name=robot_name)

    colors = {'blue': [0, 0, 1, 1], 'green': [0, 1, 0, 1], 'yellow': [1, 1, 0, 1], 'pink': [1, 0, 1, 1], 'red': [1, 0, 0, 1]}

    # Find the positions of all prismatic joints
    prismatic_pos = [idx for idx, dims in enumerate(shape_dims) if dims == [0, 0]]

    # Find the link indices on either side of prismatic joints
    collision_idx = []
    if prismatic_axis == '0 0 1':
        collision_idx = [num - 1 for num in prismatic_pos if num - 1 >= 0] + [num + 1 for num in prismatic_pos]

    # Generate base link
    base_link_name = 'base_link'
    base_link_size = 0.25
    base_link = create_link(name=base_link_name, type='box', shape_dim=[base_link_size, base_link_size, base_link_size], material='gray', color=[0.5, 0.5, 0.5, 1], collision=False)
    robot.append(base_link)

    parent_name = base_link_name
    parent_length = base_link_size / 2  # Parent link length is halved for the first joint origin

    # Turn on collision tag generation
    collision = True

    # Loop through each joint/link configuration
    for i, (child_length, shape_radius) in enumerate(shape_dims):
        color_name, color_code = list(colors.items())[i]

        # Determine joint type and axis
        if i in prismatic_pos:
            joint_type = 'prismatic'
            axis = prismatic_axis
            parent_length += 0.005 # Add a little buffer for collision issues
        else:
            joint_type = 'revolute'
            axis = '1 0 0'
        
        # Set collision tag to False if prismatic actuation in z-dir
        if i in collision_idx:
            collision = False

        # Create joint and link
        joint_name = f'joint{i}'
        child_name = f'link{i}'
        joint = create_joint(joint_name, joint_type=joint_type, parent=parent_name, child=child_name, axis=axis, origin=[0, 0, parent_length, 0, 0, 0])
        link = create_link(child_name, type=link_shape, shape_dim=[child_length, shape_radius], origin=[0, 0, child_length / 2, 0, 0, 0], material=color_name, color=color_code, collision=collision)

        # Attach joint and link to the robot
        robot.append(joint)
        robot.append(link)

        # Update parent name and length for the next iteration
        parent_name = child_name
        parent_length = child_length

    # Add a custom gripper if necessary
    custom_gripper(robot, parent=parent_name, last_link_length=parent_length)

    # Save the URDF to a file
    save_urdf(robot, f'./urdf/{filename}.urdf')


shape_dims = [[0, 0], [0.5, 0.05], [0, 0], [0.5, 0.05], [0.5, 0.05]]

create_planar_manipulator('auto_gen_manip', 'new_robot', shape_dims=shape_dims, prismatic_axis='0 1 0')
