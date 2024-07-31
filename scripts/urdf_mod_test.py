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

def save_urdf(root, file_path):
    """
    Saves the URDF tree to a file.

    :param root: The root element of the URDF tree.
    :param file_path: The file path to save the URDF.
    """
    tree = ET.ElementTree(root)
    tree.write(file_path, xml_declaration=True, encoding='utf-8', method="xml")

# Example usage
robot = ET.Element('robot', name='example_robot')

# Create a base link
base_box_dim = 0.25
base_link = create_link('base_link', type='box', shape_dim=[base_box_dim, base_box_dim, base_box_dim], collision=False)

# Create a new link and joint
joint1 = create_joint('joint1', joint_type='revolute', parent='base_link', child='link1', axis='1 0 0', origin=[0, 0, base_box_dim/2, 0, 0, 0])
link1_length = 0.5
link1 = create_link('link1', type='cylinder', shape_dim=[0.5, 0.05], origin=[0, 0, link1_length/2, 0, 0, 0], material='blue', color=[0, 0, 1, 1])

# Create a new link and joint
joint2 = create_joint('joint2', joint_type='revolute', parent='link1', child='link2', axis='1 0 0', origin=[0, 0, link1_length, 0, 0, 0])
link2_length = 0.5
link2 = create_link('link2', type='cylinder', shape_dim=[0.5, 0.05], origin=[0, 0, link2_length/2, 0, 0, 0], material='green', color=[0, 1, 0, 1])

# Create a new link and joint
joint3 = create_joint('joint3', joint_type='revolute', parent='link2', child='link3', axis='1 0 0', origin=[0, 0, link2_length, 0, 0, 0])
link3_length = 0.5
link3 = create_link('link3', type='cylinder', shape_dim=[0.5, 0.05], origin=[0, 0, link3_length/2, 0, 0, 0], material='yellow', color=[1, 1, 0, 1])

# Create a new joint and link for the wrist
joint4 = create_joint('joint4', joint_type='fixed', parent='link3', child='wrist', origin=[0, 0, link3_length + 0.025, 0, 0, 0])
wrist = create_link('wrist', type='box', shape_dim=[0.05, 0.25, 0.05], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

# Create a new joint and link for the left finger
joint5 = create_joint('joint5', joint_type='fixed', parent='wrist', child='left_finger', origin=[0, -0.1, 0.075, 0, 0, 0])
left_finger = create_link('left_finger', type='box', shape_dim=[0.05, 0.05, 0.15], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

# Create a new joint and link for the right finger
joint6 = create_joint('joint6', joint_type='fixed', parent='wrist', child='right_finger', origin=[0, 0.1, 0.075, 0, 0, 0])
right_finger = create_link('right_finger', type='box', shape_dim=[0.05, 0.05, 0.15], origin=[0, 0, 0, 0, 0, 0], material='purple', color=[0.5, 0, 0.5, 1])

# Create a new link for the center of the gripper
gripper_center = create_link('gripper_center', type='sphere', shape_dim=[0.02], origin=[0, 0, 0, 0, 0, 0], material='red', color=[1, 0, 0, 1], collision=False)
joint7 = create_joint('joint7', joint_type='fixed', parent='wrist', child='gripper_center', origin=[0, 0, 0.15, 0, 0, 0])

# Construct robot
robot.append(base_link)
robot.append(joint1)
robot.append(link1)
robot.append(joint2)
robot.append(link2)
robot.append(joint3)
robot.append(link3)
robot.append(joint4)
robot.append(wrist)
robot.append(joint5)
robot.append(left_finger)
robot.append(joint6)
robot.append(right_finger)
robot.append(gripper_center)
robot.append(joint7)

# Save to a URDF file
save_urdf(robot, './urdf/example_robot.urdf')

