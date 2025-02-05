import os
import xml.etree.ElementTree as ET

class URDFGen:
    def __init__(self, robot_name, save_urdf_dir='urdf/'):
        """
        Initializes the URDF generator with the specified robot name and directory to save the URDF file.

        Args:
            robot_name (str): The name of the robot.
            save_urdf_dir (str, optional): The directory where the URDF file will be saved. Defaults to 'urdf/'.
        """
        self.robot = ET.Element('robot', name=robot_name)
        self.save_urdf_dir = save_urdf_dir
    
    def create_link(self, name, type='cylinder', shape_dim=[1, 0.1], origin=[0, 0, 0, 0, 0, 0], material='gray', color=[0.5, 0.5, 0.5, 1], collision=True, inertial=False):
        """
        Create a URDF link element.
        Parameters:
        name (str): The name of the link.
        type (str): The type of the shape ('cylinder', 'box', 'sphere'). Default is 'cylinder'.
        shape_dim (list): Dimensions of the shape. Default is [1, 0.1].
        origin (list): Origin of the link in the format [x, y, z, roll, pitch, yaw]. Default is [0, 0, 0, 0, 0, 0].
        material (str): Name of the material. Default is 'gray'.
        color (list): RGBA color of the material. Default is [0.5, 0.5, 0.5, 1].
        collision (bool): Whether to include collision element. Default is True.
        inertial (bool): Whether to include inertial element. Default is False.
        Returns:
        xml.etree.ElementTree.Element: The created link element.
        """
        link = ET.Element('link', name=name)
        
        if inertial:
            inertial = ET.SubElement(link, 'inertial')
            ET.SubElement(inertial, 'mass', value='1.0')
            ET.SubElement(inertial, 'inertia',
                          ixx='0.0', ixy='0.0', ixz='0.0',
                          iyy='0.0', iyz='0.0', izz='0.0')
        
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        geometry = ET.SubElement(visual, 'geometry')
        material_elem = ET.SubElement(visual, 'material', name=material)
        ET.SubElement(material_elem, 'color', rgba=' '.join(map(str, color)))

        if collision:
            collision_elem = ET.SubElement(link, 'collision')
            ET.SubElement(collision_elem, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
            collision_geometry = ET.SubElement(collision_elem, 'geometry')

        shape_elements = {'cylinder': 'cylinder', 'box': 'box', 'sphere': 'sphere'}
        if type in shape_elements:
            ET.SubElement(geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
            if collision:
                ET.SubElement(collision_geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
        
        return link
    
    def create_joint(self, name, parent, child, origin=[0, 0, 0, 0, 0, 0], joint_type='revolute', axis='0 0 1', limit=[-1.57, 1.57], effort=10, velocity=1):
        """
        Create a joint element for a URDF (Unified Robot Description Format) file.
        Parameters:
        name (str): The name of the joint.
        parent (str): The name of the parent link.
        child (str): The name of the child link.
        origin (list): The origin of the joint in the format [x, y, z, roll, pitch, yaw]. Default is [0, 0, 0, 0, 0, 0].
        joint_type (str): The type of the joint (e.g., 'revolute', 'prismatic'). Default is 'revolute'.
        axis (str): The axis of rotation or translation in the format 'x y z'. Default is '0 0 1'.
        limit (list): The limits of the joint in the format [lower, upper]. Default is [-1.57, 1.57].
        effort (float): The maximum effort that can be applied by the joint. Default is 10.
        velocity (float): The maximum velocity of the joint. Default is 1.
        Returns:
        xml.etree.ElementTree.Element: The joint element.
        """
        joint = ET.Element('joint', name=name, type=joint_type)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'axis', xyz=axis)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        
        if joint_type in ['revolute', 'prismatic']:
            ET.SubElement(joint, 'limit', lower=str(limit[0]), upper=str(limit[1]), effort=str(effort), velocity=str(velocity))
        
        return joint
    
    def add_custom_gripper(self, parent, last_link_length):
        self.robot.append(self.create_joint('wrist_joint', parent, 'wrist', [0, 0, last_link_length + 0.025, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('wrist', 'box', [0.05, 0.25, 0.05], material='purple', color=[0.5, 0, 0.5, 1]))
        self.robot.append(self.create_joint('left_finger_joint', 'wrist', 'left_finger', [0, -0.1, 0.075, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('left_finger', 'box', [0.05, 0.05, 0.15], material='purple', color=[0.5, 0, 0.5, 1]))
        self.robot.append(self.create_joint('right_finger_joint', 'wrist', 'right_finger', [0, 0.1, 0.075, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('right_finger', 'box', [0.05, 0.05, 0.15], material='purple', color=[0.5, 0, 0.5, 1]))
    
    def save_urdf(self, file_name):
        """
        Save the URDF (Unified Robot Description Format) file with a unique name.

        This method ensures that the file name ends with '.urdf' and saves it to the specified directory.
        If a file with the same name already exists, it appends a counter to the file name to make it unique.

        Args:
            file_name (str): The name of the URDF file to be saved.

        Returns:
            None
        """
        if not file_name.endswith('.urdf'):
            file_name += '.urdf'

        if not os.path.exists(self.save_urdf_dir):
            os.makedirs(self.save_urdf_dir)

        file_path = os.path.join(self.save_urdf_dir, file_name)
        base, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base}_{counter}{ext}"
            counter += 1

        print(f"Saving file to: {file_path}")

        tree = ET.ElementTree(self.robot)
        tree.write(file_path, xml_declaration=True, encoding='utf-8', method='xml')
    
    def create_manipulator(self, axes, shape_dims, link_shape='cylinder', gripper=False, collision=False):
        """
        Creates a manipulator robot model with specified parameters.
        Args:
            axes (list): A list of axes for each joint in the manipulator.
            shape_dims (list): A list of tuples where each tuple contains the length and radius of each link.
            link_shape (str, optional): The shape of the links. Default is 'cylinder'.
            collision (bool, optional): Whether to enable collision for the links. Default is True.
            gripper (bool, optional): Whether to add a custom gripper at the end of the manipulator. Default is False.
        Returns:
            None
        """
        colors = {'blue': [0, 0, 1, 1], 'green': [0, 1, 0, 1], 'yellow': [1, 1, 0, 1], 'purple': [0.5, 0, 0.5, 1], 'pink': [1, 0, 1, 1], 'red': [1, 0, 0, 1]}
        
        prismatic_pos = [idx for idx, dims in enumerate(shape_dims) if dims == [0, 0]]
        collision_idx = [num - 1 for num in prismatic_pos if num - 1 >= 0] + [num + 1 for num in prismatic_pos]
        
        base_link_name = 'base_link'
        base_link_size = 0.25
        self.robot.append(self.create_link(name=base_link_name, type='box', shape_dim=[base_link_size, base_link_size, base_link_size], material='gray', color=[0.5, 0.5, 0.5, 1], collision=False))
        
        parent_name = base_link_name
        parent_length = base_link_size / 2
        
        for i, (axis, (child_length, shape_radius)) in enumerate(zip(axes, shape_dims)):
            color_name, color_code = list(colors.items())[i]
            joint_type = 'prismatic' if i in prismatic_pos else 'revolute'
            if i in prismatic_pos:
                parent_length += 0.005
            if i in collision_idx:
                collision = False
            
            joint_name = f'joint{i}'
            child_name = f'link{i}'
            self.robot.append(self.create_joint(joint_name, parent=parent_name, child=child_name, joint_type=joint_type, axis=axis, origin=[0, 0, parent_length, 0, 0, 0]))
            self.robot.append(self.create_link(child_name, type=link_shape, shape_dim=[child_length, shape_radius], origin=[0, 0, child_length / 2, 0, 0, 0], material=color_name, color=color_code, collision=collision))
            
            parent_name = child_name
            parent_length = child_length
        
        if gripper:
            self.add_custom_gripper(parent=parent_name, last_link_length=parent_length)
    
    @staticmethod
    def _shape_attributes(shape_type, shape_dim):
        """
        Generate a dictionary of attributes for a given shape and its dimensions.

        Args:
            shape_type (str): The type of shape ('cylinder', 'box', or 'sphere').
            shape_dim (list or tuple): The dimensions of the shape. For 'cylinder', 
                                       provide [length, radius]. For 'box', provide 
                                       [length, width, height]. For 'sphere', provide 
                                       [radius].

        Returns:
            dict: A dictionary containing the shape attributes. For 'cylinder', the 
                  dictionary contains 'length' and 'radius'. For 'box', the dictionary 
                  contains 'size'. For 'sphere', the dictionary contains 'radius'. 
                  Returns an empty dictionary if the shape is not recognized.
        """
        if shape_type == 'cylinder':
            return {'length': str(shape_dim[0]), 'radius': str(shape_dim[1])}
        elif shape_type == 'box':
            return {'size': ' '.join(map(str, shape_dim))}
        elif shape_type == 'sphere':
            return {'radius': str(shape_dim[0])}
        return {}


if __name__ == '__main__':
    robot_name = 'maybe'
    urdf_gen = URDFGen('dumb_robot')

    # axes = ['0 0 1', '0 0 1', '1 0 0', '1 0 0', '0 1 0', '1 0 0']
    # shape_dims = [[0.5, 0.05], [0, 0], [0.5, 0.05], [0.5, 0.05], [0.5, 0.05], [0.5, 0.05]]

    axes = ['0 1 0', '1 0 0', '0 1 0', '1 0 0', '1 0 0']
    shape_dims = [[0, 0], [0.5, 0.05], [0, 0], [0.5, 0.05], [0.5, 0.05]]
    urdf_gen.create_manipulator(axes, shape_dims, link_shape='cylinder', gripper=True)
    urdf_gen.save_urdf(robot_name)