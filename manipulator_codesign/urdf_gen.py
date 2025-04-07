import os
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
import tempfile
import random
from xml.dom import minidom
import xml.dom.minidom


class URDFGen:
    def __init__(self, robot_name, save_urdf_dir=None):
        """
        Initializes the URDF generator with the specified robot name and directory to save the URDF file.

        Args:
            robot_name (str): The name of the robot.
            save_urdf_dir (str, optional): The directory where the URDF file will be saved. Defaults to 'urdf/'.
        """
        # TODO: Add a cache clearing method for emptying the tmp/ dir
        self.robot = ET.Element('robot', name=robot_name)

        if save_urdf_dir is None:
            current_working_directory = os.getcwd()
            self.save_urdf_dir = os.path.join(current_working_directory, 'manipulator_codesign/urdf/robots/')
        else:
            self.save_urdf_dir = save_urdf_dir
    
    def create_link(self, name, type='cylinder', link_len=1, link_width=0.05, link_height=None, mass=1.0, origin=[0, 0, 0, 0, 0, 0], material='gray', color=[0.5, 0.5, 0.5, 1], collision=True):#, inertial=True):
        """
        Create a URDF link element.
        
        Args:
            name (str): The name of the link.
            type (str): The type of the shape ('cylinder', 'box', 'sphere'). Default is 'cylinder'.
            link_len (float): The length of the link. Default is 1.
            link_width (float): The width (or radius) of the link. Default is 0.05.
            mass (float): The mass of the link. Default is 1.0.
            origin (list): Origin of the link in the format [x, y, z, roll, pitch, yaw]. Default is [0, 0, 0, 0, 0, 0].
            material (str): Name of the material. Default is 'gray'.
            color (list): RGBA color of the material. Default is [0.5, 0.5, 0.5, 1].
            collision (bool): Whether to include collision element. Default is True.
            inertial (bool): Whether to include inertial element. Default is True.
        
        Returns:
            xml.etree.ElementTree.Element: The created link element.
        """
        link = ET.Element('link', name=name)
        
        inertial = ET.SubElement(link, 'inertial')
        mass = float(mass)
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', **self.inertial_calculation(mass, link_len, link_width, type))
        ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        
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
        shape_dim = [link_len, link_width]
        if link_height is not None:
            shape_dim.append(link_height)
        if type in shape_elements:
            ET.SubElement(geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
            if collision:
                ET.SubElement(collision_geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
        
        return link
    
    def create_joint(self, name, parent, child, origin=[0, 0, 0, 0, 0, 0], joint_type='revolute', axis='0 0 1', limit=[-1.57, 1.57], effort=10, velocity=1):
        """
        Create a joint element for a URDF (Unified Robot Description Format) file.
        Args:
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
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        
        if joint_type in ['revolute', 'prismatic']:
            ET.SubElement(joint, 'axis', xyz=axis)
            ET.SubElement(joint, 'limit', lower=str(limit[0]), upper=str(limit[1]), effort=str(effort), velocity=str(velocity))
        
        return joint
    
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

        # Convert ElementTree to a string
        rough_string = ET.tostring(self.robot, encoding='utf-8')

        # Pretty-print using minidom
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")

        # Write the formatted XML to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        print(f"\nSaved new urdf to: {file_path}")

    def save_temp_urdf(self):
        """
        Save the URDF (Unified Robot Description Format) file to a temporary file.

        This method saves the URDF file to a temporary location and returns the path of the saved file.

        Returns:
            str: The path of the saved temporary URDF file.
        """
        # Define the directory for temporary URDF files
        tmp_dir = os.path.join(tempfile.gettempdir(), "urdf")
        os.makedirs(tmp_dir, exist_ok=True)  # Ensure the directory exists

        # Generate a unique file name
        tmp_file_path = os.path.join(tmp_dir, next(tempfile._get_candidate_names()) + ".urdf")

        # Convert URDF to pretty XML
        rough_string = ET.tostring(self.robot, encoding="utf-8")
        pretty_string = xml.dom.minidom.parseString(rough_string).toprettyxml(indent="  ")

        # Write to the file
        with open(tmp_file_path, "w", encoding="utf-8") as tmpfile:
            tmpfile.write(pretty_string)
            
        return tmp_file_path
        
    def add_custom_gripper(self, parent, last_link_length):
        # TODO: Need to update functionality for changing shape_dim to just the link lengths
        self.robot.append(self.create_joint('wrist_joint', parent, 'wrist', [0, 0, last_link_length + 0.025, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('wrist', 'box', [0.05, 0.25, 0.05], material='purple', color=[0.5, 0, 0.5, 1]))
        self.robot.append(self.create_joint('left_finger_joint', 'wrist', 'left_finger', [0, -0.1, 0.075, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('left_finger', 'box', [0.05, 0.05, 0.15], material='purple', color=[0.5, 0, 0.5, 1]))
        self.robot.append(self.create_joint('right_finger_joint', 'wrist', 'right_finger', [0, 0.1, 0.075, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('right_finger', 'box', [0.05, 0.05, 0.15], material='purple', color=[0.5, 0, 0.5, 1]))

    def add_end_effector_link(self, parent, last_link_length):
        """
        Adds an end effector link to the robot model.

        This method appends a fixed joint and a link representing the end effector
        to the robot's URDF model. The joint connects the specified parent link to
        the end effector link.

        Args:
            parent (str): The name of the parent link to which the end effector will be attached.
            last_link_length (float): The length of the last link in the chain before the end effector.

        Returns:
            None
        """
        # Create a probe end effector
        self.robot.append(self.create_joint('probe_joint', parent, 'probe_link', [0, 0, last_link_length, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('probe_link', link_len=0.1, link_width=0.01, mass=0, collision=False))

        self.robot.append(self.create_joint('end_effector_joint', 'probe_link', 'end_effector', [0, 0, 0.1, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('end_effector', link_len=0, link_width=0, mass=0, collision=False))
    
    def create_manipulator(self, axes, joint_types, link_lens, joint_lims, link_shape='cylinder', collision=False, gripper=False):
        """
        Creates a manipulator robot model with specified parameters.

        Args:
            axes (list): A list of axes for each joint in the manipulator.
            joint_types (list): A list of joint types for each joint in the manipulator.
            link_lens (list): A list of lengths for each link in the manipulator.
            joint_lims (list): A list of joint limits for each joint in the manipulator.
            link_shape (str, optional): The shape of the links. Default is 'cylinder'.
            collision (bool, optional): Whether to enable collision for the links. Default is False.
            gripper (bool, optional): Whether to add a custom gripper at the end of the manipulator. Default is False.

        Returns:
            None
        """
        # TODO: Need to verify collision mapping. Note -> validate through doing a collision check with pybullet

        axes = [self.map_axis_input(axis) for axis in axes]
        joint_types = [self.map_joint_type(joint_type) for joint_type in joint_types]
        colors = {name: mcolors.to_rgba(color) for name, color in random.sample(list(mcolors.CSS4_COLORS.items()), len(mcolors.CSS4_COLORS))}
        
        base_link_name = 'base_link'
        base_link_size = 0.25
        self.robot.append(
            self.create_link(name=base_link_name, 
                             type='box', 
                             link_len=base_link_size,
                             link_width=base_link_size,
                             link_height=base_link_size,
                             mass=1,
                             material='gray', 
                             color=[0.5, 0.5, 0.5, 1], 
                             collision=False))
        
        parent_name = base_link_name
        parent_length = base_link_size / 2
        
        for i, (axis, child_length, joint_type, joint_limit) in enumerate(zip(axes, link_lens, joint_types, joint_lims)):
            color_name, color_code = list(colors.items())[i]
            joint_name = f'joint{i}'
            child_name = f'link{i}'

            if joint_type == 'prismatic':
                parent_length += 0.005
                link_width = 0.05
                # collision = False
            else:
                link_width = 0.05

            if joint_type == 'revolute':
                # Create a visual representation of the joint
                if axis == '1 0 0':
                    joint_viz_rot = [0, 1.57, 0]
                elif axis == '0 1 0':
                    joint_viz_rot = [1.57, 0, 0]
                else:
                    joint_viz_rot = [0, 0, 0]
                joint_viz_origin = [0, 0, 0]
                joint_viz_origin.extend(joint_viz_rot)

                ball_joint_name = f'joint_viz{i}'
                ball_joint_link_name = f'joint_viz_link{i}'
                self.robot.append(
                    self.create_joint(ball_joint_name, 
                                    parent=parent_name, 
                                    child=ball_joint_link_name, 
                                    joint_type='fixed', 
                                    origin=[0, 0, parent_length, 0, 0, 0]))
                self.robot.append(
                    self.create_link(ball_joint_link_name, 
                                    type='cylinder', 
                                    link_len=0.125,
                                    link_width=0.06,
                                    mass=0,
                                    origin=joint_viz_origin, 
                                    material=color_name, 
                                    color=color_code, 
                                    collision=False))
            
            # Create the official joint and link
            self.robot.append(
                self.create_joint(joint_name, 
                                  parent=parent_name, 
                                  child=child_name, 
                                  joint_type=joint_type, 
                                  axis=axis,
                                  limit=joint_limit, 
                                  origin=[0, 0, parent_length, 0, 0, 0]))
            self.robot.append(
                self.create_link(child_name, 
                                 type=link_shape, 
                                 link_len=child_length,
                                 link_width=link_width,
                                 origin=[0, 0, child_length / 2, 0, 0, 0], 
                                 material=color_name, 
                                 color=color_code, 
                                 collision=collision))
            
            parent_name = child_name
            parent_length = child_length
        
        if gripper:
            self.add_custom_gripper(parent=parent_name, last_link_length=parent_length)
        else:
            self.add_end_effector_link(parent=parent_name, last_link_length=parent_length)
    
    @staticmethod
    def _shape_attributes(shape_type, shape_dim):
        """
        Generate a dictionary of attributes for a given shape and its dimensions.

        Args:
            shape (str): The type of shape ('cylinder', 'box', or 'sphere').
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
            return {'size': f"{shape_dim[0]} {shape_dim[1]} {shape_dim[2]}"}
        elif shape_type == 'sphere':
            return {'radius': str(shape_dim[0])}
        return {}
    
    @staticmethod
    def map_axis_input(input_axis):
        """
        Maps input axis 'x', 'y', 'z' to '1 0 0', '0 1 0', '0 0 1', respectively.

        Args:
            input_axis (str): The input axis ('x', 'y', 'z').

        Returns:
            str: The corresponding axis in the format 'x y z'.
        """
        axis_mapping = {
            'x': '1 0 0',
            'y': '0 1 0',
            'z': '0 0 1'
        }
        return axis_mapping.get(input_axis, '0 0 0')
    
    @staticmethod
    def map_joint_type(input_joint_type):
        """
        Maps an input joint type identifier to its corresponding joint type name.

        Args:
            input_joint_type (int): The identifier for the joint type. 
                                    Expected values are:
                                    0 - 'prismatic'
                                    1 - 'revolute'
                                    2 - 'spherical'

        Returns:
            str: The name of the joint type corresponding to the input identifier.
                 Defaults to 'revolute' if the input identifier is not recognized.
        """
        joint_type_mapping = {
            0: 'prismatic',
            1: 'revolute',
            2: 'spherical',
            3: 'fixed'
        }
        return joint_type_mapping.get(input_joint_type, 'revolute')
    
    @staticmethod
    def inertial_calculation(mass, link_length, link_width, type='cylinder'):
        """
        Calculate the inertia matrix for a given link based on its mass and dimensions.

        Args:
            mass (float): The mass of the link.
            link_length (float): The length of the link.
            link_width (float): The width (or radius) of the link.
            type (str, optional): The type of the link ('cylinder', 'box'). Default is 'cylinder'.

        Returns:
            dict: A dictionary representing the inertia matrix with keys 'ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'.
        """
        if type == 'cylinder':
            ixx = (1/12) * mass * (3 * link_width**2 + link_length**2)
            iyy = (1/12) * mass * (3 * link_width**2 + link_length**2)
            izz = 0.5 * mass * link_width**2
        elif type == 'box':
            ixx = (1/12) * mass * (link_length**2 + link_width**2)
            iyy = (1/12) * mass * (link_length**2 + link_width**2)
            izz = (1/12) * mass * (link_length**2 + link_width**2)
        elif type == 'sphere':
            ixx = (2/5) * mass * link_width**2
            iyy = (2/5) * mass * link_width**2
            izz = (2/5) * mass * link_width**2
        else:
            raise ValueError("Unsupported shape type for inertia calculation.")

        return {
            'ixx': str(ixx), 'iyy': str(iyy), 'izz': str(izz),
            'ixy': '0.0', 'ixz': '0.0', 'iyz': '0.0'  # Explicitly set to string format
        }


if __name__ == '__main__':
    robot_name = 'test_robot'
    urdf_gen = URDFGen(robot_name)

    # axes = ['y', 'x', 'y', 'x', 'x']
    # joint_types = [1, 1, 1, 1, 1]
    # link_lens = [0.5, 0.5, 0.5, 0.5, 0.5]

    axes = ['y', 'x', 'z', 'y', 'x', 'y']
    joint_types = [1, 1, 1, 1, 1, 1]
    link_lens = [0.5, 0.5, 0.6, 0.4, 0.3, 0.2]

    joint_limit_prismatic = (-0.5, 0.5)
    joint_limit_revolute = (-3.14, 3.14)

    # Map joint limits based on joint type
    joint_limits = [joint_limit_prismatic if jt == 0 else joint_limit_revolute for jt in joint_types]
                   
    urdf_gen.create_manipulator(axes, joint_types, link_lens, link_shape='cylinder', joint_lims=joint_limits)
    urdf_gen.save_urdf(robot_name)