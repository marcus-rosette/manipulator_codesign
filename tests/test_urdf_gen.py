import unittest
import numpy as np
from manipulator_codesign.urdf_gen import URDFGen

class TestURDFGen(unittest.TestCase):
    def setUp(self): # unittest framework looks specifically for setUp() versus setup() or set_up()
        """Initialize a URDFGen instance before each test."""
        self.urdf_gen = URDFGen("test_robot")

    def test_create_link(self):
        """Test if create_link generates the expected XML structure."""
        link_name = "test_link"
        link = self.urdf_gen.create_link(name=link_name)

        self.assertEqual(link.tag, "link")
        self.assertEqual(link.attrib["name"], link_name)

        # Check for required sub-elements
        inertial = link.find("inertial")
        visual = link.find("visual")
        collision = link.find("collision")

        self.assertIsNotNone(inertial, "Inertial element missing")
        self.assertIsNotNone(visual, "Visual element missing")
        self.assertIsNotNone(collision, "Collision element missing")

        # Check mass value
        mass_element = inertial.find("mass")
        self.assertIsNotNone(mass_element, "Mass element missing")
        self.assertEqual(mass_element.attrib["value"], "1.0")

        # Check geometry type
        geometry = visual.find("geometry")
        self.assertIsNotNone(geometry, "Geometry element missing")
        cylinder = geometry.find("cylinder")
        self.assertIsNotNone(cylinder, "Expected cylinder geometry missing")

        # Check material and color
        material = visual.find("material")
        self.assertIsNotNone(material, "Material element missing")
        color = material.find("color")
        self.assertIsNotNone(color, "Color element missing")
        self.assertEqual(color.attrib["rgba"], "0.5 0.5 0.5 1")

    def test_create_joint(self):
        """Test if create_joint generates the expected XML structure."""
        joint_name = "test_joint"
        parent_name = "parent_link"
        child_name = "child_link"
        joint = self.urdf_gen.create_joint(name=joint_name, parent=parent_name, child=child_name)

        self.assertEqual(joint.tag, "joint")
        self.assertEqual(joint.attrib["name"], joint_name)
        self.assertEqual(joint.attrib["type"], "revolute")

        # Check parent and child elements
        parent = joint.find("parent")
        child = joint.find("child")
        self.assertIsNotNone(parent, "Parent element missing")
        self.assertIsNotNone(child, "Child element missing")
        self.assertEqual(parent.attrib["link"], parent_name)
        self.assertEqual(child.attrib["link"], child_name)

        # Check axis element
        axis = joint.find("axis")
        self.assertIsNotNone(axis, "Axis element missing")
        self.assertEqual(axis.attrib["xyz"], "0 0 1")

        # Check limit element
        limit = joint.find("limit")
        self.assertIsNotNone(limit, "Limit element missing")
        self.assertEqual(limit.attrib["lower"], "-1.57")
        self.assertEqual(limit.attrib["upper"], "1.57")
        self.assertEqual(limit.attrib["effort"], "10")
        self.assertEqual(limit.attrib["velocity"], "1")

    def test_shape_attributes_cylinder(self):
        shape_type = "cylinder"
        shape_dimensions = np.random.rand(2).tolist() # length, radius
        expected_output = {"length": str(shape_dimensions[0]), "radius": str(shape_dimensions[1])}
        result = URDFGen._shape_attributes(shape_type, shape_dimensions)
        self.assertEqual(result, expected_output)
    
    def test_shape_attributes_box(self):
        shape_type = "box"
        shape_dimensions = np.random.rand(3).tolist() # length, width, height
        expected_output = {"size": f'{shape_dimensions[0]} {shape_dimensions[1]} {shape_dimensions[2]}'}
        result = URDFGen._shape_attributes(shape_type, shape_dimensions)
        self.assertEqual(result, expected_output)
    
    def test_shape_attributes_sphere(self):
        shape_type = "sphere"
        shape_dimensions = np.random.rand(1).tolist() # radius
        expected_output = {"radius": str(shape_dimensions[0])}
        result = URDFGen._shape_attributes(shape_type, shape_dimensions)
        self.assertEqual(result, expected_output)
    
    def test_map_axis_input(self):
        self.assertEqual(URDFGen.map_axis_input('x'), '1 0 0')
        self.assertEqual(URDFGen.map_axis_input('y'), '0 1 0')
        self.assertEqual(URDFGen.map_axis_input('z'), '0 0 1')
        self.assertEqual(URDFGen.map_axis_input('invalid'), '0 0 0')  # Testing an unexpected input

    def test_map_joint_type(self):
        self.assertEqual(URDFGen.map_joint_type(0), 'prismatic')
        self.assertEqual(URDFGen.map_joint_type(1), 'revolute')
        self.assertEqual(URDFGen.map_joint_type(2), 'spherical')
        self.assertEqual(URDFGen.map_joint_type(3), 'fixed') 
        self.assertEqual(URDFGen.map_joint_type('invalid'), 'revolute')  # Testing an unexpected input


if __name__ == '__main__':
    unittest.main()
