from urdfpy import URDF, Visual, Collision, Material, Box, Cylinder, Sphere, Joint, Link

# Load the URDF file
urdf = URDF.load('./urdf/rrr_manipulator.urdf')

# Print original links and joints
print("Original Links and Joints:")
for link in urdf.links:
    # Access the visuals and their geometry
    if link.visuals:
        for visual in link.visuals:
            geometry = visual.geometry
            print(f"Link: {link.name}, Size: {geometry}")

for joint in urdf.joints:
    print(f"Joint: {joint.name}, Type: {joint.joint_type}, Axis: {joint.axis}")

# Example edits:
# 1. Change the length of link1
link1 = urdf.link_map['link1']
link1.visuals[0].geometry.cylinder.length = 0.6  # New length

# 2. Change the joint type of joint2 from 'revolute' to 'prismatic'
joint2 = urdf.joint_map['joint2']
joint2.joint_type = 'prismatic'
joint2.axis = [0, 1, 0]  # New axis for prismatic joint

# 3. Change the joint axis of joint3 to [0, 1, 0]
joint3 = urdf.joint_map['joint3']
joint3.axis = [0, 1, 0]

# 4. Change the size of the box for the wrist link
wrist = urdf.link_map['wrist']
wrist.visuals[0].geometry.box.size = [0.3, 0.05, 0.05]  # New size

# # 5. Add a new link
# new_link = Link(
#     name='new_link',
#     visuals=[
#         Visual(
#             origin=[0, 0, 0], 
#             geometry=Box(size=[0.1, 0.1, 0.1]),  # Correctly set geometry as Box object
#             material=Material(name='red', color=[1, 0, 0, 1])
#         )
#     ],
#     collisions=[
#         Collision(
#             origin=[0, 0, 0], 
#             geometry=Box(size=[0.1, 0.1, 0.1])  # Correctly set geometry as Box object
#         )
#     ]
# )
# urdf.links.append(new_link)

# # 6. Add a new joint
# new_joint = Joint(
#     name='new_joint',
#     joint_type='revolute',
#     parent='link3',
#     child='new_link',
#     origin=[0, 0, 0.5],
#     axis=[0, 0, 1],
#     limit=[-1.57, 1.57, 5.0, 1.0]  # Example limits
# )
# urdf.joints.append(new_joint)

# Save the edited URDF to a new file
urdf.save('./urdf/edited_3r_manipulator.urdf')

print("Edited URDF saved.")
