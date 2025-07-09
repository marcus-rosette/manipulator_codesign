import numpy as np


def load_plant_env(pyb_con):
    #################### ROW OBJECTS #######################
    row_dim = [2, 0.3, 0.125]

    collision_shape_id = pyb_con.createCollisionShape(
        shapeType=pyb_con.GEOM_BOX,
        halfExtents=row_dim
    )
    visual_shape_id = pyb_con.createVisualShape(
        shapeType=pyb_con.GEOM_BOX,
        halfExtents=row_dim,
        rgbaColor=[1, 1, 1, 1]
    )

    # Create the multi-body
    left_row_id = pyb_con.createMultiBody(
        baseMass=0,  # Static object
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, -(row_dim[1] + row_dim[1]/2), row_dim[2]]  # On top of the plane
    )
    right_row_id = pyb_con.createMultiBody(
        baseMass=0,  # Static object
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, (row_dim[1] + row_dim[1]/2), row_dim[2]]  # On top of the plane
    )

    #################### PLANT OBJECTS #######################
    # Define shared parameters
    box_half_extents = [0.075, 0.075, 0.05]
    cylinder_radius = 0.05
    cylinder_height = 0.3
    cylinder_color = [0.2, 0.6, 0.8, 1]
    box_color = [0.2, 0.8, 0.2, 1]
    box_positions = [
        [0, 0.3, 0.35], [0.55, 0.3, 0.35], [-0.55, 0.3, 0.35],
        [0, 0.6, 0.35], [0.55, 0.6, 0.35], [-0.55, 0.6, 0.35],
        [0, -0.3, 0.35], [0.55, -0.3, 0.35], [-0.55, -0.3, 0.35],
        [0, -0.6, 0.35], [0.55, -0.6, 0.35], [-0.55, -0.6, 0.35]
    ]

    # Create shared shapes
    box_collision_shape_id = pyb_con.createCollisionShape(
        shapeType=pyb_con.GEOM_BOX,
        halfExtents=box_half_extents
    )
    box_visual_shape_id = pyb_con.createVisualShape(
        shapeType=pyb_con.GEOM_BOX,
        halfExtents=box_half_extents,
        rgbaColor=box_color
    )

    # Create boxes and cylinders
    box_ids = []
    cylinder_ids = []
    for pos in box_positions:
        # Create box
        box_id = pyb_con.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_collision_shape_id,
            baseVisualShapeIndex=box_visual_shape_id,
            basePosition=pos
        )
        box_ids.append(box_id)

        # Create cylinder underneath
        cylinder_pos = [pos[0], pos[1], pos[2] - box_half_extents[2] - cylinder_height / 2]
        cylinder_id = pyb_con.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=pyb_con.createCollisionShape(
                shapeType=pyb_con.GEOM_CYLINDER,
                radius=cylinder_radius,
                height=cylinder_height
            ),
            baseVisualShapeIndex=pyb_con.createVisualShape(
                shapeType=pyb_con.GEOM_CYLINDER,
                radius=cylinder_radius,
                length=cylinder_height,
                rgbaColor=cylinder_color
            ),
            basePosition=cylinder_pos
        )
        cylinder_ids.append(cylinder_id)
        
    return [left_row_id, right_row_id] + box_ids + cylinder_ids
