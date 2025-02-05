import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_upright_cone_with_dome(cone_height, cone_radius, dome_height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Cone parameters
    cone_height = float(cone_height)
    cone_radius = float(cone_radius)
    
    # Dome parameters
    dome_radius = cone_radius
    
    # Generate cone data
    z_cone = np.linspace(-cone_height, 0, 30)
    theta_cone = np.linspace(0, 2 * np.pi, 30)
    theta_cone, z_cone = np.meshgrid(theta_cone, z_cone)
    x_cone = cone_radius * (1 + z_cone / cone_height) * np.cos(theta_cone)
    y_cone = cone_radius * (1 + z_cone / cone_height) * np.sin(theta_cone)
    
    # Plot cone
    ax.plot_surface(x_cone, y_cone, z_cone, color='orange', alpha=0.6, edgecolor='k')

    # Generate dome data
    u_dome = np.linspace(0, 2 * np.pi, 30)
    v_dome = np.linspace(0, np.pi / 2, 30)
    u_dome, v_dome = np.meshgrid(u_dome, v_dome)
    x_dome = dome_radius * np.sin(v_dome) * np.cos(u_dome)
    y_dome = dome_radius * np.sin(v_dome) * np.sin(u_dome)
    z_dome = dome_height * np.cos(v_dome)
    
    # Plot dome
    ax.plot_surface(x_dome, y_dome, z_dome, color='blue', alpha=0.6, edgecolor='k')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Upright Cone with Dome on Top')

    plt.show()

# Define parameters
cone_height = 5
cone_radius = np.pi/6
dome_height = 2

# Plot the upright cone with the dome
plot_upright_cone_with_dome(cone_height, cone_radius, dome_height)
