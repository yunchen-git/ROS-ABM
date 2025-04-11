# This script makes a movie from an exsisting simulation
import numpy as np
import os, sys
# Old simulations require networkx=version1.11
# Can be installed in a separate venv using the command conda install networkx=version1.11
import networkx as nx
import cv2
import pickle
import json 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def read_network(path, t0, tf):
    center = np.zeros(2)

    # get center and radius of the last available network
    last_available_n = None
    for n in range(tf, t0-1, -1):  # Check from tf down to t0
        temp = path + "network" + str(n) + ".0.gpickle"
        if os.path.exists(temp):
            last_available_n = n
            break
    
    last_pic_path = path + "network" + str(last_available_n) + ".0.gpickle"
    with open(last_pic_path, 'rb') as f:
        network = pickle.load(f, encoding='latin1')
        center = network_avg(network)
        radius = network_radius(network)
        available_species = get_available_species(network)
        print(f"Center of network: {center}")
        print(f"Radius of network: {radius}")
        print(f"Available species in network: {available_species}")
    
    # draw image for each network
    for n in range(t0, tf+1):
        temp = path + "network" + str(n) + ".0.gpickle"
        try:
            with open(temp, 'rb') as f:
                network = pickle.load(f, encoding='latin1')
        except FileNotFoundError:
            print('No file found!')
            return
        
        for species in available_species:
            # Draw the image for each species
            draw_species_image(network, path, n, center, radius, species)
        draw_state_image(network, path, n, center, radius)
        
        print(f"Time {n}")

def draw_species_image(network, path, ind, center, radius, species, scale=1, background=(0,0,0), origin_bottom=True, bound_rad=30, image_name='image'):
    outline_dict={'U': (0, 0, 0),'T': (0, 0, 0),'D': (0, 0, 0),'A':(0,0,0)}

    image_size = int(2*radius)
    image_C = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    background = (background[2], background[1], background[0])
    image_C[:, :] = background

    img_center = image_size // 2
    nodes=network.nodes()

    # Step 1: Define the color map
    if species == 'GSH':
        colormap = plt.cm.jet#viridis  # Choose a color map, e.g., viridis
        normalizer = Normalize(vmin=0.00040, vmax=0.0015)  # Replace with the range you expect
    elif species == 'H2O2i':
        colormap = plt.cm.viridis
        normalizer = Normalize(vmin=0, vmax=5e-9)  # Replace with the range you expect
    elif species == 'cAMP':
        colormap = plt.cm.viridis
        normalizer = Normalize(vmin=0, vmax=1)

    def get_color(C_value):
        normalized_value = normalizer(C_value)  # Map C_value directly within -100 to 100 range
        return colormap(normalized_value)
    
    # go through all of the agents
    for node in nodes:
        # get xy coordinates, the axis lengths, and color of agent
        x,y,z = node.location
        r=node.radius

        # Get color based on fixed range for node.C
        C_value = node.C[species]  # Assuming node.C is a list with one value
        node_color_rgba = get_color(C_value)
        
        # Convert RGBA to BGR for OpenCV
        col_C = tuple(int(255 * c) for c in reversed(node_color_rgba[:3]))

        out=outline_dict[node.state]
        
        # Position nodes relative to the center of the image
        x_p = int(scale * (x - center[0] + img_center))
        y_p = int(scale * (y - center[1] + img_center))
        major, minor = int(scale * r), int(scale * r)

        # Draw the agent and a black outline to distinguish overlapping agents
        image_C = cv2.ellipse(image_C, (x_p, y_p), (major, minor), 0, 0, 360, col_C, -1)
        image_C = cv2.ellipse(image_C, (x_p, y_p), (major, minor), 0, 0, 360, out, 1)

    if origin_bottom:
        image_C = cv2.flip(image_C, 0)

    # save the image as a PNG
    folder_C = "all_images" + "_" + species
    full_path_C = os.path.join(path, folder_C)
   
    if not os.path.exists(full_path_C):
        os.makedirs(full_path_C)

    # Add a color bar using Matplotlib
    plt.imshow(cv2.cvtColor(image_C, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title = species + " Concentration"
    plt.title(title)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=colormap), ax=plt.gca(), location='left', pad=0.01, shrink=1.0)
    cbar.set_label('Concentration')
    plt.savefig(os.path.join(full_path_C, f"{image_name}_with_colorbar_{format(ind, '02')}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def draw_state_image(network, path, ind, center, radius, scale=1, background=(0,0,0), origin_bottom=True, bound_rad=30, image_name='image'):
    col_dict={'U': (50, 255, 50),'T': (255, 50, 50),'D': (50, 50, 255),'A':(200,200,200)}
    outline_dict={'U': (0, 0, 0),'T': (0, 0, 0),'D': (0, 0, 0),'A':(0,0,0)}

    image_state = np.zeros((int(center[0]+2*radius), int(center[1]+2*radius), 3), dtype=np.uint8)

    background = (background[2], background[1], background[0])
    image_state[:, :] = background

    nodes=network.nodes()
    
    # go through all of the agents
    for node in nodes:
        # get xy coordinates, the axis lengths, and color of agent
        x,y,z = node.location
        r=node.radius
        col_state=col_dict[node.state]
        out=outline_dict[node.state]
        x_p, y_p = int(scale * (x+radius)), int(scale * (y+radius))
        major, minor = int(scale * r), int(scale * r)

        # Draw the agent and a black outline to distinguish overlapping agents
        image_state = cv2.ellipse(image_state, (x_p, y_p), (major, minor), 0, 0, 360, col_state, -1)
        image_state = cv2.ellipse(image_state, (x_p, y_p), (major, minor), 0, 0, 360, out, 1)

    if origin_bottom:
        image_state = cv2.flip(image_state, 0)

    # save the image as a PNG
    folder_state = "all_images_state"
    full_path_state = os.path.join(path, folder_state)
    image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)

    if not os.path.exists(full_path_state):
        os.makedirs(full_path_state)

    file_name_state = f"{image_name}_state{format(ind,'02')}.png"
    cv2.imwrite(os.path.join(full_path_state, file_name_state), image_state, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])


def network_avg(network):
    center = np.zeros(2)
    for node in network.nodes():
        center += node.location[0:2]
    #Normalize
    center = center / len(network.nodes())

    return center

def network_radius(network):
    #first get the center
    center = network_avg(network)
    #gets the maximum radial size of the network
    radius = -1
    for node in network.nodes():
        r = node.location[0:2] - center
        radius = max(np.linalg.norm(r), radius)

    return radius

def get_available_species(network):
    nodes = network.nodes()
    if not nodes:
        print("Warning: Network has no nodes")
        return []
    
    # Get the first node to determine available species
    first_node = list(nodes)[0]
    available_species = list(first_node.C.keys())
    return available_species

def make_txt(name, path):
    state_dict={"U":0,"T":1,"D":2,0:"U",1:"T",2:"D"}
    with open(f'{name}.gpickle', 'rb') as f:
    # network = nx.Graph()
    # #print(f.readline().decode(encoding='utf-32'))
    # old_network = pickle.load(f, encoding="latin1")
    # network.add_nodes_from(old_network.nodes(data=True))
        network = pickle.load(f, encoding='latin1')
    nodes = network.nodes()
    center = network_avg(network)
    radius = network_radius(network)    
    with open(f'{path}/{name}.txt', 'w') as f:
        i = 1
        for node in nodes:
            x,y,z = node.location
            f.write(f'{i},{x},{y},{state_dict[node.state]}\n')
    f.close()
    print(f'Saved network with {len(nodes)} cells as {name}')
    draw_cell_image(network, f'{path}/', 0, center, radius, image_name=name)
    print(f'Reference image is: {name}_{format(0,"02")}.png')

if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/12.0/'
    
    read_network(path, 0, 75)


    """ 
    I think below is for saving as txt instead of pic, not used as of 09apr2025
    """
    #path = f'/Users/andrew/PycharmProjects/pythonABM_CG/outputs/{sys.argv[1]}.0/'
    # if sys.argv[1] == 'png':
    #     path = f'/Users/huyunchen/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Melissa Kemp Lab/Models/Glen CM Nat Comm 2018/mESC-Intercellular-Differentiation-master/Tests/10.0/{sys.argv[2]}.0/'
    #     t0 = input('initial step: ')
    #     tf = input('end step: ')
    #     read_network(path, int(t0), int(tf))
    # elif sys.argv[1] == 'txt':
    #     network_name = input('Reference network name: ')
    #     if len(sys.argv) > 2:
    #         path = sys.argv[2]
    #     else:
    #         path = os.getcwd()
    #     make_txt(network_name, path)
        

        
        
