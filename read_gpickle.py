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


# Step 1: Define the color map
colormap = plt.cm.viridis  # Choose a color map, e.g., viridis

normalizer = Normalize(vmin=0, vmax=1e-4)  # Replace with the range you expect

def get_color(C_value):
    normalized_value = normalizer(C_value)  # Map C_value directly within -100 to 100 range
    return colormap(normalized_value)

def read_network(path, t0, tf):
    files = os.listdir(path)
    center = np.zeros(2)
    for n in range(t0, tf+1):
        # load it
        cIter = 1
        concentration = False
        temp = path + "network" + str(n) + ".0.gpickle"
        try:
            with open(temp, 'rb') as f:
                # network = nx.Graph()
                # #print(f.readline().decode(encoding='utf-32'))
                # old_network = pickle.load(f, encoding="latin1")
                # network.add_nodes_from(old_network.nodes(data=True))
                network = pickle.load(f, encoding='latin1')
                if n == t0:
                #     print(node.C)
                        center = network_avg(network)
                        radius = network_radius(network)
        except FileNotFoundError:
            print('No file found!')
            return

        draw_cell_image(network, path, n, center, radius)
        # deinfe the bounding box to take images at
        
        print(f"Time {n}")
        print(f"center {center}")
        print(f"radius {radius}")

def draw_cell_image(network, path, ind, center, radius, scale=1, background=(0,0,0), origin_bottom=True, bound_rad=30, image_name='image'):
    col_dict={'U': (50, 255, 50),'T': (255, 50, 50),'D': (50, 50, 255),'A':(200,200,200)}
    outline_dict={'U': (0, 0, 0),'T': (0, 0, 0),'D': (0, 0, 0),'A':(0,0,0)}

    nodes=network.nodes()
    image_C = np.zeros((int(radius)*7, int(radius)*7, 3), dtype=np.uint8)
    image_state = np.zeros((int(radius)*7, int(radius)*7, 3), dtype=np.uint8)
    background = (background[2], background[1], background[0])
    image_C[:, :] = background
    image_state[:, :] = background
    
    # go through all of the agents
    for node in nodes:
        # get xy coordinates, the axis lengths, and color of agent
        x,y,z = node.location
        r=node.radius
        col_state=col_dict[node.state]
        # print(node.C)

        # Get color based on fixed range for node.C
        C_value = node.C['H2O2i']  # Assuming node.C is a list with one value
        node_color_rgba = get_color(C_value)
        
        # Convert RGBA to BGR for OpenCV
        col_C = tuple(int(255 * c) for c in reversed(node_color_rgba[:3]))

        out=outline_dict[node.state]
        x_p, y_p = int(scale * (x+2*radius)), int(scale * (y+2*radius))
        major, minor = int(scale * r), int(scale * r)
        #print(node.C)
         # Draw the agent and a black outline to distinguish overlapping agents
        image_C = cv2.ellipse(image_C, (x_p, y_p), (major, minor), 0, 0, 360, col_C, -1)
        image_C = cv2.ellipse(image_C, (x_p, y_p), (major, minor), 0, 0, 360, out, 1)
        image_state = cv2.ellipse(image_state, (x_p, y_p), (major, minor), 0, 0, 360, col_state, -1)
        image_state = cv2.ellipse(image_state, (x_p, y_p), (major, minor), 0, 0, 360, out, 1)
    # try:
    #     bounds=nodes[0].bounds
    #     i = 0
    #     for bound in bounds:
    #         x,y=bounds[i]
    #         if i < len(bounds)-1:
    #             x1,y1 = bounds[i+1]
    #         else:
    #             x1,y1 = bounds[0]
    #         x1_p, y1_p = int(scale * (x1 + center[0])), int(scale * (y1 + center[1]))
    #         major, minor = int(scale * bound_rad), int(scale * bound_rad)

    #         # draw the agent and a black outline to distinguish overlapping agents
    #         image = cv2.ellipse(image, (x1_p, y1_p), (major, minor), 0, 0, 360, (0, 0, 0), 1)
    #         x0_p, y0_p = int(scale * (x + center[0])), int(scale * (y + center[1]))
    #         image = cv2.line((x0_p, y0_p), (x1_p, y1_p),color=(0,0,0), thickness=2)
    #         i += 1
    # except KeyError:
    #     print('no bounds...')
    # except AttributeError:
        # print('no bounds...')

    if origin_bottom:
        image_C = cv2.flip(image_C, 0)
        image_state = cv2.flip(image_state, 0)
    #print(x, y)
    #print(list(network.nodes()))
    #print(network.nodes())
    # save the image as a PNG
    folder_C = "all_images_C"
    folder_state = "all_images_state"
    full_path_C = os.path.join(path, folder_C)
    full_path_state = os.path.join(path, folder_state)
    image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)

    if not os.path.exists(full_path_C):
        os.makedirs(full_path_C)
    file_name_C = f"{image_name}_C{format(ind,'02')}.png"
    cv2.imwrite(os.path.join(full_path_C, file_name_C), image_C, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])
    
    # Add a color bar using Matplotlib
    plt.imshow(cv2.cvtColor(image_C, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=colormap), ax=plt.gca())
    cbar.set_label('Concentration')
    plt.savefig(os.path.join(full_path_C, f"{image_name}_with_colorbar_{format(ind, '02')}.png"), bbox_inches='tight', dpi=300)
    plt.close()

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
    #now return
    return center

def network_radius(network):
    #first get the center
    center = network_avg(network)
    #gets the maximum radial size of the network
    
    radius = -1
    for node in network.nodes():
        r = node.location[0:2] - center
        radius = max(np.linalg.norm(r), radius)
    #now return

    return radius

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
    path = '/Users/huyunchen/Documents/PhD/test/34.0/'
    
    read_network(path, 0, 75)
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
        

        
        
