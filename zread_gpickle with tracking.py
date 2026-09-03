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


def read_network(path, t0, tf, target_positions=None):
    center = np.zeros(2)
    tracked_cells_data = {}  # Dictionary to store data for multiple tracked cells
    initial_positions = {}   # Dictionary to store initial positions for legend

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
    
    # Select cells to track based on provided positions or default to center
    temp = path + "network" + str(t0) + ".0.gpickle"
    with open(temp, 'rb') as f:
        initial_network = pickle.load(f, encoding='latin1')
        if target_positions:
            tracked_cells = select_cells_by_position(initial_network, target_positions)
        else:
            tracked_cells = [select_cell_to_track(initial_network)]
            
        if tracked_cells:
            print(f"Tracking {len(tracked_cells)} cells")
            for i, cell in enumerate(tracked_cells):
                print(f"Cell {i+1}: position {cell.location}")
                tracked_cells_data[i] = []  # Initialize empty list for each cell
                
                # Store initial position for legend
                x, y, _ = cell.location
                initial_positions[i] = (x, y)
        else:
            print("No cells selected for tracking")
            return
    
    # draw image for each network and track the cells
    for n in range(t0, tf+1):
        temp = path + "network" + str(n) + ".0.gpickle"
        try:
            with open(temp, 'rb') as f:
                network = pickle.load(f, encoding='latin1')
        except FileNotFoundError:
            print('No file found!')
            break
        
        # Track all selected cells in this network
        next_tracked_cells = []
        for i, cell in enumerate(tracked_cells):
            if cell is None:  # Skip if cell was lost in previous iterations
                next_tracked_cells.append(None)
                continue
                
            matched_cell = find_cell_in_network_by_id(network, cell)
            if matched_cell:
                camp_conc = matched_cell.C.get('cAMP', 0)
                h2o2_conc = matched_cell.C.get('H2O2i', 0)
                cai_conc = matched_cell.C.get('Ca', 0)
                tracked_cells_data[i].append((n, h2o2_conc, cai_conc, matched_cell.state))
                print(f"Time {n}, Cell {i+1}: cAMP = {camp_conc}, H2O2i = {h2o2_conc}, iCa2+ = {cai_conc}, State = {matched_cell.state}")
                # Update tracked_cell to match the current position for better tracking in next frame
                next_tracked_cells.append(matched_cell)
            else:
                print(f"Time {n}, Cell {i+1}: Not found")
                next_tracked_cells.append(None)  # Mark as lost
        
        # Update tracked_cells list
        tracked_cells = next_tracked_cells
        
        for species in available_species:
            # Draw the image for each species
            draw_species_image(network, path, n, center, radius, species)
        draw_state_image(network, path, n, center, radius)
        
        print(f"Time {n}")
    
    # Plot the H2O2 concentration for all tracked cells
    if any(tracked_cells_data.values()):
        plot_tracked_cells_data(tracked_cells_data, path, initial_positions)

def select_cell_to_track(network):
    """
    Select a cell to track. This implementation selects the cell closest to the center.
    You can modify this to select a specific cell by location, state, or other criteria.
    """
    nodes = list(network.nodes())
    if not nodes:
        return None
    
    # Get the network center
    center = network_avg(network)
    
    # Find the cell closest to the center
    closest_cell = nodes[0]
    min_distance = float('inf')
    
    for node in nodes:
        x, y, _ = node.location
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_cell = node
    
    return closest_cell

def select_cells_by_index(network, indices):
    """
    Select cells by their index in the network node list.
    """
    nodes = list(network.nodes())
    selected_cells = []
    for idx in indices:
        if 0 <= idx < len(nodes):
            selected_cells.append(nodes[idx])
            print(f"Selected cell #{idx} at {nodes[idx].location}")
        else:
            print(f"Cell index {idx} out of range.")
    return selected_cells


def find_cell_in_network(network, target_cell):
    """
    Find the cell in the current network that matches the target cell.
    Using proximity-based matching assuming cells don't move too much between frames.
    """
    nodes = list(network.nodes())
    if not nodes:
        return None
    
    tx, ty, _ = target_cell.location
    closest_cell = None
    min_distance = float('inf')
    
    for node in nodes:
        x, y, _ = node.location
        distance = np.sqrt((x - tx)**2 + (y - ty)**2)
        if distance < min_distance:
            min_distance = distance
            closest_cell = node
    
    # Only return if the cell is close enough (prevent wrong matching)
    if min_distance < target_cell.radius * 2:  # Threshold can be adjusted
        return closest_cell
    return None

def find_cell_in_network_by_id(network, target_cell):
    for node in network.nodes():
        if hasattr(node, 'ID') and node.ID == target_cell.ID:
            return node
    return None

def select_cells_by_position(network, target_positions):
    """
    Select cells closest to the specified positions.
    
    Parameters:
    - network: The network to select cells from
    - target_positions: List of (x, y) coordinates to find cells near
    
    Returns:
    - List of selected cells
    """
    nodes = list(network.nodes())
    if not nodes or not target_positions:
        return []
    
    selected_cells = []
    
    for pos in target_positions:
        target_x, target_y = pos
        closest_cell = None
        min_distance = float('inf')
        
        for node in nodes:
            x, y, _ = node.location
            distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_cell = node
        
        if closest_cell:
            selected_cells.append(closest_cell)
            print(f"Selected cell {closest_cell.ID} at {closest_cell.location} (closest to target {pos})")
    
    return selected_cells

def plot_tracked_cells_data(tracked_cells_data, path, initial_positions=None):
    """
    Plot the H2O2 and iCa2+ concentration for all tracked cells over time on the same graph.
    Include cell positions in the legend.
    """
    plt.figure(figsize=(12, 8))
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(tracked_cells_data)))

    # Plot H2O2i for each cell
    for i, (cell_id, data) in enumerate(tracked_cells_data.items()):
        if not data:
            continue
        times, h2o2, cai, states = zip(*data)
        if initial_positions and cell_id in initial_positions:
            x, y = initial_positions[cell_id]
            label = f'Cell {cell_id+1} (x={x:.2f}, y={y:.2f})'
        else:
            label = f'Cell {cell_id+1}'
        plt.plot(times, h2o2, marker='o', color=colors[i], linewidth=2, label=label + ' H2O2i')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('H2O2i Concentration', fontsize=14)
    plt.yscale('log')
    plt.title('Tracked Cells H2O2i Concentration Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_dir = os.path.join(path, "all_cell_tracking")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, "all_tracked_cells_h2o2.png"), dpi=300)
    plt.close()

    # Plot iCa2+ for each cell
    plt.figure(figsize=(12, 8))
    for i, (cell_id, data) in enumerate(tracked_cells_data.items()):
        if not data:
            continue
        times, h2o2, cai, states = zip(*data)
        if initial_positions and cell_id in initial_positions:
            x, y = initial_positions[cell_id]
            label = f'Cell {cell_id+1} (x={x:.2f}, y={y:.2f}, ID={indices})'
        else:
            label = f'Cell {cell_id+1}'
        plt.plot(times, cai, marker='o', color=colors[i], linewidth=2, label=label + ' iCa2+')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('iCa2+ Concentration', fontsize=14)
    plt.yscale('log')
    plt.ylim(9e-8, 5e-5)
    plt.title('Tracked Cells iCa2+ Concentration Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "all_tracked_cells_ica.png"), dpi=300)
    plt.close()

    # Save the data as CSV with both H2O2i and iCa2+ for all cells
    import csv
    with open(os.path.join(plot_dir, "all_tracked_cells_data.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Time']
        for cell_id in tracked_cells_data.keys():
            if initial_positions and cell_id in initial_positions:
                x, y = initial_positions[cell_id]
                header += [f'Cell {cell_id+1} (x={x:.2f}, y={y:.2f}) H2O2i', f'Cell {cell_id+1} iCa2+', f'Cell {cell_id+1} State']
            else:
                header += [f'Cell {cell_id+1} H2O2i', f'Cell {cell_id+1} iCa2+', f'Cell {cell_id+1} State']
        writer.writerow(header)
        all_times = set()
        for data in tracked_cells_data.values():
            for entry in data:
                all_times.add(entry[0])
        all_times = sorted(all_times)
        for t in all_times:
            row = [t]
            for cell_id in tracked_cells_data.keys():
                data = tracked_cells_data.get(cell_id, [])
                entry = next((item for item in data if item[0] == t), None)
                if entry:
                    row += [entry[1], entry[2], entry[3]]
                else:
                    row += ['', '', '']
            writer.writerow(row)
    print(f"All tracked cells data saved to {plot_dir}")

    # State plot (unchanged)
    state_map = {'U': 0, 'T': 1, 'D': 2, 'A': 3}
    plt.figure(figsize=(12, 8))
    for i, (cell_id, data) in enumerate(tracked_cells_data.items()):
        if not data:
            continue
        times, _, _, states = zip(*data)
        state_values = [state_map.get(s, -1) for s in states]
        if initial_positions and cell_id in initial_positions:
            x, y = initial_positions[cell_id]
            label = f'Cell {cell_id+1} (x={x:.2f}, y={y:.2f})'
        else:
            label = f'Cell {cell_id+1}'
        plt.plot(times, state_values, marker='s', color=colors[i], linewidth=2, label=label)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Cell State', fontsize=14)
    plt.yticks(list(state_map.values()), list(state_map.keys()))
    plt.title('Tracked Cells State Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "all_tracked_cells_state.png"), dpi=300)
    plt.close()

def draw_species_image(network, path, ind, center, radius, species, scale=1, background=(0,0,0), origin_bottom=True, bound_rad=30, image_name='image'):
    outline_dict={'U': (0, 0, 0),'T': (0, 0, 0),'D': (0, 0, 0),'A':(0,0,0)}

    image_size = int(2*radius)
    image_C = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    background = (background[2], background[1], background[0])
    image_C[:, :] = background

    img_center = image_size // 2
    nodes=network.nodes()

    # Step 1: Define the color map
    if species == 'cAMP':
        colormap = plt.cm.viridis
        normalizer = Normalize(vmin=0.4, vmax=1)
    elif species == 'GSH':
        colormap = plt.cm.jet#viridis  # Choose a color map, e.g., viridis
        normalizer = Normalize(vmin=0.00040, vmax=0.0015)  # Replace with the range you expect
    elif species == 'H2O2i':
        colormap = plt.cm.viridis
        normalizer = Normalize(vmin=0, vmax=1e-7)  # Replace with the range you expect
    else:
        colormap = plt.cm.plasma
        normalizer = Normalize(vmin=9e-8, vmax=5e-5)
    
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
    
    image_size = int(2*radius)
    image_state = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    background = (background[2], background[1], background[0])
    image_state[:, :] = background
    img_center = image_size // 2
    nodes=network.nodes()
    
    # go through all of the agents
    for node in nodes:
        # get xy coordinates, the axis lengths, and color of agent
        x,y,z = node.location
        r=node.radius
        col_state=col_dict[node.state]
        out=outline_dict[node.state]
        x_p = int(scale * (x - center[0] + img_center))
        y_p = int(scale * (y - center[1] + img_center))
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

if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/8.0/'

    # First, display info about the initial network to help with cell selection
    temp = path + "network0.0.gpickle"
    try:
        with open(temp, 'rb') as f:
            initial_network = pickle.load(f, encoding='latin1')
            center = network_avg(initial_network)
            radius = network_radius(initial_network)
            print(f"Network center: {center}")
            print(f"Network radius: {radius}")
    except FileNotFoundError:
        print("Initial network file not found. Using default values.")
        center = (0, 0)
        radius = 100
    
    # Ask if user wants to specify positions
    user_input = input("Do you want to specify cell positions to track? (y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\nYou can specify cell positions as x,y coordinates or cell numbers (index).")
        print(f"Network center is approximately at {center}")
        print(f"Network radius is approximately {radius}")
        print("Enter positions as x,y or cell numbers as #n (e.g., #5), one at a time. Press Enter with no input when done.")
        
        positions = []
        indices = []
        while True:
            pos_input = input("Enter cell position (x,y) or cell number (#n), or press Enter to finish: ").strip()
            if not pos_input:
                break
                
            if pos_input.startswith('#'):
                try:
                    idx = int(pos_input[1:])
                    indices.append(idx)
                    print(f"Added cell number: {idx}")
                except ValueError:
                    print("Invalid cell number. Use #n (e.g., #5)")
            else:
                try:
                    if ',' in pos_input:
                        x, y = map(float, pos_input.split(','))
                        positions.append((x, y))
                        print(f"Added position: ({x}, {y})")
                    else:
                        print("Invalid format. Use x,y or #n")
                except ValueError:
                    print("Invalid coordinates. Use numeric values for x and y.")
        
        if positions or indices:
            with open(temp, 'rb') as f:
                initial_network = pickle.load(f, encoding='latin1')
                selected_cells = []
                if positions:
                    selected_cells += select_cells_by_position(initial_network, positions)
                if indices:
                    selected_cells += select_cells_by_index(initial_network, indices)
                if selected_cells:
                    read_network(path, 0, 600, target_positions=[cell.location[0:2] for cell in selected_cells])
                else:
                    print("No valid cells specified. Tracking cell at center.")
                    read_network(path, 0, 600)
        else:
            print("No positions specified. Tracking cell at center.")
            read_network(path, 0, 600)
    else:
        print("Tracking cell closest to center.")
        read_network(path, 0, 600)

        
        
