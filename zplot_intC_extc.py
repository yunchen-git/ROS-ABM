import simulationMath1 as simMath
from vpython import *
import random as r
import time as t_m
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import map_coordinates
import matplotlib.colors as mcolors
import math
import os
import Analysis as af
import Simulation as sim
import pickle as p
import cv2
import pandas as pd
from matplotlib.colors import Normalize
from ExtC_gradient import ExtC_Diff

import atexit,gc
@atexit.register
def garbage_collect():
    gc.collect()

def ZeroPad(maxTime, time):
    """ Zero pads the current number
        Returns the current number as an string weith the correct number
        of 0s appended
    """
    #figure out the max number of diogits in the final time
    l = math.log(maxTime, 10)
    if(abs(l - math.ceil(l)) < 0.000001):
        l = math.ceil(l)
    d1 = int(math.floor(l))
    #figure out the number of diogits in the current time
    if(time == 0):
        d2 = 0
    else:
        l = math.log(time, 10)
        if(abs(l - math.ceil(l)) < 0.000001):
            l = math.ceil(l)
        d2 = int(math.floor(l))
    #now append 0's the fron of the current time proportional to the difference
    s = ""
    for i in range(0, d1-d2):
        s = s + "0"
    s = s + repr(time)
    return s

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

def s_for_radius_in_data_units_3d(ax, r_data, at_xyz=None, clip=(1.0, 113.83443722989533)):
    """
    Convert a radius in data units to a scatter size (points^2) on a 3D Axes.
    - ax: 3D axes (projection='3d')
    - r_data: desired radius in data units
    - at_xyz: reference point (x,y,z) where to calibrate the radius (defaults to center of current limits)
    - clip: (min,max) to keep sizes reasonable
    """
    if at_xyz is None:
        at_xyz = (np.mean(ax.get_xlim3d()),
                  np.mean(ax.get_ylim3d()),
                  np.mean(ax.get_zlim3d()))
    x0, y0, z0 = at_xyz

    # Project two points separated by r_data along x into 2D display space
    M = ax.get_proj()
    x2, y2, _ = proj3d.proj_transform(x0, y0, z0, M)
    x3, y3, _ = proj3d.proj_transform(x0 + r_data, y0, z0, M)

    p0 = ax.transData.transform((x2, y2))
    p1 = ax.transData.transform((x3, y3))
    r_pixels = float(np.hypot(*(p1 - p0)))

    # Convert pixel radius -> points -> area (points^2)
    r_points = r_pixels * 72.0 / ax.figure.dpi
    s = np.pi * (r_points ** 2)

    # Keep it sane
    return float(np.clip(s, *clip))


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
        normalizer = Normalize(vmin=0, vmax=1)
    elif species == 'GSH':
        colormap = plt.cm.jet#viridis  # Choose a color map, e.g., viridis
        normalizer = Normalize(vmin=0.00040, vmax=0.0015)  # Replace with the range you expect
    elif species == 'H2O2i':
        colormap = plt.cm.viridis
        normalizer = Normalize(vmin=0, vmax=1e-8)  # Replace with the range you expect
    else:
        colormap = plt.cm.plasma
        normalizer = Normalize(vmin=0, vmax=1e-9)
    
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
    #cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=colormap), ax=plt.gca(), location='left', pad=0.01, shrink=1.0)
    #cbar.set_label('Concentration')
    plt.savefig(os.path.join(full_path_C, f"{image_name}_with_colorbar_{format(ind, '02')}.png"), bbox_inches='tight', dpi=100)
    plt.close()

def save_combined_h2o2_images_3d(path, data, t0, tf, h2o2_gradient_name, plot_2d=True, plot_3d=True, make_graph=True, squished=True):
    """ Combines H2O2 gradient from simulation data with H2O2 from cells in both 2D and 3D """
    
    # Get gradient data
    times = list(data.get_times())
    
    # Create output directories
    output_dir_2d = path + "all_combined_h2o2_2d/"
    output_dir_3d = path + "all_combined_h2o2_3d/"
    
    if plot_2d and not os.path.exists(output_dir_2d):
        os.makedirs(output_dir_2d)
    if plot_3d and not os.path.exists(output_dir_3d):
        os.makedirs(output_dir_3d)
    
    print(f"Processing {len(times)} time points...")
    
    # Get min and max values for consistent color scaling
    mx_gradient = -1
    mn_gradient = float('inf')
    mx_h2o2i = -1
    mn_h2o2i = float('inf')
    all_data = []
    z_slice_info = None
    
    # First pass: collect all data and find min/max
    for time_idx, time in enumerate(times):
        if time < t0 or time > tf:
            continue
        
        try:
            g = data.get_gradient_at_time(time, h2o2_gradient_name)
            
            # Handle 3D gradient data
            if len(g.C.shape) == 3:
                gradient_3d = g.C  # Full 3D gradient
                # Sum over the z-axis to get a 2D projection
                plane_2d = np.sum(g.C, axis=2)
                if z_slice_info is None:
                    z_slice_info = {
                        'total_z_slices': g.C.shape[2],
                        'summed': True,
                        'is_3d': True
                    }
                    print(f"3D Gradient detected and summed over z-axis:")
                    print(f"  Total z-slices: {g.C.shape[2]}")
                    print(f"  Summed shape: {plane_2d.shape}")
                    print(f"  Full 3D shape: {g.C.shape}")
            else:
                gradient_3d = None
                plane_2d = g.C  # Already 2D
                if z_slice_info is None:
                    z_slice_info = {'is_3d': False}
                    print(f"2D Gradient detected: {g.C.shape}")
            
            # Load network data for this time point
            network_file = path + "network" + str(int(time)) + ".0.gpickle"
            if not os.path.exists(network_file):
                print(f"No network file found for time {time}")
                continue
                
            with open(network_file, 'rb') as f:
                network = p.load(f, encoding='latin1')
            
            all_data.append((time, plane_2d, gradient_3d, g, network))
            
            # Update min/max for gradient
            mn_gradient = min(np.min(plane_2d), mn_gradient)
            mx_gradient = max(np.max(plane_2d), mx_gradient)
            
            if gradient_3d is not None:
                mn_gradient = min(np.min(gradient_3d), mn_gradient)
                mx_gradient = max(np.max(gradient_3d), mx_gradient)
            
            # Update min/max for H2O2i
            for node in network.nodes():
                h2o2i_value = node.C.get('H2O2i', 0)
                mn_h2o2i = min(h2o2i_value, mn_h2o2i)
                mx_h2o2i = max(h2o2i_value, mx_h2o2i)
            
        except Exception as e:
            print(f"Error getting gradient at time {time}: {e}")
            continue
    
    print(f"Gradient range: {mn_gradient} to {mx_gradient}")
    print(f"H2O2i range: {mn_h2o2i} to {mx_h2o2i}")
    
    # Set up normalizers
    gradient_normalizer = mcolors.Normalize(vmin=mn_gradient, vmax=mx_gradient)
    h2o2i_normalizer = mcolors.Normalize(vmin=mn_h2o2i, vmax=mx_h2o2i)
    h2o2e_track_589 = []
    h2o2i_track_589 = []
    h2o2e_track_877 = []
    h2o2i_track_877 = []

    
    # Process each time point
    for time, plane_2d, gradient_3d, g, network in all_data:
        print(f"Processing time {time}...")
        
        # Collect cell data
        cell_positions = []
        cell_h2o2i = []
        cell_ids = []
        cell_radii = []
        
        for node in network.nodes():
            x, y, z = node.location[0], node.location[1], node.location[2]
            
            # Convert to grid coordinates
            try:
                if gradient_3d is not None:
                    x_grid, y_grid, z_grid = g.get_object_position_on_grid([x, y, z])
                else:
                    x_grid, y_grid = g.get_object_position_on_grid([x, y])
                    z_grid = 0  # Default for 2D
            except:
                x_grid, y_grid = g.get_object_position_on_grid([x, y])
                z_grid = 0  # Default for 2D
            
            cell_positions.append([x_grid, y_grid, z_grid])
            cell_h2o2i.append(node.C.get('H2O2i', 0))
            cell_ids.append(node.ID)
            cell_radii.append(node.radius)
        
        cell_positions = np.array(cell_positions)
        cell_h2o2i = np.array(cell_h2o2i)
        cell_radii = np.array(cell_radii)

        # Create 2D plot (existing functionality)
        if plot_2d:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot 2D gradient
            im = ax.imshow(plane_2d,
                          cmap='Blues',
                          norm=gradient_normalizer,
                          interpolation='bicubic',
                          origin='lower')
            
            # Add gradient colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, label='Extracellular H2O2 Gradient')
            
            # Plot cells
            ax.figure.canvas.draw()  # Needed to get correct transform

            from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap

            # Matplotlib scatter 's' is area in points^2
            marker_size = 6.5 ** 2 * np.pi

            marker_sizes = np.full_like(cell_radii, marker_size)

            h2o2i_normalizer = mcolors.Normalize(vmin=0, vmax=1e-10)
            # Custom black-to-red colormap (or black-to-yellow if you prefer)
            intracellular_cmap = LinearSegmentedColormap.from_list('black_to_red', ['black', 'yellow', 'orange', 'red', 'darkred', 'purple'])


            scatter = ax.scatter(cell_positions[:, 0], cell_positions[:, 1], 
                                 c=cell_h2o2i, cmap=intracellular_cmap, norm=h2o2i_normalizer,
                                 s=marker_sizes, alpha=0.8, edgecolors='black', linewidths=0.1)

            # Highlight specific cells if needed
            for i, cell_id in enumerate(cell_ids):
                if cell_id in [589, 877]:  # Your cells of interest
                    color = 'red' if cell_id == 589 else 'blue'
                    ax.scatter(cell_positions[i, 0], cell_positions[i, 1],
                            s=marker_size * 1.5, marker='*', color=color, edgecolors='black', 
                            linewidths=2, zorder=10, alpha=0.8,
                            label=f'Cell {cell_id}')
            
            # After plotting scatter and before saving the figure:
            cbar_ax = fig.add_axes([0.07, 0.25, 0.02, 0.5])  # Adjust as needed for your layout
            cbar_int = plt.colorbar(scatter, cax=cbar_ax)
            cbar_int.set_label('Intracellular H2O2')
            
            ax.set_title(f'2D H2O2 Distribution - Time {time}')
            ax.set_xlabel('X Position')
            #ax.set_ylabel('Y Position')
            
            # Save 2D plot
            output_file_2d = output_dir_2d + f"h2o2_2d_t{ZeroPad(tf, int(time))}.png"
            plt.savefig(output_file_2d, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved 2D: {output_file_2d}")

        # --- 3D PLOT (absolute values, airy translucency) ---
        if plot_3d and gradient_3d is not None:
            fig = plt.figure(figsize=(15, 12))
            ax  = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho')

            # Gradient array is (y, x, z)
            H, W, D = gradient_3d.shape  # (height, width, depth)

            # Slightly denser sampling when NOT squished
            base = max(1, min(W, H, D) // 40)
            sample_rate = max(2, base) if squished else base

            # Indices in (y, x, z) with z flipped so z=0 is at bottom visually
            y_idx = np.arange(0, H, sample_rate)
            x_idx = np.arange(0, W, sample_rate)
            z_idx = np.arange(0, D, sample_rate)[::-1]

            # Meshgrid in the SAME order as the array (y, x, z)
            Yg, Xg, Zg = np.meshgrid(y_idx, x_idx, z_idx, indexing='ij')  # (Ny, Nx, Nz)
            vals       = gradient_3d[::sample_rate, ::sample_rate, ::-sample_rate]  # (Ny, Nx, Nz)

            # Flatten (C-order) for scatter
            X_flat = Xg.ravel()
            Y_flat = Yg.ravel()
            Z_flat = Zg.ravel()
            gradient_flat = vals.ravel()

            # ---- Extracellular gradient coloring (absolute, shared across frames) ----
            from matplotlib.colors import LogNorm, Normalize
            USE_LOG = True
            vmin_global = max(1e-22, mn_gradient)
            vmax_global = max(vmin_global * 10.0, mx_gradient)
            norm = LogNorm(vmin=vmin_global, vmax=vmax_global) if USE_LOG else Normalize(vmin=vmin_global, vmax=vmax_global)

            # Plot all points (lets the colorbar show full range)
            non_zero_mask = gradient_flat >= 0
            vals_plot = gradient_flat[non_zero_mask]
            col_scaled = norm(vals_plot)               # -> 0..1
            colors_rgba = plt.cm.Blues(col_scaled)
            # airy translucency: alpha grows with absolute value (clipped)
            alphas = 0.02 + 0.45 * (vals_plot / vmax_global)
            colors_rgba[:, -1] = np.clip(alphas, 0.02, 0.5)

            gradient_scatter = None
            if np.any(non_zero_mask):
                gradient_scatter = ax.scatter(
                    X_flat[non_zero_mask], Y_flat[non_zero_mask], Z_flat[non_zero_mask],
                    color=colors_rgba, s=1, depthshade=False, label='Extracellular H2O2'
                )
            else:
                ax.text2D(0.02, 0.98, "No extracellular points ≥ vmin",
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=9, color='0.4')

            # Colorbar for extracellular gradient
            sm_grad = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
            sm_grad.set_array([])
            cbar_grad = plt.colorbar(sm_grad, ax=ax, shrink=0.5, aspect=20, pad=0.1)
            cbar_grad.set_label('Extracellular H2O2 Gradient (absolute)')

            # --- Cells (intracellular) ------------------------------------------------
            from matplotlib.colors import LogNorm, LinearSegmentedColormap

            cell_radius_physical = 6.5  # μm
            # Log scaling for intracellular
            vmin_h2o2i = max(1e-22, mn_h2o2i)
            vmax_h2o2i = max(vmin_h2o2i * 10.0, mx_h2o2i)
            h2o2i_normalizer = LogNorm(vmin=vmin_h2o2i, vmax=vmax_h2o2i)

            intracellular_cmap = LinearSegmentedColormap.from_list(
                'black_to_red', ['black', 'yellow', 'orange', 'red', 'darkred', 'purple']
            )

            # Initial placeholder size; we’ll update via hook
            cell_scatter = ax.scatter(
                cell_positions[:, 0], cell_positions[:, 1], cell_positions[:, 2],
                c=cell_h2o2i, cmap=intracellular_cmap, norm=h2o2i_normalizer,
                s=1, alpha=0.6, edgecolors='black', linewidths=0.5,
                label=f'Cells (r={cell_radius_physical}μm)'
            )

            # Optional star highlights
            highlight_artists = []
            for i, cid in enumerate(cell_ids):
                if cid in [589, 877]:
                    color = 'red' if cid == 589 else 'blue'
                    star = ax.scatter(
                        cell_positions[i, 0], cell_positions[i, 1], cell_positions[i, 2],
                        s=1, marker='*', color=color, edgecolors='black',
                        linewidths=2, zorder=10, alpha=0.8,
                        label=f'Cell {cid}'
                    )
                    highlight_artists.append(star)

            # Dynamic marker sizing so a 6.5 μm radius looks right in data units
            def _update_cell_marker_sizes(_event):
                try:
                    base = s_for_radius_in_data_units_3d(
                        ax, cell_radius_physical, at_xyz=np.mean(cell_positions, axis=0)
                    )
                    cell_scatter.set_sizes(np.full(cell_positions.shape[0], base))
                    for star in highlight_artists:
                        star.set_sizes([base * 1.5])
                except Exception:
                    pass

            cid_sizes = fig.canvas.mpl_connect('draw_event', _update_cell_marker_sizes)
            _update_cell_marker_sizes(None)

            # Labels / title
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z Position')
            ax.set_title(f'3D H2O2 Distribution - Time {time}')

            # Limits
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_zlim(0, D)

            # Aspect: squish Z if requested
            if squished:
                ax.set_box_aspect((W, H, D))
            else:
                ax.set_box_aspect((W, H, D*10))

            # Intracellular colorbar on the left
            cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])
            cbar_cell = plt.colorbar(cell_scatter, cax=cbar_ax)
            cbar_cell.set_label('Intracellular H2O2')

            ax.legend()

            # Save
            if squished:
                output_file_3d = output_dir_3d + f"h2o2_3d_t{ZeroPad(tf, int(time))}.png"
            else:
                output_file_3d = output_dir_3d + f"h2o2_3d_t{ZeroPad(tf, int(time))}_unsquished.png"

            plt.savefig(output_file_3d, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved 3D: {output_file_3d}")

    
        if make_graph:
            for i, cell_id in enumerate(cell_ids):
                point = cell_positions[i]
                if cell_id == 589:
                    # if plot_2d:
                    #     extracellular_h2o2 = g.get_gradient_value_at_point_circle(point)
                    # else:
                    extracellular_h2o2 = g.get_gradient_value_at_point_circle(point)
                    h2o2e_track_589.append([time, extracellular_h2o2])
                    h2o2i_track_589.append([time, cell_h2o2i[i]])
                if cell_id == 877:
                    # if plot_2d:
                    #     extracellular_h2o2 = g.get_gradient_value_at_point_circle(point)
                    # else:
                    extracellular_h2o2 = g.get_gradient_value_at_point_circle(point)
                    h2o2e_track_877.append([time, extracellular_h2o2])
                    h2o2i_track_877.append([time, cell_h2o2i[i]])
        
        if plot_3d:
            # --- SLICE THROUGH A LINE THAT PASSES CELL 589 --------------------------------
            # Helper: compute domain-edge endpoints of a line with angle theta (deg) through point p
            def _endpoints_from_angle_through_point(w, h, p, theta_deg):
                px, py = float(p[0]), float(p[1])
                th = np.deg2rad(theta_deg)
                dx, dy = np.cos(th), np.sin(th)
                eps = 1e-9
                ts = []
                if abs(dx) > eps:
                    ts.extend([(0 - px)/dx, ((w-1) - px)/dx])
                if abs(dy) > eps:
                    ts.extend([(0 - py)/dy, ((h-1) - py)/dy])
                cand = []
                for t in ts:
                    qx, qy = px + t*dx, py + t*dy
                    if -1e-6 <= qx <= w-1+1e-6 and -1e-6 <= qy <= h-1+1e-6:
                        cand.append((t, np.array([qx, qy])))
                if len(cand) < 2:
                    # fall back to full diagonal if something degenerate happens
                    return np.array([0, 0]), np.array([w-1, h-1])
                cand.sort(key=lambda x: x[0])
                return cand[0][1], cand[-1][1]

            # Helper: extract a vertical slice image along a polyline of XY points (nearest-neighbor),
            # averaging over a small thickness (orthogonal offsets)
            def _extract_slice_image(grad3d, xy0, xy1, depth, thickness_px=0, n_samp=None, n_z=None):
                seg = xy1 - xy0
                L = float(np.hypot(seg[0], seg[1]))
                if n_samp is None:
                    n_samp = max(2, int(np.ceil(L)) + 1) * 4  # 2x finer sampling
                xs = np.linspace(xy0[0], xy1[0], n_samp)
                ys = np.linspace(xy0[1], xy1[1], n_samp)

                # perpendicular for thickness
                if thickness_px > 0:
                    n = np.array([-seg[1], seg[0]], dtype=float)
                    n_norm = np.linalg.norm(n)
                    if n_norm > 0:
                        n /= n_norm
                    offsets = np.arange(-thickness_px, thickness_px+1, 1, dtype=int)
                else:
                    n = np.array([0.0, 0.0]); offsets = np.array([0], dtype=int)

                H, W, D = grad3d.shape
                if n_z is None:
                    n_z = D * 4  # 2x finer in z
                zs = np.linspace(0, D-1, n_z)

                acc = np.zeros((n_z, n_samp), dtype=float)
                cnt = np.zeros((n_z, n_samp), dtype=int)

                for off in offsets:
                    x_off = xs + off * n[0]
                    y_off = ys + off * n[1]
                    for zi, zf in enumerate(zs):
                        coords = np.vstack([y_off, x_off, np.full_like(x_off, zf)])
                        vals = map_coordinates(grad3d, coords, order=1, mode='nearest')
                        acc[zi, :] += vals
                        cnt[zi, :] += 1

                slice_img = acc / np.maximum(cnt, 1)
                dist_axis = np.linspace(0.0, L, n_samp)
                return slice_img, dist_axis, zs


            # 1) Find cell 589 grid position
            cell_ids_arr = np.array(cell_ids)
            if np.any(cell_ids_arr == 589):
                i589 = int(np.where(cell_ids_arr == 589)[0][0])
                cx, cy, cz = cell_positions[i589]  # grid coords
            if np.any(cell_ids_arr == 877):
                i877 = int(np.where(cell_ids_arr == 877)[0][0])
                cx, cy, cz = cell_positions[i877]  # grid coords
            else:
                # fallback: use the brightest cell or the first one
                i589 = 0
                cx, cy, cz = cell_positions[i589]
                print("Cell 589 not found; slicing through first cell instead.")

            # 2) Choose the slice line orientation (angle in degrees).
            #    Set this to match your drawn line. Example: 35°.
            theta_deg = 35.0

            # 3) Compute endpoints of the line (edge-to-edge) through cell 589
            p0, p1 = _endpoints_from_angle_through_point(W, H, (cx, cy), theta_deg)

            # 4) Extract the slice image (absolute values) with a little thickness averaging
            #    Increase thickness_px (e.g., 2–5) if the line looks too sparse/noisy.
            slice_img, dist_axis, zs = _extract_slice_image(gradient_3d, p0, p1, D, thickness_px=5)

            # 5) Plot the slice: distance along line (x) vs Z (y). Absolute values; consistent color scale.
            fig_slice, ax_slice = plt.subplots(figsize=(12, 7))  # taller figure helps

            slice_abs = np.asarray(slice_img, dtype=float)
            pos = slice_abs[slice_abs > 0]

            if pos.size == 0:
                print("Slice has no positive values to display.")
                plt.close(fig_slice)
            else:
                print(f"[slice] n={pos.size}, min={pos.min():.3e}, max={pos.max():.3e}")

                vmin = 1e-17
                vmax = 5e-9
                use_log = (vmax / vmin) >= 20

                dz = getattr(g, "_dz", 1.0)
                dist_axis_phys = dist_axis * getattr(g, "_dx", 1.0)
                z_axis_phys = zs * dz

                norm = None
                if use_log:
                    from matplotlib.colors import LogNorm
                    norm = LogNorm(vmin=vmin, vmax=vmax)

                color_map = intracellular_cmap = LinearSegmentedColormap.from_list('black_to_red', ['black', 'yellow', 'orange', 'red', 'darkred', 'purple'])

                im = ax_slice.imshow(
                    slice_abs,
                    origin='lower',
                    cmap=color_map,
                    extent=[dist_axis_phys[0], dist_axis_phys[-1], z_axis_phys[0], z_axis_phys[-1]],
                    norm=norm,
                    vmin=None if use_log else vmin,
                    vmax=None if use_log else vmax,
                    interpolation='bicubic'
                )

                aspect_physical = (z_axis_phys[-1] - z_axis_phys[0]) / (dist_axis_phys[-1] - dist_axis_phys[0] + 1e-12)
                ax_slice.set_aspect(aspect_physical * 1000)

                # Mark the vertical through Cell 589 (red dashed)
                seg = p1 - p0
                x_along_589 = float(np.dot(np.array([cell_positions[i589, 0], cell_positions[i589, 1]]) - p0, seg) / np.linalg.norm(seg))
                ax_slice.plot([x_along_589, x_along_589], [z_axis_phys[0], z_axis_phys[-1]], 'r--', lw=1.5, label='Through Cell 589')

                # Mark the vertical through Cell 877 (green dashed)
                x_along_877 = float(np.dot(np.array([cell_positions[i877, 0], cell_positions[i877, 1]]) - p0, seg) / np.linalg.norm(seg))
                ax_slice.plot([x_along_877, x_along_877], [z_axis_phys[0], z_axis_phys[-1]], 'g--', lw=1.5, label='Through Cell 877')

                ax_slice.legend(loc='upper right')

                cbar = plt.colorbar(im, ax=ax_slice, pad=0.02)
                cbar.set_label(f'Extracellular H₂O₂ (absolute, {"log" if use_log else "linear"} scale)')

                output_file_slice = output_dir_3d + f"h2o2_slice_cell589_t{ZeroPad(tf, int(time))}.png"
                plt.savefig(output_file_slice, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig_slice)
                print(f"Saved slice: {output_file_slice}")

        # --- Plot H2O2 over time for both cells ---
        plt.figure(figsize=(10, 6))
        if h2o2e_track_589:
            plt.plot(*zip(*h2o2e_track_589), marker='o', color='tab:blue', linestyle='-', label='Extracellular H2O2 at Cell 589')
        if h2o2i_track_589:
            plt.plot(*zip(*h2o2i_track_589), marker='o', color='tab:orange', linestyle='-', label='Intracellular H2O2 in Cell 589')
        if h2o2e_track_877:
            plt.plot(*zip(*h2o2e_track_877), marker='o', color='tab:blue', linestyle='--', label='Extracellular H2O2 at Cell 877')
        if h2o2i_track_877:
            plt.plot(*zip(*h2o2i_track_877), marker='o', color='tab:orange', linestyle='--', label='Intracellular H2O2 in Cell 877')

        plt.xlabel('Time')
        plt.ylabel('H2O2 Concentration')
        plt.legend()
        plt.title('H2O2 Over Time')
        plt.yscale('log')
        plt.grid()
        plt.savefig(path + 'all_h2o2_over_time_cell_589_877.png', dpi=150)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(*zip(*h2o2i_track_877), marker='o', color='tab:orange', linestyle='--', label='Intracellular H2O2 in Cell 877')
        plt.ylabel('H2O2i 877')
        plt.xlabel('Time')
        plt.title('H2O2i in Cell 877 Over Time')
        plt.yscale('log')
        plt.ylim(0.1e-10, 1e-8)
        plt.savefig(path + 'all_h2o2i_877.png')
        plt.close()

        all_times = sorted(set(
            [t for t, _ in h2o2e_track_589] +
            [t for t, _ in h2o2i_track_589] +
            [t for t, _ in h2o2e_track_877] +
            [t for t, _ in h2o2i_track_877]
        ))
        df = pd.DataFrame({'time': all_times})
        df = df.set_index('time')

        df['h2o2e_track_589'] = pd.Series(dict(h2o2e_track_589))
        df['h2o2i_track_589'] = pd.Series(dict(h2o2i_track_589))
        df['h2o2e_track_877'] = pd.Series(dict(h2o2e_track_877))
        df['h2o2i_track_877'] = pd.Series(dict(h2o2i_track_877))
        df = df.reset_index()

        excel_path = os.path.join(path, 'all_h2o2_tracks.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Saved H2O2 tracks to {excel_path}")

    print(f"Completed! 2D images: {output_dir_2d if plot_2d else 'Skipped'}")
    print(f"3D images: {output_dir_3d if plot_3d else 'Skipped'}")

def getCombinedH2O2Images3D(path, t0=0, tf=200, plot_2d=True, plot_3d=True, make_graph=True, squished=True):
    """ Main function to generate combined H2O2 images in 2D and/or 3D """
    # Load simulation data
    data = sim.TimeSeries(path)
    
    # Check if H2O2 gradient exists
    grad_names = data._grad_names
    h2o2_gradient = None
    for name in grad_names:
        if 'H2O2' in name or 'h2o2' in name:
            h2o2_gradient = name
            break
    
    if h2o2_gradient is None:
        print("No H2O2 gradient found in simulation data!")
        return
    
    print(f"Found H2O2 gradient: {h2o2_gradient}")
    
    # Generate combined images
    save_combined_h2o2_images_3d(path, data, t0, tf, h2o2_gradient, plot_2d, plot_3d, make_graph, squished)

# ...existing code...

if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/13.0/'
    # Generate both 2D and 3D plots
    getCombinedH2O2Images3D(path, t0=0, tf=600, plot_2d=True, plot_3d=False, make_graph=True, squished=False)