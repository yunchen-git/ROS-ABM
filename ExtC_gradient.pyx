# cython: language_level=3

cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

# Libraries
import numpy as np
cimport numpy as np
import pickle
import time
import cython
cimport cython
import math

from cython.view cimport array as cvarray
from libc.stdlib cimport abs
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.interpolate import griddata

import gc, atexit
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------

"""
To change this code, first modify the code in this file then
save a copy of this script as ExtC_gradient.pyx

To compile go to a command console and:
1. Change to this directory (type cd, press space then drag/drop this folder to the console. Press enter)
2. Type out: "python setup.py build_ext --inplace" without the quotation marks. Press enter.

"""

class ExtC_Diff(object):
    """ Class which defines a gradient.
        name - the name of the species
        D - the diffusion coeffiecnt
        x, y, z - the spaital size of the grid
        dx, dy, dz - the spatial resolution of each grid step
        outside_c - the value fo the gradient outside the box
        unit_vol = the uinit vole in L (ul) m or um
    """

    def __init__(self, name, D, x, y, z, ddim,
                 outside_c, vol_units = 1e-15):
        dx = ddim
        dy = ddim
        dz = ddim

        # save the unit volume
        self._vol_units = vol_units
        # save the name
        self.name = name
        # set the diffusion coefficient
        self._D = float(D)
        # set the spatial size of the grid
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        # set the spatial resolution of each grid step
        self._dx = float(ddim)
        self._dy = float(ddim)
        self._dz = float(ddim)
        # solve for the index of the array
        self.x_dim = int(x/dx)
        self.y_dim = int(y/dy)
        self.z_dim = int(z/dz)
        # solve some more of the spatial constraints
        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz

        # time step solution
        if z > 0:  # Use z instead of z_dim for this check
            self.dt = 0.5 / ((D/dx2) + (D/dy2) +(D/dz2))
            # compute the unit volume for 3D
            self._grid_vol = self._vol_units * self._dx * self._dy * self._dz
            
            # Create 3D arrays
            self.Ci = np.zeros((self.x_dim, self.y_dim, self.z_dim)) * outside_c
            self.C = np.zeros((self.x_dim, self.y_dim, self.z_dim))
        else:
            self.dt = 0.5 / ((D/dx2) + (D/dy2))
            # compute the unit volume for 2D (no dz multiplication)
            self._grid_vol = self._vol_units * self._dx * self._dy
            
            # Create 2D arrays
            self.Ci = np.zeros((self.x_dim, self.y_dim)) * outside_c
            self.C = np.zeros((self.x_dim, self.y_dim))
        
        # set the boundary conditions
        self._c_out = outside_c

        # define the laplacian operator
        self._ly = np.array([
            [0,1./dy2,0],
            [0,-2./(dy2),0],
            [0,1./dy2,0]
            ])
        self._lx = np.array([
            [0,0,0],
            [1./dx2, -2./dx2, 1./dx2],
            [0,0,0],
            ])

    def set_initial_conditions(self, mask):
        """ Set the initial conditions of the gradient
            mask - a 2D array of the initial conditions
        """
        self.Ci = (mask==0) * self._c_out
        self.C = self.Ci.copy()

    def save(self, base_path, time_stamp):
        """ Save the current state of the gradient
            base_path - the base path to save the file
            time_stamp - the time stamp of the file
        """
        path = base_path + self.name + "_" + time_stamp
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

        return path

    def shape(self):
        """ Return the shape of the gradient
        """
        return self.Ci.shape
    
    def cubic_interpolate_at_object_locations(self, agents):
        """
        Interpolate self.C (x,y,z) at agent positions using cubic (order=3).
        Agents may provide 2D or 3D positions; z defaults to center slice if missing.
        """
        if self.C.ndim != 3 or self.z_dim == 0:
            # Fallback to legacy 2D behavior if you ever pass a 2D field
            points = []
            for a in agents:
                pos = a.get_gradient_location(self.name)
                x = float(pos[0]); y = float(pos[1])
                # Map to grid coords (center+offset)
                gx, gy = self.get_object_position_on_grid([x, y])
                points.append([gx, gy])
            coords = np.array(points, dtype=np.float64).T  # shape (2,N)
            # clip to [0, dim-1) for cubic
            coords[0] = np.clip(coords[0], 0.0, self.x_dim - 1 - 1e-6)
            coords[1] = np.clip(coords[1], 0.0, self.y_dim - 1 - 1e-6)
            zi = ndimage.map_coordinates(self.C, coords, order=3, mode='reflect')
            for i, a in enumerate(agents):
                a.set_gradient_value(self.name, float(zi[i]))
            return

        # --- 3D path ---
        xs, ys, zs = [], [], []
        z_center = (self.z_dim - 1) / 2.0
        for a in agents:
            pos = a.get_gradient_location(self.name)  # physical coords or distance vector
            # allow 2D or 3D agent locations
            if len(pos) >= 3:
                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            else:
                px, py, pz = float(pos[0]), float(pos[1]), 0.0

            # Map to grid indices (center + offset)
            gx, gy, gz = self.get_object_position_on_grid([px, py, pz])

            # If caller provided only 2D semantics, you can choose to pin to center slice:
            if len(pos) < 3:
                gz = z_center

            xs.append(gx); ys.append(gy); zs.append(gz)

        coords = np.vstack([np.asarray(xs, dtype=np.float64),
                            np.asarray(ys, dtype=np.float64),
                            np.asarray(zs, dtype=np.float64)])  # (3,N)

        # Clip into valid open interval for cubic interpolation
        coords[0] = np.clip(coords[0], 0.0, self.x_dim - 1 - 1e-6)
        coords[1] = np.clip(coords[1], 0.0, self.y_dim - 1 - 1e-6)
        coords[2] = np.clip(coords[2], 0.0, self.z_dim - 1 - 1e-6)

        zi = ndimage.map_coordinates(self.C, coords, order=3, mode='reflect')  # (N,)

        for i, a in enumerate(agents):
            a.set_gradient_value(self.name, float(zi[i]))
    
    def get_object_position_on_grid(self, distance):
        """ Assumes the distance vector is assigned as a vector pointing from
            the center out towards the point to index. Will return an error if
            point is outside the range of the grid
        """
        #now we know the spatial resolution per grid spacing
        if self.z_dim > 0:
            dx = distance[0] / self._dx
            dy = distance[1] / self._dy
            dz = distance[2] / self._dz
        else:
            dx = distance[0] / self._dx
            dy = distance[1] / self._dy
            dz = 0

        ##get the center loction
        #cx = int(self.x_dim / 2)
        #cy = int(self.y_dim / 2)
        #cz = int(self.z_dim / 2)

        #return the x, y, z indicies to add at
        if self.z_dim > 0:
            return dx, dy, dz
        else:
            return dx, dy#cx + dx, cy + dy, cz + dz

    def get_gradient_value_at_point(self, point):
        """ Will get the gradient value at a specific point. Useful for
            interpolation when the grid size is much bigger than a single cell.
            Assume the point is a distance vector representing the radius
            of the cell within the strutural aggregate
        """
        #get the indicies of the point
        x = int(point[0])
        y = int(point[1])
        z = int(point[2]) if len(point) > 2 else 0

        # Use appropriate point dimensionality
        if len(point) > 2:
            point_coords = [x, y, z]
            grid_pos = self.get_object_position_on_grid(point_coords)
            x, y, z = grid_pos
            ix, iy, iz = int(x), int(y), int(z)
        else:
            point_coords = [x, y]
            grid_pos = self.get_object_position_on_grid(point_coords)
            x, y = grid_pos
            ix, iy = int(x), int(y)

        #perform a trilinear interpolation from wikipedia.org
        #x

        x0 = ix
        x1 = ix + 1

        #y

        y0 = iy
        y1 = iy + 1

        #z
        """
        if(z - iz > .5):
            z0 = iz
            z1 = iz + 1
        else:
            z0 = iz - 1
            z1 = iz    
        """

        #solve for xd, yd
        xd = (abs(x - (ix + .5))) / (x1 - x0)
        yd = (abs(y - (iy + .5))) / (y1 - y0)

        # interpolate along the x-axis
        f0 = self.C[x0, y0]*(1 - xd) + self.C[x1, y0]*xd
        f1 = self.C[x0, y1]*(1 - xd) + self.C[x1, y1]*xd

        # interpolate along the y-axis
        c = f0*(1 - yd) + f1*yd

        """
        # TODO: For future 3D trilinear interpolation, use:
        c00 = self.C[x0, y0, z0]*(1 - xd) + self.C[x1, y0, z0]*xd
        c01 = self.C[x0, y0, z1]*(1 - xd) + self.C[x1, y0, z1]*xd
        c10 = self.C[x0, y1, z0]*(1 - xd) + self.C[x1, y1, z0]*xd
        c11 = self.C[x0, y1, z1]*(1 - xd) + self.C[x1, y1, z1]*xd
        c0 = c00*(1 - yd) + c10*yd
        c1 = c01*(1 - yd) + c11*yd
        c = c0*(1 - zd) + c1*zd
        """

        #return the predicted value
        return c
    
    def get_gradient_value_at_point_circle(self, point):
        """ Will get the gradient value at a specific point by averaging
            over a circular area with radius 6.5. Useful for
            interpolation when the grid size is much bigger than a single cell.
            Assume the point is a distance vector representing the radius
            of the cell within the strutural aggregate
        """
        #get the indicies of the point
        x = int(point[0])
        y = int(point[1])
        z = int(point[2]) if len(point) > 2 else 0

        # Use appropriate point dimensionality
        if len(point) > 2 and self.z_dim > 0:
            point_coords = [x, y, z]
            grid_pos = self.get_object_position_on_grid(point_coords)
            center_x, center_y, center_z = grid_pos
        else:
            point_coords = [x, y]
            grid_pos = self.get_object_position_on_grid(point_coords)
            center_x, center_y = grid_pos

        # Define circle radius
        radius = 6.5
        
        # Calculate the bounding box for the circle
        min_x = max(0, int(center_x - radius))
        max_x = min(self.x_dim - 1, int(center_x + radius))
        min_y = max(0, int(center_y - radius))
        max_y = min(self.y_dim - 1, int(center_y + radius))
        
        # Sample points within the circular area
        total_concentration = 0.0
        count = 0
        
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                # Calculate distance from center point
                distance = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                
                # Only include points within the circle
                if distance <= radius:
                    # Check if this is effectively a 2D gradient (z_dim == 0)
                    if self.z_dim == 0 and self.z_dim < 1:
                        # For 2D case, access the single z-slice (should be at index 0)
                        total_concentration += self.C[i, j]
                    else:
                        # For 3D case, use the center z-slice
                        center_z_int = max(0, min(self.z_dim - 1, int(center_z)))
                        total_concentration += self.C[i, j, center_z_int]
                    count += 1
        
        # Return average concentration within the circle
        if count == 0:
        # Fallback to single point if no points found (shouldn't happen with radius 6.5)
            ix, iy = int(center_x), int(center_y)
            return self.C[ix, iy]
        else:
            #print(count)
            return total_concentration / count

    def get_gradient_value_at_point_sphere(self, point, radius_phys=6.5):
        # used to get gradient value in 3D
        # average within a sphere of physical radius (e.g., microns)

        x = float(point[0]); y = float(point[1]); z = float(point[2]) if len(point) > 2 else 0.0
        gx, gy, gz = self.get_object_position_on_grid([x, y, z])

        rx = radius_phys / self._dx
        ry = radius_phys / self._dy
        rz = radius_phys / self._dz

        min_x = max(0, int(gx - rx)); max_x = min(self.x_dim - 1, int(gx + rx))
        min_y = max(0, int(gy - ry)); max_y = min(self.y_dim - 1, int(gy + ry))
        min_z = max(0, int(gz - rz)); max_z = min(self.z_dim - 1, int(gz + rz))

        total = 0.0; count = 0
        C = self.C
        
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                for k in range(min_z, max_z + 1):
                    # Calculate actual physical distance
                    phys_dist_sq = ((i - gx) * self._dx)**2 + ((j - gy) * self._dy)**2 + ((k - gz) * self._dz)**2
                    
                    if phys_dist_sq <= radius_phys**2:
                        total += C[i, j, k]; count += 1

        if count == 0:
            ix = max(0, min(self.x_dim - 1, int(round(gx))))
            iy = max(0, min(self.y_dim - 1, int(round(gy))))
            iz = max(0, min(self.z_dim - 1, int(round(gz))))
            return C[ix, iy, iz]
        else:       
            #print(count)
            return total / count

    def _surface_voxels_2d(self, gx, gy, r_g, s_g):
        """Return list of (ix,iy) on an annulus |d - r_g| <= s_g using voxel centers."""
        x0 = max(0, int(np.floor(gx - (r_g + s_g)))); x1 = min(self.x_dim - 1, int(np.ceil(gx + (r_g + s_g))))
        y0 = max(0, int(np.floor(gy - (r_g + s_g)))); y1 = min(self.y_dim - 1, int(np.ceil(gy + (r_g + s_g))))
        S = []
        for ix in range(x0, x1 + 1):
                for iy in range(y0, y1 + 1):
                    dx = (ix + 0.5) - gx
                    dy = (iy + 0.5) - gy
                    d  = (dx*dx + dy*dy)**0.5
                    if abs(d - r_g) <= s_g:   # same rule everywhere
                        S.append((ix, iy))
        return S

    def _surface_voxels_3d(self, gx, gy, gz, r_g, s_g):
        """Return list of (ix,iy,iz) on a spherical shell |d - r_g| <= s_g using voxel centers."""
        x0 = max(0, int(np.floor(gx - (r_g + s_g)))); x1 = min(self.x_dim - 1, int(np.ceil(gx + (r_g + s_g))))
        y0 = max(0, int(np.floor(gy - (r_g + s_g)))); y1 = min(self.y_dim - 1, int(np.ceil(gy + (r_g + s_g))))
        z0 = max(0, int(np.floor(gz - (r_g + s_g)))); z1 = min(self.z_dim - 1, int(np.ceil(gz + (r_g + s_g))))
        S = []
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                for iz in range(z0, z1 + 1):
                    dx = (ix + 0.5) - gx
                    dy = (iy + 0.5) - gy
                    dz = (iz + 0.5) - gz
                    d  = (dx*dx + dy*dy + dz*dz)**0.5
                    if abs(d - r_g) <= s_g:
                        S.append((ix, iy, iz))
        return S

    def get_gradient_value_on_surface_shell(self, cell, radius_phys=6.5, shell_thickness=0.01):
        """Average concentration over the same surface shell used for deposition."""
        point = cell.location 
        if self.z_dim == 0:
            x = float(point[0]); y = float(point[1])
            z = 0.0
            gx, gy = self.get_object_position_on_grid([x, y])
        else:
            x = float(point[0]); y = float(point[1]); z = float(point[2])
            gx, gy, gz = self.get_object_position_on_grid([x, y, z])
        r_g = radius_phys / self._dx
        s_g = shell_thickness / self._dx

        if getattr(self, 'z_dim', 0) > 0:
            voxels = self._surface_voxels_3d(gx, gy, gz, r_g, s_g)
            vals = [self.C[ix, iy, iz]*self._grid_vol for (ix, iy, iz) in voxels]
        else:
            voxels = self._surface_voxels_2d(gx, gy, r_g, s_g)
            vals = [self.C[ix, iy]*self._grid_vol for (ix, iy) in voxels]
        
        #if cell.ID == 877:
        #    print("Surface voxels:", len(voxels), (np.sum(vals)/(self._grid_vol * len(vals))) if vals else 0.0)

        return (np.sum(vals)/(self._grid_vol * len(vals)) if vals else (self.C[int(round(gx)), int(round(gy))] if self.z_dim == 0
                                            else self.C[int(round(gx)), int(round(gy)), int(round(gz))]))

    def set_gradient_value_at_point(self, point, value):
        x = float(point[0])
        y = float(point[1]) 
        z = float(point[2]) if len(point) > 2 else 0.0
        
        grid_pos = self.get_object_position_on_grid([x, y, z])
        
        if self.z_dim > 0:
            # 3D case - expect 3 values
            gx, gy, gz = grid_pos
            ix = max(0, min(self.x_dim - 1, int(gx)))
            iy = max(0, min(self.y_dim - 1, int(gy)))
            iz = max(0, min(self.z_dim - 1, int(gz)))
            self.Ci[ix, iy, iz] = value
        else:
            # 2D case - expect 2 values
            gx, gy = grid_pos
            ix = max(0, min(self.x_dim - 1, int(gx)))
            iy = max(0, min(self.y_dim - 1, int(gy)))
            self.Ci[ix, iy] = value

    def add_gradient_value_on_cell_surface(self, cell, molar, shell_thickness=0.01):
        if self.z_dim > 0:
            x, y, z = cell.location
            gx, gy, gz = self.get_object_position_on_grid([x, y, z])
        else:
            x, y, z = cell.location[0], cell.location[1], 0.0
            gx, gy = self.get_object_position_on_grid([x, y])

        r_g = cell.radius / self._dx
        s_g = max(shell_thickness, 1e-9) / self._dx

        if getattr(self, 'z_dim', 0) > 0:
            surface = self._surface_voxels_3d(gx, gy, gz, r_g, s_g)
        else:
            surface = self._surface_voxels_2d(gx, gy, r_g, s_g)

        if not surface:
            surface = [(int(round(gx)), int(round(gy)))] if self.z_dim == 0 else [(int(round(gx)), int(round(gy)), int(round(gz)))]

        N = len(surface)  # <-- this N
        delta_C = molar / N  # M per voxel
        #if cell.ID == 877:
        #    print(N, delta_C)

        if self.z_dim == 0:
            for ix, iy in surface:
                self.Ci[ix, iy] += delta_C
        else:
            for ix, iy, iz in surface:
                self.Ci[ix, iy, iz] += delta_C

    def update(self, t_end, sink, source):
        """ Solves the system over using the predetermined time step dt
            until the end time of the simulation is reached.
            t_end - the end time to solve the system towards
        """
    
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Convert fixed parameters to torch tensors
        dt = torch.tensor(self.dt, dtype=torch.float32).to(device)
        D = torch.tensor(self._D, dtype=torch.float32).to(device)
        grid_vol = torch.tensor(self._grid_vol, dtype=torch.float32).to(device)

        is_2d = (self.Ci.ndim == 2)  # This will be True for truly 2D arrays

        if is_2d:
            # 2D case: work directly with 2D arrays
            # Convert to 4D tensors (1, 1, y, x) for 2D convolution
            Ci_2d = torch.tensor(self.Ci, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            sink_2d = torch.tensor(sink, dtype=torch.float32).to(device)
            source_2d = torch.tensor(source, dtype=torch.float32).to(device)

            # Build 2D Laplacian kernel
            dx2 = self._dx ** 2
            dy2 = self._dy ** 2
            
            laplacian_kernel_2d = torch.tensor([
                [0.0, 1.0/dy2, 0.0],
                [1.0/dx2, -2.0*(1.0/dx2 + 1.0/dy2), 1.0/dx2],
                [0.0, 1.0/dy2, 0.0]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            t = 0
            epsilon = 1e-10
            diff = epsilon * 2
            print(f"before : {np.sum(Ci_2d.cpu().numpy())}")
            while(t <= t_end and diff >= epsilon):
                # Pad for 2D convolution
                Ci_padded = F.pad(Ci_2d, (1, 1, 1, 1), mode='reflect')
                
                # Apply 2D Laplacian
                lapCi_2d = F.conv2d(Ci_padded, laplacian_kernel_2d, padding=0)
                
                # Diffusion step for 2D
                C_tensor_2d = Ci_2d + D * dt * lapCi_2d
                
                # Add source/sink terms
                C_tensor_2d += (-sink_2d * dt + source_2d * dt) / grid_vol
                
                # Clamp negative values
                C_tensor_2d = torch.clamp(C_tensor_2d, min=0.0)

                # Compute difference for convergence
                diff = torch.sum(torch.abs(C_tensor_2d - Ci_2d)).item()

                # Update for next iteration
                Ci_2d = C_tensor_2d
                t += self.dt
            
            # Convert back to 2D NumPy array
            self.Ci = Ci_2d.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            print(f"after : {np.sum(self.Ci)}")
        else:
            # 3D case: use helper functions for proper tensor conversion
            def np_to_t5(arr_xyz: np.ndarray) -> torch.Tensor:
                t = torch.as_tensor(arr_xyz, dtype=torch.float32, device=device)   # (x,y,z)
                return t.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)                # (1,1,z,y,x)

            def t5_to_np(t_11zyx: torch.Tensor) -> np.ndarray:
                t = t_11zyx.squeeze(0).squeeze(0).permute(2, 1, 0)                 # (x,y,z)
                return t.detach().cpu().numpy().astype(np.float32)

            # Tensors (shape to 1x1xZxYxX)
            Ci = np_to_t5(self.Ci)
            sink_t = np_to_t5(sink)
            source_t = np_to_t5(source)

            # Build kernels based on dimensionality
            dx2 = self._dx ** 2
            dy2 = self._dy ** 2
            dz2 = self._dz ** 2

            # Use 3D convolution
            lap = torch.zeros((1,1,3,3,3), dtype=torch.float32, device=device)
            lap[0,0,1,1,1] = -2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
            lap[0,0,1,1,0] =  1.0/dx2  # x-
            lap[0,0,1,1,2] =  1.0/dx2  # x+
            lap[0,0,1,0,1] =  1.0/dy2  # y-
            lap[0,0,1,2,1] =  1.0/dy2  # y+
            lap[0,0,0,1,1] =  1.0/dz2  # z-4
            lap[0,0,2,1,1] =  1.0/dz2  # z+

            t = 0
            epsilon = 1e-10
            diff = epsilon * 2

            while(t <= t_end and diff >= epsilon):
                # 3D case: pad all dimensions
                Ci_padded = F.pad(Ci, (1, 1, 1, 1, 1, 1), mode='reflect')
    
                # Apply Laplacian to CONCENTRATION (not moles)
                lapCi = F.conv3d(Ci_padded, lap, padding=0)
                
                # Update concentration
                C_tensor = Ci + D * dt * lapCi
                
                # Add source/sink terms (convert from moles/time to concentration/time)
                C_tensor += (-sink_t * dt + source_t * dt) / grid_vol

                # Clamp negative values
                C_tensor = torch.clamp(C_tensor, min=0.0)

                # Compute difference for convergence
                diff = torch.sum(torch.abs(C_tensor - Ci)).item()

                # Update for next iteration
                Ci = C_tensor
                t += self.dt

            # Save back to CPU NumPy array
            self.Ci = t5_to_np(Ci)

        self.C = self.Ci.copy()

    def __repr__(self):
        """ Returns a string representation of the gradient
        """
        return self.name + ": " + repr(self.Ci.shape) + " " + repr(np.min(self.Ci)) + " " + repr(np.max(self.Ci))
        
    #OPERATORS - equals and not equals
    #eq - eqauls
    def __eq__(self, other):
        if(other is None):
            #return not equal
            return False
        if(isinstance(other, ExtC_Diff)):
            if(other.name == self.name):
                return True
            else:
                return False
        #if nothing esle then return false
        return False
    
    #ne - not equals
    def __ne__(self, other):
        return not(self.__eq__(other))
    
    #also make it hashable
    def __hash__(self):
        return hash(self.name)