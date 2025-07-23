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
                 outside_c = 0.0, vol_units = 1e-6):
        dx = ddim
        dy = ddim
        dz = ddim

        # save the unit volume
        self._vol_units = 1e-6
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
        self.dt = 0.5 / ((D/dx2) + (D/dy2))#+(D/dz2))

        # set the initial conditions
        #self.Ci = np.zeros((self.x_dim, self.y_dim, self.z_dim))*outside_c
        self.Ci = np.zeros((self.x_dim, self.y_dim))*outside_c
        #self.C = np.zeros((self.x_dim, self.y_dim, self.z_dim))
        self.C = np.zeros((self.x_dim, self.y_dim))
        # set the boundary conditions
        self._c_out = outside_c

        # compute the unit volume
        self._grid_vol = (self._vol_units **3) * self._dx * self._dy

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
        return (self.x_dim, self.y_dim)#, self.z_dim)
    
    def cubic_interpolate_at_object_locations(self, agents):
        """ Interpolate the gradient at the location of the agents
            agents - a list of agents
        """
        #get all the agent location
        points = []
        for i in range(0, len(agents)):
            #get the indicies of the gradient
            pos = agents[i].get_gradient_location(self.name)
            #print(pos)
            x = int(pos[0])
            y = int(pos[1])
            #z = int(pos[2])
            #points.append([x,y,z])
            points.append([x,y])
        #Now convert the points to an array
        points = np.array(points)
        #transpose it to the correct format
        cords = points.T
        #get the interpolated values
        zi = ndimage.map_coordinates(self.C, cords, order=5, mode='reflect')
        #set all the agents values
        for i in range(0, len(agents)):
            #now set the gradient value at this point
            agents[i].set_gradient_value(self.name, zi[i])
    
    def get_object_position_on_grid(self, distance):
        """ Assumes the distance vector is assigned as a vector pointing from
            the center out towards the point to index. Will return an error if
            point is outside the range of the grid
        """
        #now we know the spatial resolution per grid spacing
        dx = distance[0] / self._dx
        dy = distance[1] / self._dy
        dz = distance[2] / self._dz

        #get the center loction
        cx = int(self.x_dim / 2)
        cy = int(self.y_dim / 2)
        cz = int(self.z_dim / 2)

        #return the x, y, z indicies to add at
        return dx, dy#cx + dx, cy + dy #, cz + dz

    def get_gradient_value_at_point(self, point):
        """ Will get the gradient value at a specific point. Useful for
            interpolation when the grid size is much bigger than a single cell.
            Assume the point is a distance vector representing the radius
            of the cell within the strutural aggregate
        """
        #get the indicies of the point
        x = int(point[0])
        y = int(point[1])
        z = int(point[2])
        point = [x, y, z]
        x, y = self.get_object_position_on_grid(point)
        #get the indices
        ix = int(x)
        iy = int(y)
        iz = int(z)

        #perform a trilinear interpolation from wikipedia.org
        #x
        if(x - ix >= .5):
            x0 = ix
            x1 = ix + 1
        else:
            x0 = ix - 1
            x1 = ix
        #y
        if(y - iy >= .5):
            y0 = iy
            y1 = iy + 1
        else:
            y0 = iy - 1
            y1 = iy
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

        #return the predicted value
        return c
    
    def set_gradient_value_at_point(self, point, value):
        """ Will set the gradient value at a specific point. Useful for
            interpolation when the grid size is much bigger than a single cell.
            Assume the point is a distance vector representing the radius
            of the cell within the strutural aggregate
        """
        #get the indicies of the point
        x = int(point[0])
        y = int(point[1])
        z = int(point[2])
        point = [x, y, z]
        x, y = self.get_object_position_on_grid(point)

        #set as int
        x = int(x)
        y = int(y)

        #return the predicted value
        self.Ci[x, y] = value
    
    def update(self, t_end, sink, source):
        """ Solves the system over using the predetermined time step dt
            until the end time of the simulation is reached.
            t_end - the end time to solve the system towards
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Convert fixed parameters to torch tensors on the GPU
        dt = torch.tensor(self.dt, dtype=torch.float32).to(device)
        D = torch.tensor(self._D, dtype=torch.float32).to(device)
        grid_vol = torch.tensor(self._grid_vol, dtype=torch.float32).to(device)
        c_out = torch.tensor(self._c_out, dtype=torch.float32).to(device)

        # Convert sink/source arrays to GPU tensors
        sink_tensor = torch.tensor(sink, dtype=torch.float32).to(device)
        source_tensor = torch.tensor(source, dtype=torch.float32).to(device)

        # Initial concentration as GPU tensor with shape (1, 1, H, W)
        Ci_tensor = torch.tensor(self.Ci, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Build combined 2D Laplacian kernel
        dx2 = self._dx ** 2
        dy2 = self._dy ** 2
        laplacian_kernel = torch.tensor([
            [0, 1/dy2, 0],
            [1/dx2, -2*(1/dx2 + 1/dy2), 1/dx2],
            [0, 1/dy2, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        t = 0
        epsilon = 1e-10
        diff = epsilon  * 2

        while(t <= t_end and diff >= epsilon):
            # Pad the concentration field with constant boundary
            Ci_padded = F.pad(Ci_tensor, (1, 1, 1, 1), mode='reflect')

            # Apply Laplacian via convolution
            lap = F.conv2d(Ci_padded, laplacian_kernel, padding=0)

            # Diffusion step
            C_tensor = Ci_tensor + D * dt * lap

            # Add source/sink terms
            C_tensor += (-sink_tensor * dt + source_tensor * dt) / grid_vol

            # Clamp negative values
            C_tensor = torch.clamp(C_tensor, min=0.0)

            # Compute difference for convergence
            diff = torch.sum(torch.abs(C_tensor - Ci_tensor))

            # Update for next iteration
            Ci_tensor = C_tensor
            t += self.dt

        # Save back to CPU NumPy array
        self.Ci = Ci_tensor.squeeze().cpu().numpy()
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