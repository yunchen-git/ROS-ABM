
import simulationMath1 as simMath
from vpython import *
import random as r
import time as t_m
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import os

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

def save_2D_gradient_images(name, save_path, data, axis):
    """ Will save a 2D slice of the gradient specified by name along the given
        axis which by default is xy
    """
    #load the data
    times = list(data.get_times())
    #get the min and max overall
    mx = -1
    mn = 1000000000000
    
    #save the planes
    planes = []
    for i in range(0, len(times)):
        #get the gradient
        g = data.get_gradient_at_time(times[i], name)
        #now slice it depending on the axis
        plane = None
        #compute half plane vbalue for x and y
        x, y = g.C.shape
        #print(len(g.C.shape))
        if(axis == "xy"):
            n=2     
            plane=g.C
##            for k in range(1,n):
##                plane=plane+g.C[:,:,k]/(n+1)
##            temp=g.C[:, :, 1]-g.voidSpace[:,:,1]*10e6
##            temp1=np.maximum(g.C[:,:,1],temp)
##            plane = temp1
        if(axis=="xyv"):
            plane=g.voidSpace[:,:,1]
        if (axis=="xzv"):
            plane=g.voidSpace[:,y/2,:]
        if(axis == "xz"):
            plane = g.C[:, y/2, :]
        if(axis == "yz"):
            plane = g.C[x/2, :, :]
        #save the slice
        planes.append(plane.T)
        #compute the min and max for graphing
        mn = min(np.min(plane), mn)
        mx = max(np.max(plane), mx)      

    # print(len(planes))
    #now make and save a figure for each slice
    for i in range(0, len(planes)):
        #set the min and max
        #draw the figure
        #plt.figure(figsize=(6,6))
        plt.imshow(planes[i],
                   cmap=plt.cm.seismic,
                   vmin = mn,
                   vmax = mx,
                   interpolation = 'bicubic',
                   origin= 'lower')
        
        #add a title
##        plt.title("Morphogen Gradient" + "   Time: " + repr(times[i]))
        plt.title("H2O2 Media Gradient")

        ax=plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider=make_axes_locatable(ax)
        cax=divider.append_axes("left", size="5%", pad=0.05)    
        cb=plt.colorbar(cax=cax)

        cb.set_label('Morphogen Concentration',labelpad=-70) 
        cb.ax.yaxis.set_ticks_position('left')
        #zeropad it
        s = ZeroPad(10, times[i])        
        #save the figure
        
        if axis=="xyv":
            plt.savefig(save_path + "void_" + s +axis + ".png")
        else:
            new_path = save_path + "all_" + g.name
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            plt.savefig(new_path + "/" + s +axis + ".png")
            print("Saving: " + new_path + "/" + s + axis +".png")
        plt.clf()
        plt.cla()
    plt.close()
    
        

        
