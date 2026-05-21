################################################################################
# File: legacy_shared_utils.py
# Purpose:
#   Shared legacy utility helpers that are duplicated across analysis scripts.
#   Keep function behavior byte-for-byte equivalent to existing call sites.
################################################################################
import gc
import math

import numpy as np


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
    for _ in range(0, d1-d2):
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
