################################################################################
# File: gradientImage.py
# Purpose:
#   Thin convenience wrapper script to generate extracellular gradient images
#   from an existing simulation run folder.
#
# Key public API:
#   - getGradientImages(path)
#
# How it fits in the codebase:
#   Calls Simulation.TimeSeries to load saved data, then delegates image export
#   to Analysis.save_2D_gradient_images(...).
#
# Inputs/Outputs:
#   - Reads simulation run outputs from disk
#   - Writes PNG images (gradient snapshots) to the run folder
#
# Notes / gotchas:
#   - The __main__ section contains hard-coded example paths; keep as examples.
################################################################################
from ros_abm.analysis import analysis as af
import Simulation as sim

def getGradientImages(path):
    data=sim.TimeSeries(path)
    grad_names=data._grad_names
    
    for i in range(len(grad_names)):
        af.save_2D_gradient_images(grad_names[i],path,data,axis="xy")


if __name__=='__main__':
    path='/Users/huyunchen/Documents/PhD/test/2.0/'
    getGradientImages(path)


