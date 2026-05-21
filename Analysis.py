################################################################################
# File: Analysis.py
# Purpose:
#   Legacy compatibility shim for gradient postprocess helpers now housed at:
#   ros_abm.analysis.analysis
################################################################################
from ros_abm.analysis.analysis import *  # noqa: F401,F403


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/10.0/'
    getGradientImages(path)
