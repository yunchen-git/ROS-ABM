################################################################################
# File: gradientImage.py
# Purpose:
#   Legacy compatibility shim for gradient-image wrapper now housed at:
#   ros_abm.analysis.gradient_image
################################################################################
from ros_abm.analysis.gradient_image import *  # noqa: F401,F403


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/2.0/'
    getGradientImages(path)
