################################################################################
# File: zplot_intC_extc.py
# Purpose:
#   Legacy compatibility shim for combined intracellular/extracellular
#   visualization helpers now housed at:
#   ros_abm.analysis.zplot_intC_extc
################################################################################
from ros_abm.analysis.zplot_intC_extc import *  # noqa: F401,F403


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/22.0/'
    getCombinedH2O2Images3D(path, t0=0, tf=600, plot_2d=True, plot_3d=False, make_graph=True, squished=False)
