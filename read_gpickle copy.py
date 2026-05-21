################################################################################
# File: read_gpickle copy.py
# Purpose:
#   Legacy compatibility shim for the tracking-oriented API now housed at:
#   ros_abm.analysis.read_gpickle_tracking
################################################################################
from ros_abm.analysis.read_gpickle_tracking import *  # noqa: F401,F403


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/1.0/'
    run_tracking_cli(path, t0=0, tf=600)
