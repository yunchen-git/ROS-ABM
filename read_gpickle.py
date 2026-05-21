################################################################################
# File: read_gpickle.py
# Purpose:
#   Legacy compatibility shim for network snapshot rendering helpers now housed
#   at: ros_abm.analysis.read_gpickle
################################################################################
from ros_abm.analysis.read_gpickle import *  # noqa: F401,F403


if __name__ == '__main__':
    path = '/Users/huyunchen/Documents/PhD/test/12.0/'
    read_network(path, 0, 200)
