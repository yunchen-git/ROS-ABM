################################################################################
# File: Model_run.py
# Purpose:
#   Batch/entry-point script for launching one or more ROS-ABM simulations.
#   Defines run parameters (save path, inhibition flags, magnitudes, timing)
#   and calls Model_initialParameters.main(...) to build + run each simulation.
#
# Key public API:
#   - Script entry point under `if __name__ == "__main__":`
#
# How it fits in the codebase:
#   Primary “launcher” for the project.
#   - Uses CountDirectory.newDirect(...) to create the next numeric run folder.
#   - Delegates all model construction to Model_initialParameters.py.
#
# Inputs/Outputs:
#   - Creates a new run directory under the configured save path.
#   - Produces all simulation outputs via Simulation.save() called downstream.
#
# Notes / gotchas:
#   - Keep parameter names/ordering stable: Model_initialParameters.main(...)
#     is called from here with positional arguments.
################################################################################
import Model_initialParameters as mod
from CountDirectory import newDirect
import getpass
import os
from dataclasses import dataclass, field


def resolve_output_base_path(output_base_path=None):
    if output_base_path is None:
        return os.path.expanduser('~/Documents/PhD/test')
    return os.path.expanduser(output_base_path)


@dataclass
class RunSettings:
    length: int = 1
    output_base_path: str = '~/Documents/PhD/test'
    inh_val: list = field(default_factory=lambda: [1.0])
    per_inh: list = field(default_factory=lambda: [1.0])
    inh_time: int = 1000
    inh_time1: int = 1000
    inhib_type: str | None = None
    magnitudes: list = field(default_factory=lambda: [1])
    chem_mode: str = "full"


def main():
    settings = RunSettings()
    length = settings.length
    user=getpass.getuser()

    ##Set save path for simulations. If you create a new, empty folder 
    ##then everytime you run this code it will make a new Simulation folder
    path = resolve_output_base_path(settings.output_base_path)

    inh_val = settings.inh_val
    per_inh = settings.per_inh
    inh_time = settings.inh_time
    inh_time1 = settings.inh_time1
    inhib_type = settings.inhib_type
    magnitudes = settings.magnitudes
    chem_mode = settings.chem_mode  # set to "hill" for reduced Hill-minimal H2O2-Ca mode
    ##Current settings don't cause any inhibition.
    ##Possible settings include 'AC' which inhibits production rates and
    ## 'GJ which inhibits transport. Unless you want on/off though you can obtain
    ## this effect by changing other parameters
    if not os.path.exists(path):
        os.makedirs(path)

    for c in magnitudes:
        new_path = path #+ '/n=' + str(c)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        #print("test")
        for _ in range(length):
            x=newDirect(new_path)
            mod.main(x,new_path,inh_val[0],inhib_type,inh_time,inh_time1, c, chem_mode=chem_mode)


if __name__ == '__main__':
    main()
