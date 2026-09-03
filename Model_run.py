import Model_initialParameters as mod
from CountDirectory import newDirect
import sys
import getpass
import os
import numpy as np


if __name__ == '__main__':
    length=1
    user=getpass.getuser()
    
    ##Set save path for simulations. If you create a new, empty folder 
    ##then everytime you run this code it will make a new Simulation folder
    path=os.path.expanduser('~/Documents/PhD/test')

    inh_val=[1.0]
    per_inh=[1.0]
    inh_time=1000
    inh_time1=1000
    inhib_type=None
    magnitudes = [1]
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
        for i in range(length):
            x=newDirect(new_path)
            mod.main(x,new_path,inh_val[0],inhib_type,inh_time,inh_time1, c)
