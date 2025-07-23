import sys
import random as r
from Simulation import *
from simulationObjects import *
import networkx as nx
import numpy as np
import pickle
import platform
from IntC_diff import IntC_Diff
from ExtC_gradient import ExtC_Diff
from CountDirectory import newDirect
from read_gpickle import network_radius

def main(x,path,inhib_value,inhib_type,inhib_time,inhib_time1, c):
    #Start time, end time, time step
    ts = 0
    te = 60
    dt = 1

    #Now make a new simulation, instance of Simulation class
    
    sim = Simulation(x,
                     path,
                     ts,
                     te,
                     dt,
                     inhib_type,
                     inhib_time=inhib_time,
                     inhib_time1=inhib_time1)

    #Parameters for intercellular diffusion
    #Index 0 represents a pluripotent cell
    #Index 1 represents a differentiated cell
    num_cell_types = 2
  
    # Probability of a channel being open in each cell types
    open_prob=np.zeros([num_cell_types])

    #Number of active channels at interface
    N=1000. ## arbitrary value 
    #Intrinsic permeability of channel to diffusing substance
    Y=5e-9  ## arbitrary value
    Pmax=Y*N
    ## Each cell type has a set of production and degradation parameters
    ## First 3 represent production following an inhibition curve (1-Hill) to represent a decrease in
    ## production when the cell is already at a high level 
    # a,the max_prod or plateau of curve, index 0
    # k, coefficient representing when 50% of max_prod occurs, index 1
    # n, the steepness of the curve (represents cooperation in hill function), index 2
    ## Degradation rate is Index 3, with the cell degrading this percent of its current concentration each time step 
    celltype_param=np.zeros([num_cell_types,4])
    
    #Pluripotent phenotype parameters
    celltype_param[0,0]=4.89e-6
    celltype_param[0,1]=0.03
    celltype_param[0,2]=int(c)
    celltype_param[0,3]=1.3e-7
     
    open_prob[0]=0.45
    ##Differentiated phenotype parameters
    celltype_param[1,0]=6.0e-6
    celltype_param[1,1]=0.04
    celltype_param[1,2]=int(c)
    celltype_param[1,3]=1.3e-7
    
    open_prob[1]=0.85
    
    fastest_division_time=18.0
    
    D_camp = 444 ##um^2/sec diffusion coefficient for cyclic AMP Dworkin & Keller 1977 J. Bio Chem
    D_h2o2i = 2500 ##um^2/sec diffusion coefficient for H2O2
    D_h2o2e = 2500 ##um^2/sec diffusion coefficient for H2O2 in the extracellular space 
    
    cell_path = os.getcwd() + os.sep + 'new_populated.txt' #'RA0_4.5.txt'
    cells = np.loadtxt(cell_path, delimiter=',')
    locations = cells[:,1:3]

    num = cells.shape[0]
    #print(f"Number of nodes: {num}")

    if te < fastest_division_time:
        maxCells = int(num * 2)
    else:
        maxCells= int(num * 2 ** ((te+1)/fastest_division_time)) # Calculating the maximum possible number of cells
    
    division_times = np.array([18, 51], dtype=np.int32)
    division_times = division_times * 60 # convert to minutes
    
    x = 750
    y = 750
    z = 30
    ddim=1.0                      #Grid spacing in um
    sync_time=[500,500]             # This isn't used

    ## Initializing the intercellular diffusion module
    XI = IntC_Diff("cAMP",maxCells,D_camp, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(XI)
    
    H2O2i = IntC_Diff("H2O2i",maxCells,D_h2o2i, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(H2O2i)

    H2O2e = ExtC_Diff("H2O2e", D_h2o2e, x, y, z, ddim)
    sim.add_gradient(H2O2e)

    # GSH = IntC_Diff("GSH",maxCells,D_h2o2, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
    #                 celltype_param,division_times,sync_time)
    # sim.add_interC(GSH)

    iCa = IntC_Diff("iCa",maxCells, 300, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(iCa)

    eCa = ExtC_Diff("eCa", 300, x, y, z, ddim)
    sim.add_gradient(eCa)

    # superoxide_i = IntC_Diff("superoxide_i",maxCells,D_h2o2,grid_dim,ddim,Pmax,open_prob,              #create instance of IntC_Diff class
    #                 celltype_param,division_times,sync_time)
    # sim.add_interC(superoxide_i)

    # superoxide_e = ExtC_Diff("superoxide_e", D_h2o2, x, y, z, ddim)
    # sim.add_gradient(superoxide_e)

    # dist=range(num)
    # r.shuffle(dist)
    #print(XI.vx)

    rand1 = np.arange(0, 10, 1)
    rand2 = np.arange(200, 201, 1)

    for i in range(0,num):
        ID=i
        point=[locations[i,0], locations[i,1], 0]
        radius=6.5 
        div_set=r.random()*19.*60 #randomly assign a division time between 0 and 19 hours

        if i == 92:#1890:#9116:#1890: 
            sim_obj=StemCell(point,radius,ID,"U",inhib_value=inhib_value, division_set=div_set)
            sim_obj.setC(5.25,'H2O2i')
            sim_obj.setC(0.595+0.05*x,'cAMP')
            sim_obj.setC(0, 'iCa')
            #sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)
            #sim.gradients[0].set_gradient_value_at_point(point, 5.25)
            #sim_obj.set_gradient_source_sink_coeff("eCa", 1e-4, 0.0)
            #H2O2i.change_C(ID, 1.0e-4)
        else:
            sim_obj=StemCell(point,radius,ID,"U",inhib_value=inhib_value, division_set=div_set)   #create instance of StemCell class
            sim_obj.setC(1.0e-9,'H2O2i')
            sim_obj.setC(0.595+0.05*x,'cAMP')
            sim_obj.setC(0.0,'iCa')
            #sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)
        
        sim.inc_current_ID()
        x = rand.random()
        #sim_obj.setC(3.68e-4,'GSH')

        # sim_obj.setC(0.0,'superoxide_i')
        sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)
        sim_obj.set_gradient_source_sink_coeff("eCa", 0.0, 0.0)

        sim.add(sim_obj)

        # print(f"object self.C: {ID} {sim_obj.C}")                           #StemCell object
        # print(f"XI self.C: {ID} {XI.get_C(ID)}")                            #IntC_Diff object
        # #print(f"len sim.interC[0].C: {ID} {len(sim.interC[0].C)}")          #IntC_Diff object
        # print(f"sim self.C: {ID} {np.array(sim.interC[0].C[i])}")
        #print(f"sim self.C: {ID} {sim.interC[0]}")                          #IntC_Diff object

    try:
        sim.collide(x,y,z)
    except Exception as e:
        print(f"Manual Collision due to: {e}")
        sim.collide_lowDens()
    
    sim.run(x,y,z)

  #set the seperator
    if(platform.system() == "Windows"):
        #windows
        sep = "\\"
    else:
        #linux/unix
        sep = "/"   


