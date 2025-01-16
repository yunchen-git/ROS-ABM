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
    te = 50
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
    
    D_camp=444 ##um^2/sec diffusion coefficient for cyclic AMP Dworkin & Keller 1977 J. Bio Chem
    D_h2o2 = 444 ##um^2/sec diffusion coefficient for H2O2
    
    # # Read the graph using pickle
    # with open(os.path.join(os.getcwd(), "Colony.gpickle"), "rb") as f:
    #     test = pickle.load(f)
    
    # # Debugging statements
    # print(f"Type of net: {type(test)}")
    # print(f"Type of net.nodes(): {type(test.nodes())}")
    # #print(network)
    # #print(network.nodes())
    
    # # Inspect the graph's internal state
    # print(f"Graph nodes: {list(test.nodes(data=True))}")
    # print(f"Graph edges: {list(test.edges(data=True))}")
    
    # # Access nodes using list(net.nodes)
    # try:
    #     nodes_list = list(test.nodes(data=False))
    #     print(f"Nodes list: {nodes_list}")
    # except Exception as e:
    #     print(f"Error accessing nodes: {e}")
    #     return
    
    cell_path = os.getcwd() + os.sep + 'RA0_4.5.txt'
    cells = np.loadtxt(cell_path, delimiter=',')
    locations = cells[:,1:3]

    num = cells.shape[0]
    #print(f"Number of nodes: {num}")

    if te < fastest_division_time:
        maxCells = int(num * 2)
    else:
        maxCells= int(num * 2 ** ((te+1)/fastest_division_time)) # Calculating the maximum possible number of cells
    
    division_times = np.array([18, 51], dtype=np.int32)
    
    grid_dim=130.0                  #Grid dimension (NxNxN), (I'm pretty sure I stopped using this altogether...)

    x = 2000
    y = 2000
    z = 30
    ddim=2.0                        #Grid spacing in um
    sync_time=[500,500]             # This isn't used

    ## Initializing the intercellular diffusion module
    XI =IntC_Diff("cAMP",maxCells,D_camp,grid_dim,ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                  celltype_param,division_times,sync_time)
    sim.add_interC(XI)
    #print(f"interC value for sim: {sim.interC[0]}")

    H2O2 = ExtC_Diff("H2O2", D_h2o2, x, y, z, ddim)
    sim.add_gradient(H2O2)
    # dist=range(num)
    # r.shuffle(dist)
    #print(XI.vx)

    rand1 = np.arange(0, 10, 1)
    rand2 = np.arange(100, 110, 1)

    for i in range(0,num):
        ID=i
        point=[locations[i,0], locations[i,1], 0]
        radius=6.5 
        div_set=r.random()*19.

        sim_obj=StemCell(point,radius,ID,"U",inhib_value=inhib_value, division_set=div_set)   #create instance of StemCell class
        
        if i in rand1 or i in rand2:
            print(i)
            sim_obj.C = 1
            XI.change_C(ID, 1)
            #sim.interC[0].C[i]

        sim.add(sim_obj)
        sim.inc_current_ID()
        sim_obj.set_gradient_source_sink_coeff(H2O2.name, 2.0e-20, 1.8e-20)

        # print(f"object self.C: {ID} {sim_obj.C}")                           #StemCell object
        # print(f"XI self.C: {ID} {XI.get_C(ID)}")                            #IntC_Diff object
        # #print(f"len sim.interC[0].C: {ID} {len(sim.interC[0].C)}")          #IntC_Diff object
        # print(f"sim self.C: {ID} {np.array(sim.interC[0].C[i])}")
        #print(f"sim self.C: {ID} {sim.interC[0]}")                          #IntC_Diff object

    try:
        sim.collide()
    except Exception as e:
        print(f"Manual Collision due to: {e}")
        sim.collide_lowDens()
    
    sim.run()

  #set the seperator
    if(platform.system() == "Windows"):
        #windows
        sep = "\\"
    else:
        #linux/unix
        sep = "/"   


