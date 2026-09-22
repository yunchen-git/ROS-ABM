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
from zread_gpickle import network_radius

def main(n,path,inhib_value,inhib_type,inhib_time,inhib_time1, c):
    #Start time, end time, time step
    ts = 0
    te = 600
    dt = 1

    x = 750
    y = 750
    z = 0
    grid_size = [x, y, z]

    ddim = 1.0                      #Grid spacing in um
    sync_time=[500,500]             # This isn't used
    #Now make a new simulation, instance of Simulation class
    
    sim = Simulation(n,
                     path,
                     ts,
                     te,
                     dt,
                     grid_size,
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
    
    division_times = np.array([18, 51], dtype=np.int32)
    division_times = division_times * 60 * 60 # convert to minutes then seconds
    fastest_division_time= np.min(division_times) #18.0*60*60 #18 hours in seconds
    
    D_camp = 444 ##um^2/sec diffusion coefficient for cyclic AMP Dworkin & Keller 1977 J. Bio Chem
    D_h2o2i = 1000 ##um^2/sec diffusion coefficient for H2O2
    D_h2o2e = 1800 ##um^2/sec diffusion coefficient for H2O2 in the extracellular space 
    D_Ca = 760 ##um^2/sec diffusion coefficient for Calcium
    
    cell_path = os.getcwd() + os.sep + 'cell_populations.txt' 
    cells = np.loadtxt(cell_path, delimiter=',')
    locations = cells[:,1:3]

    num = cells.shape[0]
    #print(f"Number of nodes: {num}")

    if te < fastest_division_time:
        maxCells = int(num * 2)
    else:
        maxCells= int(num * 2 ** ((te+1)/fastest_division_time)) # Calculating the maximum possible number of cells

    ## Initializing the intercellular diffusion module
    XI = IntC_Diff("cAMP",maxCells,D_camp, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(XI)
    
    H2O2i = IntC_Diff("H2O2i",maxCells,D_h2o2i, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(H2O2i)

    Ca = IntC_Diff("Ca",maxCells,D_Ca, ddim,10*Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(Ca)

    ddim_e = 1.0
    H2O2e = ExtC_Diff("H2O2e", D_h2o2e, x, y, z, ddim_e, 1e-9)
    sim.add_gradient(H2O2e)

    # Define the center of the circle (you can change these coordinates)
    # Using cell 92's position as the center, or you can specify your own coordinates
    center_x = locations[877, 0]  # or specify like 350.0
    center_y = locations[877, 1]  # or specify like 350.0
    circle_radius = 50.0  # 50 μm radius for 100 μm diameter

    # Find all cells within the circle
    high_h2o2_cells = []
    for i in range(num):
        cell_x = locations[i, 0]
        cell_y = locations[i, 1]
        distance = np.sqrt((cell_x - center_x)**2 + (cell_y - center_y)**2)
        if distance <= circle_radius:
            high_h2o2_cells.append(i)

    for i in range(0,num):
        ID=i
        c = rand.random()
        point=[locations[i,0], locations[i,1], 0]
        point[2] = 0#z/2  # Add some variation in z-axis
        radius=6.5 
        div_set=0#r.random()*19.*60*60 #randomly assign a division time between 0 and 19 hours

        if i == 877:#877:#1890:#9116:#1890: 
        #if i in high_h2o2_cells:  # Check if cell is in the high H2O2i circle
            sim_obj=StemCell(point,radius,ID,"U",inhib_value=inhib_value, division_set=div_set)
            sim_obj.setC(1e-9,'H2O2i')
            sim_obj.setC(1,'cAMP')
            sim_obj.setC(1e-7,'Ca')
  
        else:
            sim_obj=StemCell(point,radius,ID,"U",inhib_value=inhib_value, division_set=div_set)   #create instance of StemCell class
            sim_obj.setC(1e-9,'H2O2i')
            sim_obj.setC(0.595+0.05*c,'cAMP')
            sim_obj.setC(1e-7,'Ca')
        
        sim.inc_current_ID()
        sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)
        sim.add(sim_obj)
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


