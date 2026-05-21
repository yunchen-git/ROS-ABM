################################################################################
# File: Model_initialParameters.py
# Purpose:
#   Model “wiring” / initialization layer:
#   - Defines grid size, timestep, diffusion constants, and run-time settings.
#   - Loads initial cell locations (e.g., new_populated.txt) and instantiates
#     StemCell agents with initial intracellular states.
#   - Creates + registers extracellular gradients (ExtC_Diff) and intercellular
#     transport modules (IntC_Diff) with the Simulation object.
#
# Key public API:
#   - main(n, path, inhib_value, inhib_type, inhib_time, inhib_time1, c)
#
# How it fits in the codebase:
#   Called by Model_run.py for each run.
#   Produces a fully configured Simulation instance and starts sim.run().
#
# Inputs/Outputs:
#   Inputs:
#     - Reads initial cell coordinates from a local text file (CWD-relative).
#   Outputs:
#     - Writes a new run folder containing networks/gradients/time series via
#       Simulation.save() during the run loop.
#
# Notes / gotchas:
#   - Many parameters here are “experimental knobs”; keep names/semantics stable
#     because analysis scripts often assume specific gradient/species names.
################################################################################
from Simulation import *
from simulationObjects import *
import numpy as np
from IntC_diff import IntC_Diff
from ExtC_gradient import ExtC_Diff
import FBA_ode
import os
from dataclasses import dataclass, field

PERTURBATION_CELL_INDEX = 877

INTERC_SPECIES_CAMP = "cAMP"
INTERC_SPECIES_H2O2I = "H2O2i"
INTERC_SPECIES_CA = "Ca"
INTERC_SPECIES_ORDER = (
    INTERC_SPECIES_CAMP,
    INTERC_SPECIES_H2O2I,
    INTERC_SPECIES_CA,
)


def resolve_cell_path(cell_path=None, cwd=None):
    if cell_path is not None:
        return cell_path
    if cwd is None:
        cwd = os.getcwd()
    return cwd + os.sep + 'new_populated.txt'


@dataclass
class ModelSettings:
    ts: float = 0
    te: float = 300
    dt: float = 1
    x: float = 750
    y: float = 750
    z: float = 0
    ddim: float = 1.0
    sync_time: list = field(default_factory=lambda: [500, 500])
    num_cell_types: int = 2
    channel_count: float = 1000.0
    channel_permeability: float = 5e-9
    pluripotent_params: tuple = (4.89e-6, 0.03, 1.3e-7)
    differentiated_params: tuple = (6.0e-6, 0.04, 1.3e-7)
    open_probabilities: tuple = (0.45, 0.85)
    division_times_hours: tuple = (18, 51)
    D_camp: float = 444
    D_h2o2i: float = 1000
    D_h2o2e: float = 1800
    D_Ca: float = 760
    ddim_e: float = 1.0
    initial_h2o2e: float = 1e-9
    perturbation_cell_index: int = PERTURBATION_CELL_INDEX
    circle_radius: float = 50.0
    cell_radius: float = 6.5
    point_z: float = 0
    division_set: float = 0
    default_cell_state: str = "U"
    special_h2o2i: float = 1e-9
    special_camp: float = 1
    special_ca: float = 1e-7
    default_h2o2i: float = 1e-9
    default_camp_base: float = 0.595
    default_camp_random_scale: float = 0.05
    default_ca: float = 1e-7


def main(n,path,inhib_value,inhib_type,inhib_time,inhib_time1, c, chem_mode="full", cell_path=None):
    settings = ModelSettings()
    #Start time, end time, time step
    ts = settings.ts
    te = settings.te
    dt = settings.dt

    x = settings.x
    y = settings.y
    z = settings.z
    grid_size = [x, y, z]

    ddim = settings.ddim            #Grid spacing in um
    sync_time=settings.sync_time    # This isn't used
    FBA_ode.set_chem_mode(chem_mode)
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
    num_cell_types = settings.num_cell_types
  
    # Probability of a channel being open in each cell types
    open_prob=np.zeros([num_cell_types])

    #Number of active channels at interface
    N=settings.channel_count ## arbitrary value 
    #Intrinsic permeability of channel to diffusing substance
    Y=settings.channel_permeability  ## arbitrary value
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
    celltype_param[0,0]=settings.pluripotent_params[0]
    celltype_param[0,1]=settings.pluripotent_params[1]
    celltype_param[0,2]=int(c)
    celltype_param[0,3]=settings.pluripotent_params[2]
     
    open_prob[0]=settings.open_probabilities[0]
    ##Differentiated phenotype parameters
    celltype_param[1,0]=settings.differentiated_params[0]
    celltype_param[1,1]=settings.differentiated_params[1]
    celltype_param[1,2]=int(c)
    celltype_param[1,3]=settings.differentiated_params[2]
    
    open_prob[1]=settings.open_probabilities[1]
    
    division_times = np.array(settings.division_times_hours, dtype=np.int32)
    division_times = division_times * 60 * 60 # convert to minutes then seconds
    fastest_division_time= np.min(division_times) #18.0*60*60 #18 hours in seconds
    
    D_camp = settings.D_camp ##um^2/sec diffusion coefficient for cyclic AMP Dworkin & Keller 1977 J. Bio Chem
    D_h2o2i = settings.D_h2o2i ##um^2/sec diffusion coefficient for H2O2
    D_h2o2e = settings.D_h2o2e ##um^2/sec diffusion coefficient for H2O2 in the extracellular space 
    D_Ca = settings.D_Ca ##um^2/sec diffusion coefficient for Calcium
    
    resolved_cell_path = resolve_cell_path(cell_path=cell_path)
    cells = np.loadtxt(resolved_cell_path, delimiter=',')
    locations = cells[:,1:3]

    num = cells.shape[0]
    #print(f"Number of nodes: {num}")

    if te < fastest_division_time:
        maxCells = int(num * 2)
    else:
        maxCells= int(num * 2 ** ((te+1)/fastest_division_time)) # Calculating the maximum possible number of cells

    ## Initializing the intercellular diffusion module
    XI = IntC_Diff(INTERC_SPECIES_CAMP,maxCells,D_camp, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(XI)
    
    H2O2i = IntC_Diff(INTERC_SPECIES_H2O2I,maxCells,D_h2o2i, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(H2O2i)

    Ca = IntC_Diff(INTERC_SPECIES_CA,maxCells,D_Ca, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
                    celltype_param,division_times,sync_time)
    sim.add_interC(Ca)

    ddim_e = settings.ddim_e
    #H2O2e = ExtC_Diff("H2O2e", D_h2o2e, max_x + 0.01 * max_x, max_y + 0.01 * max_y, z, ddim)
    H2O2e = ExtC_Diff("H2O2e", D_h2o2e, x, y, z, ddim_e, settings.initial_h2o2e)
    sim.add_gradient(H2O2e)

    # GSH = IntC_Diff("GSH",maxCells,D_h2o2, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
    #                 celltype_param,division_times,sync_time)
    # sim.add_interC(GSH)

    # iCa = IntC_Diff("iCa",maxCells, 300, ddim,Pmax,open_prob,              #create instance of IntC_Diff class
    #                 celltype_param,division_times,sync_time)
    # sim.add_interC(iCa)

    # eCa = ExtC_Diff("eCa", 300, x, y, z, ddim)
    # sim.add_gradient(eCa)

    # superoxide_i = IntC_Diff("superoxide_i",maxCells,D_h2o2,grid_dim,ddim,Pmax,open_prob,              #create instance of IntC_Diff class
    #                 celltype_param,division_times,sync_time)
    # sim.add_interC(superoxide_i)

    # superoxide_e = ExtC_Diff("superoxide_e", D_h2o2, x, y, z, ddim)
    # sim.add_gradient(superoxide_e)

    # dist=range(num)
    # r.shuffle(dist)
    #print(XI.vx)

    # Define the center of the circle (you can change these coordinates)
    # Using cell 92's position as the center, or you can specify your own coordinates
    center_x = locations[settings.perturbation_cell_index, 0]  # or specify like 350.0
    center_y = locations[settings.perturbation_cell_index, 1]  # or specify like 350.0
    circle_radius = settings.circle_radius  # 50 μm radius for 100 μm diameter

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
        point[2] = settings.point_z#z/2  # Add some variation in z-axis
        radius=settings.cell_radius 
        div_set=settings.division_set#r.random()*19.*60*60 #randomly assign a division time between 0 and 19 hours

        if i == settings.perturbation_cell_index:#877:#1890:#9116:#1890: 
        #if i in high_h2o2_cells:  # Check if cell is in the high H2O2i circle
            sim_obj=StemCell(point,radius,ID,settings.default_cell_state,inhib_value=inhib_value, division_set=div_set)
            sim_obj.setC(settings.special_h2o2i,INTERC_SPECIES_H2O2I)
            sim_obj.setC(settings.special_camp,INTERC_SPECIES_CAMP)
            sim_obj.setC(settings.special_ca,INTERC_SPECIES_CA)
            #sim_obj.set_gradient_source_sink_coeff("H2O2e", 2.08e-12, 0.0)
            #sim.gradients[0].set_gradient_value_at_point(point, 5.25)
            #sim_obj.set_gradient_source_sink_coeff("eCa", 1e-4, 0.0)
            #H2O2i.change_C(ID, 1.0e-4)
        else:
            sim_obj=StemCell(point,radius,ID,settings.default_cell_state,inhib_value=inhib_value, division_set=div_set)   #create instance of StemCell class
            sim_obj.setC(settings.default_h2o2i,INTERC_SPECIES_H2O2I)
            sim_obj.setC(settings.default_camp_base+settings.default_camp_random_scale*c,INTERC_SPECIES_CAMP)
            sim_obj.setC(settings.default_ca,INTERC_SPECIES_CA)
            #sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)
        
        sim.inc_current_ID()
        #sim_obj.setC(3.68e-4,'GSH')

        # sim_obj.setC(0.0,'superoxide_i')
        sim_obj.set_gradient_source_sink_coeff("H2O2e", 0.0, 0.0)

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

