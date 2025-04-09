################################################################################
# Name:   SimulationObjects
# Author: Douglas E. White
# Date:   10/17/2013
################################################################################
from simulationMath1 import *
import random as rand
import math as math
import numpy as np
class SimulationObject(object):
    """ Base class from which all simulation obejcts must inherit
    """

    def __init__(self, location, radius, ID ,owner_ID, sim_type):
        """ Base class which defines properties all sim objects MUST have
            location - the location of the sphere
            radius - the radius of the sphere
            ID - the ID of the object #WARNING# This ID is used ot hash objects
                  it MUST BE UNIQUE
            owner_ID - usually the same as the ID, also must be unique between
                       agents, unless all agents are part of a larger group
                       i.e. this is the mechanism for multi-agent agents
            sim_type - the type of object the simulation object is
        """
        self.location = location
        self.distance=0
        
        self.C={}#0.595+0.05*rand.random()
        # self.C['cAMP']=0.595+0.05*rand.random()
        # self.C['H2O2i']=1.0e-9

        self.thresCount=0
        self.influ=0
        
        self.radius = radius
        self.sim_type = sim_type
        self.ID = ID
        self.owner_ID = owner_ID
        self.z=0
        #keep track of the opt and col vecs
        #print(len(self.location))
        if len(self.location)==3:
            self._disp_vec = [0,0,0]
            self._fixed_contraint_vec = [0,0,0]
            self._v = [0,0,0]
        else:
            self._disp_vec = [0,0]
            self._fixed_contraint_vec = [0,0]
            self._v = [0,0]
        
        #keep track of production consumptions values
        self.gradient_source_sink_coeff = dict()
        #keep track of the relative indices in the gradient array
        self.gradient_position = dict()
        #keep track of the value of the gradient associated with these agents
        self.gradient_value = dict()


    def update(self, sim, dt):
        """ Updates the simulation object
        """
        pass

    def setC(self,C,name):
        self.C[name]=C

    def update_d(self, sim):
        nbs=sim.network.neighbors(self)
        for i in range(0,len(nbs)):
            nbs[i].d+=1
    def set_extra_C(self,C):
        self.extra_C=C
        
    def splitC(self,C):
        self.C=C
  
    def get_max_interaction_length(self):
        """ Get the max interaction length of the object
        """
        return self.radius*2.0 #in um

    def get_interaction_length(self):
        return self.radius #in um

    def get_spring_constant(self, other):
        """ Gets the spring constant of the object
            Returns: 1.0 by default
            NOTE: Meant to be overwritten by a base class if more
                  functionality is required
        """
        return 0.77

    def add_displacement_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._disp_vec = AddVec(self._disp_vec, vec)

    def add_fixed_constraint_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._fixed_contraint_vec = AddVec(self._fixed_contraint_vec, vec)

    def set_gradient_source_sink_coeff(self, name, source, sink):
        """ Adds a production/consumption terms to the dicationary based on
            the gradient name
        """
        #overwrite exsisting data
        self.gradient_source_sink_coeff[name] = (source, sink)

    def get_gradient_source_sink_coeff(self, name):
        """ Gets the graident terms (source, sink) for the given gradient name
            returns - a tuple of the (source, sink) values. If these are not
                      in the dictionary, returns (0,0)
        """
        #returns the gradient values for the source and sink
        if(name in self.gradient_source_sink_coeff.keys()):
            #name is in the dictionary
            #print(len(self.gradient_source_sink_coeff[name]))
            return self.gradient_source_sink_coeff[name]
        else:
            #the name is not in the dictionary
            return (0,0)

    def set_gradient_location(self, name, location):
        """ Adds the location of the agent on the grid for the gradient
            whose name is specified by name
        """
        self.gradient_position[name] = location

    def get_gradient_location(self, name):
        """ Return the location of the agent on the grid for the gradient
            specified by the name, name
        """
        if(name in self.gradient_position.keys()):
            #name is in the dictionary
            return self.gradient_position[name]
        else:
            #the name is not in the dictionary
            return None

    def set_gradient_value(self, name, value):
        """ Adds the value of the gradient at this agent
        """
        self.gradient_value[name] = value

    def get_gradient_value(self, name):
        """ Return the value fo the gradient at this agent
        """
        if(name in self.gradient_value.keys()):
            #name is in the dictionary
            return self.gradient_value[name]
        else:
            #the name is not in the dictionary
            return -1
        
    def update_constraints(self, dt, scale=5.0):
        """ Updates all of the contraints on the object
        """
        #first update the posiiton by the col and opt vectors
        #make sure neither of these vectors is greater than error
        mag = Mag(self._disp_vec)
        if(mag > 5):
            n = NormVec(self._disp_vec)
            self._disp_vec = ScaleVec(n, scale)
        self.location = AddVec(self.location, self._disp_vec)
        if self.location[2]<0:
                self.location[2]=0

        #then clear it
        dim=len(self.location)
        if dim==3:
            self._disp_vec = [0,0,0]
        else:
            self._disp_vec = [0,0]
        #then update the the pos using the fixed vectors
        mag = Mag(self._fixed_contraint_vec)
        if(mag > 5):
            n = NormVec(self._disp_vec)
            self._fixed_contraint_vec = ScaleVec(n, scale)
        self.location = AddVec(self.location, self._fixed_contraint_vec)
        if self.location[2]<0:
            self.location[2]=0
        #htne clear it
        if dim==3:
            self._fixed_contraint_vec = [0,0,0]
        else:
            self._fixed_contraint_vec = [0,0]
        
        
    def __repr__(self):
        """ Returns a string representation of the object
        """
        return self.sim_type+": "+repr(self.ID)+" "+repr(self.location)

    def __eq__(self, other):
        """ Handles the equal operator for the object
        """
        if(isinstance(other, SimulationObject)):
            return self.ID == other.ID
        #otherwise
        return False

    def __hash__(self):
        """ Handles the hashing operator for the object
        """
        return hash(self.ID)


class StemCell(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID, state, inhib_value=1.0,
                 division_set = 0.0,
                 division_time = 19.0,
                 owner_ID = None):
        """ Constructor for a stem cell
            location - the location fo the stem cell
            radius - the size of the stem cell
            ID - the unique ID for the agent
            state - the state of the stem cell
            division_set - the initial division set for the cell
            division_time - the time it takes the cell to divide
            owner_ID - the ID associated with the owner of this agent
        """
        #define some variables
        if(owner_ID == None):
            owner_ID = ID
        #set thet state
        self.state = state
        self.inhib_value=inhib_value
        self.division_timer = division_set
        self.division_time = division_time
        #print(owner_ID, self.division_timer)
        #call the parent constructor
        super(StemCell, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "stemcell")

    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        # Store concentration in stem cell object from the intercellular diffusion module
        for interC in sim.interC:
            #print(f"for {self.ID} getting C {interC.name}, {interC.get_C(self.ID)}")
            self.C[interC.name]=interC.get_C(self.ID)
        #growth kinetics
        self.division_timer += dt

        # #you can grow unless you are in the A state meaning apoptosis
        # if(self.division_timer >= self.division_time):
        #     n=2
        #     #get the radius
        #     radius = self.radius
        #     #Location of daughter cell 
        #     location = RandomPointOnSphere(n)*radius/2.0 + self.location
            
        #     #get the ID
        #     ID = sim.get_ID()
        #     #Split intracellular concentration between daughter cells
        #     splitC = {key: float(value) / 2 for key, value in self.C.items()}
        #     #print(f"splitting {self.ID}, {splitC}")

        #     for interC in sim.interC:
        #         #print(f"for {self.ID} splitting C {interC.name}, {-splitC[interC.name]}")
        #         interC.add_to_dC(self.ID,-splitC[interC.name])
            
        #     #Make object, copy over all of the coefficients to the new cells 
        #     sc = StemCell(location, radius, ID, self.state, inhib_value=self.inhib_value,
        #                   division_time = self.division_time)
        #     for key, value in splitC.items():
        #         sc.setC(value, key)

        #     #print(f"adding {sc.ID}, {sc.C}, because of {self.ID}, {self.C}")
        #     #add it to the imsulation
        #     sim.add_object_to_addition_queue(sc)
        #     #reset the division time
        #     self.division_timer =0

        #     for interC in sim.interC:
        #         interC.restart_div_cycle(self.ID)

        # #STATE DEPENDANT BEHAVIOR
        # if(self.state == "A"):
        #     #self.set_gradient_source_sink_coeff("H2O2e", 5e-9, 0)
        #     print(f"removing {self.ID}")
        #     sim.add_object_to_remnoval_queue(self)
        self.division_timer = 0 
        #STATE DEPENDANT BEHAVIOR
        if(self.state == "T"):
            #change the current sytate to D
            self.state = "D"
            self.update_d
            #self.set_gradient_source_sink_coeff("D", 0, 1e-9)
            self.division_time = 51 #in hours

        #Concentration threshold influencing differentiation; chance to differentiation increases the longer a cell has
        # an elevated intracellular concentration
        threshold=0.74
        #print(self.kth)
        if self.C['cAMP']>threshold:
            #print(f"self.C['cAMP'] {self.C['cAMP']}")
            ni=4.
            ki=2.0
            self.thresCount+=1
            tC=self.thresCount
            self.influ= tC**ni/(ki**ni+tC**ni)
            #print(ki)
        elif self.C['cAMP'] <=threshold and self.influ>0:
            ni=4.
            ki=2.0
            self.thresCount-=1
            if self.thresCount<0:
                self.thresCount=0
            tC=self.thresCount
            self.influ= tC**ni/(ki**ni+tC**ni)

        if(self.state == "U"):          #then the stem cell is still a stem cell
            #HANDLE DIFFERENTIATION

            #RANDOM RULE
            x = rand.random()
            t=sim.time
            # Base rate with oscillation associated with RA dosing
            prob = 0.002+abs(0.001*math.sin((t+1.)/-8.)) #used to be 0.0025 but we need it to take slightly

            if(x < prob):
                #differentiation occurs
                print("diff")
                #Transition state lasts an hour
                #(currently treated as identical to diff state) 
                self.state = "T"
                self.diff="rand"
                #self.set_gradient_source_sink_coeff("H2O2e", 0.0, 2.5e-9)
                sim.interC[0].change_state(self.ID,1) #Change state in Intercellular Diffusion Module
                sim.interC[1].change_state(self.ID,1) #Change state in Intercellular Diffusion Module

            if self.state=="U":
                #Influence of intercellular concentration on the probability of differentiation    
                x=rand.random()
                if x<self.influ:
                    print("C Diff")
                    self.state="T"
                    #self.set_gradient_source_sink_coeff("H2O2e", 0.0, 2.5e-9)
                    sim.interC[0].change_state(self.ID,1) #Change state in Intercellular Diffusion Module
                    sim.interC[1].change_state(self.ID,1) #Change state in Intercellular Diffusion Module

        # if self.state=="D" or self.state=="T" or self.state=="U":
        #     val = self.get_gradient_value("H2O2e")
        #     x = rand.random()
        #     prob_extC = (1.0*val)/(0.0001+val)
        #    # print(val, prob_extC,x)
        #     t=sim.time
        #     prob = 0.002+abs(0.001*math.sin((t+1.)/-8.))
        #     if x<prob_extC:
        #         #Apoptosis
        #         self.state = "A"
        #         print(f"feroptosis {self.ID}")
        #         #self.set_gradient_source_sink_coeff("H2O2e", 5e-9, 0.0)
        #         sim.interC[0].change_state(self.ID,2)
        #     if x<0.1 * prob:
        #         self.state = "A"
        #         print(f"apoptosis {self.ID}")
        #         #self.set_gradient_source_sink_coeff("H2O2e", 5e-9, 0.0)
        #         sim.interC[0].change_state(self.ID,2)


    def __hash__(self):
        return hash(self.ID)
    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius+1.5  #in um
