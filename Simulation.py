################################################################################
# Name:   Simulation v 3.0
# Author: Douglas E. White
# Date:   10/16/2013
################################################################################
import os, sys, shutil
import platform
import networkx as nx
import numpy as np
from simulationMath1 import *
from simulationObjects import *
from IntC_diff import IntC_Diff
from ExtC_gradient import ExtC_Diff
from scipy.spatial import *
from matplotlib import pyplot as plt
import pickle
import math as math
import random as rand
import time
from read_gpickle import read_network
from gradientImage import getGradientImages
#create the time series class
class TimeSeries(object):
    """ A TimeSeries class meant to hold all of the information about
        a given simulation data set
    """

    def __init__(self, path):
        """ Initilization function for the time series setup
            path - the directory holding the simulation information
            ERRORS - TypeError if path is not a string
        """
        #set some base params
        self._networks = dict()
        self._gradients = dict()
        self._interC=dict()
        
        #check types
        if(type(path) is not str):
            raise TypeError("The path argument must be of type string")
        self.path = path
        #make sure path exsists
        if(not(os.path.exists(path))):
            raise IOError("""The path specified either does not exist, or you
                            don not have the privledges to acess it.""")       

        #keep track of the fuile sepeartor to use
        if(platform.system() == "Windows"):
            #windows
            self.__sep = "\\"
        else:
            #linux/unix
            self.__sep = "/"
        #load the actual file
        self._load()

    def _load(self):
        """ Takes care of laoding the simulation header file
        """
        #try to load
        try:
            f = open(self.path + self.__sep + "info.sim", "r")
        except IOError:
            raise IOError("""Simulation header file not found""")
        #ok now the file has been loaded
        #get the sim name
        line = f.readline()
        data = line.split(":")
        self.name = data[1]
        #get the base path
        line = f.readline()
        data = line.split(":")
        base_path = data[1]
        if(base_path != self.path):
            base_path = self.path
        #get the simulation seperator
        line = f.readline()
        data = line.split(":")
        data = data[1].strip("\n")
        if(data == "Windows"):
            self._sep = "\\"
        else:
            self._sep = "/"
        #get the time range
        line = f.readline()
        data = line.split(":")
        data = data[1].split(",")
        start_time = float(data[0])
        end_time = float(data[1])
        time_step = float(data[2])
        #get the gradient names
        line = f.readline()
        data = line.split(":")
        data = data[1].strip("\n")
        data = data.split(",")
        if(data[0] == "" or data[0] == " "):
            self._grad_names = []
        else:
            self._grad_names = data
        line = f.readline()
        data = line.split(":")
        data = data[1].strip("\n")
        data = data.split(",")
        if(data[0] == "" or data[0] == " "):
            self._intC_names = []
        else:
            self._intC_names = data
        #now for all the ranges in time loop over the list
        num_times = int((end_time - start_time) / time_step) + 1 #inclusive
        for i in range(0, num_times):
            #read the line
            line = f.readline()
            if(line != ""):
                #get the time
                data = line.split(",")
                time = float(data[0])
                #get the network path
                n_path = data[1]
                #make sure no new line characters on this'
                n_path = n_path.strip("\n")
                #rebuild the path
                paths = n_path.split(self._sep)
                #just take the last part
                path = paths[-1]
                #then add it to the path
                n_path = self.path + self.__sep + path
                #associate it with the time
                self._networks[time] = n_path
                #get all the gradient paths
                grad_paths = dict()
                intC_paths=dict()
                n1=len(self._grad_names)
                n2=len(self._intC_names)
                for j in range(0, n1):
                    #add this path to the grad path list
                    g_path = data[j+2].strip("\n")
                    #rebuild the path
                    paths = g_path.split(self._sep)
                    #just take the last part
                    path = paths[-1]
                    #then add it to the path
                    g_path = self.path + self.__sep + path
                    grad_paths[self._grad_names[j]] = g_path
                for j in range(0, n2):
                    #add this path to the grad path list
                    g_path = data[j+n1+2].strip("\n")
                    #rebuild the path
                    paths = g_path.split(self._sep)
                    #just take the last part
                    path = paths[-1]
                    #then add it to the path
                    g_path = self.path + self.__sep + path
                    intC_paths[self._grad_names[j]] = g_path
                self._gradients[time] = grad_paths
                self._interC[time]=intC_paths
        #once this is done close this file
        f.close()

    def get_raw_agent_data(self):
        """ Return all fo the networks X objects at once in an ordered list
        """
        time_series = []
        for i in range(0, len(self._networks.keys())):
            time = self._networks.keys()[i]
            #print(time)
            tp  = self.get_time_point(time)
            time_series.append(tp)
        return time_series

  
    def get_times(self):
        return self._networks.keys()

  
    def get_interC(self):
        """ Returns a list of all the gradient names
        """
        return self._intC_names
        
    def get_time_point(self, time):
        """ Loads the specific network from the time point and returns it
            returns - network file
        """
        with open(self._networks[time], 'rb') as f:
            return pickle.load(f)

    def get_gradient_at_time(self, time, name):
        """ Loads the gradient data specified by the specific time
            and name
        """
        #print(self._gradients)
        #print(self._gradients[time][name])
        f = open(self._gradients[time][name], "rb")
        g = pickle.load(f)
        f.close()
        return g

    def get_intC_at_time(self, time, name):
        """ Loads the gradient data specified by the specific time
            and name
        """
        #print(self._interC[time][name])
        f = open(self._interC[time][name], "rb")
        g = pickle.load(f)
        f.close()
        return g

    def convert_to_data_base(self):
        """ 
        """
        #use can use the dict method here

    def __eq__(self, other):
        """ Handles the equal operator for the object
        """
        if(isinstance(other, TimeSeries)):
            return self.name == other.name
        #otherwise
        return False

    def __hash__(self):
        """ Handles the hashing operator for the object
        """
        return hash(self.name)
        
#Create a new simulation class
class Simulation(object):
    """ A class for running simulations
    """

    def __init__(self, name, path, start_time, end_time, time_step,inhib_type, inhib_time=1.0,inhib_time1=24.0,inhib_duration=6.0):
        """ Initialization function for the simulation setup.
            name - the simulation name (string)
            path - the path to save the simulation information to (string)
            start_time - the start time for the simulation (float)
            end_time - the end time for the simulation (float)
            time_step - the time step to increment the simulation by (float)
        """
        #set the base parameters
        #do some basic type checking
        if(type(name) is str):
            self.name = name
        else:
            self.name = repr(name)
        if(type(path) is str):
            self.path = path
        else:
            self.path = repr(path)
        #now convert all fo these to float
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.time_step = float(time_step)
        self.time=float(start_time)
        self.inhib_time=inhib_time
        self.inhib_time1=inhib_time1
        self.inhib_duration=inhib_duration
        self.inhib_type=inhib_type

        #make a list to keep track of the sim objects
        self.objects = []
        
        #list of object concentration gradients
        self.interC=[]
        self._intC_by_name=dict()
        #also the gradients
        self.gradients = []
        self._gradients_by_name = dict()
        #keep track of the fixed constraints
        self._fixed_constraints = nx.Graph()
        self.network = nx.Graph()
        #add the add/remove buffers
        self._objects_to_remove = []
        self._objects_to_add = []

        #also keep track of the current sim object ID
        self._current_ID = 0

        #keep track of the fuile sepeartor to use
        if(platform.system() == "Windows"):
            #windows
            self._sep = "\\"
        else:
            #linux/unix
            self._sep = "/"

    def add(self, sim_object):
        """ Adds the specified object to the list
        """
        if(isinstance(sim_object, SimulationObject)):
            self.objects.append(sim_object)
            #also add it to the network
            self.network.add_node(sim_object)
            if sim_object.state=="U":
                state=0
            else:
                state=1

            for i in range(len(self.interC)):
                if self.interC[i].name =='cAMP':
                    self.interC[i].add_cell(sim_object.ID,sim_object.C[self.interC[i].name],state,sim_object.division_timer,1.0,1.0)
                else:
                    self.interC[i].add_cell(sim_object.ID,sim_object.C[self.interC[i].name],state,sim_object.division_timer,1.0,1.0)
                #print(f"sim interC[0] value: {self.interC[i]}")

    def inc_current_ID(self):
        self._current_ID += 1
   
    def remove(self, sim_object):
        """ Removes the specified object from the list
        """
        self.objects.remove(sim_object)
        #remove it from the network
        self.network.remove_node(sim_object)
        #also remove it fomr the fixed network
        try:
            self._fixed_constraints.remove_node(sim_object)
        except nx.NetworkXError:
            pass

    def add_object_to_addition_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be added to the simulation at the end of
            the update phase.
        """
        self._objects_to_add.append(sim_object)
        #increment the sim ID
        self._current_ID += 1
    def add_inhb_objects(self,obj_list):
        self.inhb_objects=obj_list
        
    def add_object_to_remnoval_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be removed from the simulation at the end of
            the update phase.
        """
        #place the obejct in the removal queue
        self._objects_to_remove.append(sim_object)

    def add_gradient(self, gradient):
        """ Adds gradients to the simulation
        """
        if(isinstance(gradient, ExtC_Diff)):
            self.gradients.append(gradient)
            self._gradients_by_name[gradient.name] = gradient
    
    def add_interC(self, interC):
        """ Adds gradients to the simulation
        """
        if(isinstance(interC, IntC_Diff)):
            self.interC.append(interC)
            self._intC_by_name[interC.name] = interC
            #print(f"sim interC[0] value: {self.interC[0]}")

    def remove_gradient(self, gradient):
        """ Removes a gradient form the simulation
        """
        self.gradients.remove(gradient)
        del self._gradients_by_name[gradient.name]

    def add_fixed_constraint(self, obj1, obj2):
        """ Adds a fixed immutable constraint between two objects which is
            processed with the other optimization constraints
        """
        self._fixed_constraints.add_edge(obj1, obj2)

    def remove_fixed_contraint(self, obj1, obj2):
        """ Removes a fixed immutable constraint between two objects
        """
        self._fixed_constraints.remove_edge(obj1, obj2)

    def get_ID(self):
        """ Returns the current unique ID the simulation is on
        """
        #return the next avilable ID
        return self._current_ID

    def get_gradient_by_name(self, name):
        """ Returns the specific gradient by it's name
        """
        return self._gradients_by_name[name]
            
    def get_intC_by_name(self, name):
        return self._intC_by_name[name]

    def cellProfiler_fix(self,n):
        net=self.network
        agents=self.objects
        clusters=nx.connected_components(net)
        clust_size=0
        for i in range(len(clusters)):
            if i!= n:
                clust=clusters[i]
                for j in range(len(clust)):
                    self.remove(clust[j])
        self.update_object_queue()
                    
        
    def run(self):
        """ Runs the simulation until either the stopping criterio is reached
            or the simulation time runs out.
        """
        self.time = self.start_time

        #try to make a new directory for the simulation
        try:
            os.mkdir(self.path + self._sep + self.name)
        except OSError:
            #direcotry already exsists... overwrite it
            print("Directory already exsists... overwriting directory")
        struct_path=os.getcwd()
        
        final_path=self.path + self._sep + self.name+self._sep+"copiedTM"
        try:
            shutil.copytree(struct_path,final_path)
        except OSError:
            print("Replacing Model files...hope you wanted that.")
            shutil.rmtree(final_path)
            shutil.copytree(struct_path,final_path)
        #create the simulation header file
        self._create_header_file()
##        #save the inital state configuration
        self.save()
        self._set_gradient_inital_conditions()
        #nor run the loop
        while(self.time <= self.end_time):
            print("Time: " + repr(self.time))
            print("Number of objects: " + repr(len(self.objects)))
            # print("Max C value: " + repr(max([x.C for x in self.objects])))
            # print("Min C value: " + repr(min([x.C for x in self.objects])))
            # print("average C value: " + repr(sum([x.C for x in self.objects])/len(self.objects)))

            #Update the objects and gradients
            self.update()
            #remove/add any objects
            self.update_object_queue()
            #perform physcis
            try:
                self.collide()
            except:
                print("manual collide")
                self.collide_lowDens()
                #now optimize the resultant constraints
            self.optimize()
            
            # for i in range(len(self.objects)):                      #checking the C values of the classes
            #     print(f"{i} sim_obj C: " + repr(self.objects[i].C), "interC C: " + repr(self.interC[0].C[i]))

            #increment the time
            self.time += self.time_step
            #save 
            self.save()
        #once the sim is done close the header file
        self._header.flush()
        self._header.close()

        read_network(self.path + self._sep + self.name + self._sep, int(self.start_time), int(self.end_time))
        
        # getGradientImages(self.path + self._sep + self.name + self._sep, int(self.start_time), int(self.end_time))

    def update_object_queue(self):
        """ Updates the object add and remove queue
        """
        print("Adding " + repr(len(self._objects_to_add)) + " objects...")
        print("Removing " + repr(len(self._objects_to_remove)) + " objects...")
##        print("Removing " + repr(len(self._objects_to_remove)) + " objects...")
        for i in range(0, len(self._objects_to_remove)):
            self.remove(self._objects_to_remove[i])
        for i in range(0, len(self._objects_to_add)):
            self.add(self._objects_to_add[i])
        #then clear these lists
        self._objects_to_remove = []
        self._objects_to_add= []
        

    def update(self):
        """ Updates all of the objects in the simulation
        """
        #update the gradients
        agents=self.objects
        n=len(agents)
        
        """
        Gap Junction inhibition (GJ) or Adenylyl Cyclase inhibition (AC)
        """
        if self.inhib_type=='GJ':
            if self.time==self.inhib_time:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_GJ(agent.ID,(1.0-agent.inhib_value)/2.)
            elif self.time==self.inhib_time+1:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_GJ(agent.ID,agent.inhib_value)
            elif self.time==self.inhib_time+self.inhib_duration-1:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_GJ(agent.ID,(1.0-agent.inhib_value)/2.)
            elif self.time==self.inhib_time+self.inhib_duration:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_GJ(agent.ID,1.0)
        elif self.inhib_type=='AC':
            if self.time==self.inhib_time:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_AC(agent.ID,agent.inhib_value)
            elif self.time==self.inhib_time+self.inhib_duration:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_AC(agent.ID,1.0)
            if self.time==self.inhib_time1:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_AC(agent.ID,agent.inhib_value)
            elif self.time==self.inhib_time1+self.inhib_duration:
                for i in range(len(agents)):
                    agent=agents[i]
                    self.interC[0].change_inhib_AC(agent.ID,1.0) 
 
        #define the sinks and sources
        print("Updating gradients...")
        t0=time.time()
        if(len(self.gradients) > 0):
            sources, sinks = self._define_sinks_and_sources()
        
        #print(self.objects)
        #update the gradients
        #print(f"gradients length {len(self.gradients)}")
        for i in range(0, len(self.gradients)):
            self.gradients[i].update(1, sinks[i], sources[i])
            self.gradients[i].cubic_interpolate_at_object_locations(self.objects)
            
            #print(self.gradients[i])
            print("Sources: " + repr(np.max(sources[i])) + " , " +repr(np.min(sources[i])))
            print("Sinks: " + repr(np.max(sinks[i])) + " , " + repr(np.min(sinks[i])))
        t1=time.time()
        print ("Gradient Diffusion Time: {}".format(t1-t0))
        #Update nieghbor matrix for intercellular module
        nbs1=[]
        edges = self.network.edges()      
        for edge in edges:
            #first find the distance of the edge
            obj1 = edge[0]
            obj2 = edge[1]
            i=obj1.ID
            j=obj2.ID
            nbs1.append([i,j])
        nbs2=np.asarray(nbs1, dtype=np.int32)

        t0=time.time()
        for i in range(len(self.interC)):
            #print(i)    #returns 0
            if self.interC[i].name=='cAMP':
                self.interC[i].update_C1(nbs2,n,1200)
            #else:
                #self.interC[i].update_C(nbs2,n,1200)
        t1=time.time()
        print ("Inter Transport Time: {}".format(t1-t0))
            
        #RUN A FASTER UPDATE LOOP     
        split = 1
        for p in range(0, split):
            dt = self.time_step / float(split)
            #print(dt)
            for i in range(0, len(self.objects)):
                print(self.objects[i])
                self.objects[i].update(self, dt)
                
                
    def collide(self):
        """ Handles the collision map generation 
        """
        #first make a list of points for the delaunay to use
        k=0
        for i in range(0,len(self.objects)):
            k+=self.objects[i].location[2]
        if k==0:
            n=2
        else:
            n=3
        points = np.zeros((len(self.objects), n))
        
        for i in range(0, len(self.objects)):
            #Now we can add these points to the list
            if n==2:
                points[i] = [self.objects[i].location[0], self.objects[i].location[1]]
            else:
                points[i]=self.objects[i].location
        #now perform the nearest neghbor assessment by building a delauny triangulation
        
        tri = Delaunay(points)
        #keep track of this as a network
        self.network = nx.Graph()
        #add all the simobjects
        self.network.add_nodes_from(self.objects)
        #get the data
        nbs = tri.simplices
        #iterate over all the nbs to cull the data for entry into the list
        for i in range(0, len(nbs)):
            #loop over all of the combination in this list
            ns = nbs[i]
            for a in range(0, len(ns)):
                for b in range(a+1, len(ns)):
                    #these are the IDs
                    # ns[a]
                    # ns[b]
                    self.network.add_edge(self.objects[ns[a]],
                                          self.objects[ns[b]])
        #now loop over all of these cells and cull the interaction lists
        edges = self.network.edges()
        #keep track of interactions checked
        
        for edge in edges:
            #first find the distance of the edge
            obj1 = edge[0]
            obj2 = edge[1]
            #figure out if the interaction is ok 
            #gte the minimum interaction length
            l1 = obj1.get_max_interaction_length()
            l2 = obj2.get_max_interaction_length()
            #add these together to get the connection length
            interaction_length = l1 + l2
            #get the distance
            dist_vec = SubtractVec(obj2.location, obj1.location)
            #get the magnitude
            dist = Mag(dist_vec)
            #if it does not meet the criterion, remove it from the list
            if(dist > interaction_length):
                self.network.remove_edge(obj1, obj2)


    def collide_lowDens(self):
        """ If there are too few cells, or the first iteration of the model. 
        """
        k=0
        for i in range(0,len(self.objects)):
            try:
                k+=self.objects[i].location[2]
            except:
                k+=0

        if k==0:
            n=2
        else:
            n=3

        points = np.zeros((len(self.objects), n))
        index_to_object = dict()
        for i in range(0, len(self.objects)):
                points[i]=self.objects[i].location
        
        #keep track of this as a network
        self.network = nx.Graph()
        #add all the simobjects
        self.network.add_nodes_from(self.objects)
        #Create connection between all cells 
        for i in range(0,len(self.objects)):
            for j in range(i+1,len(self.objects)):
                self.network.add_edge(self.objects[i],
                                      self.objects[j])

        edges = self.network.edges()
        #keep track of interactions checked
        
        for edge in edges:
            #first find the distance of the edge
            obj1 = edge[0]
            obj2 = edge[1]
            #figure out if the interaction is ok 
            #gte the minimum interaction length
            l1 = obj1.get_max_interaction_length()
            l2 = obj2.get_max_interaction_length()
            #add these together to get the connection length
            interaction_length = l1 + l2
            #get the distance
            dist_vec = SubtractVec(obj2.location, obj1.location)
            #get the magnitude
            dist = Mag(dist_vec)
            
            #if it does not meet the criterion, remove it from the list
            if(dist > interaction_length):
                self.network.remove_edge(obj1, obj2)
                
                
    def optimize(self):                
        #apply constraints from each object and update the positions
        #keep track of the global col and opt vectors
        opt = 2
        col = 2
        fixed = 2
        itrs = 0
        max_itrs = 50
        avg_error = 0.2 # um
        while ((opt + col + fixed) >= avg_error and itrs < max_itrs):
            #reset these values
            opt = 0
            col = 0
            fixed = 0
           
            opt, col = self._handle_spring_constraints()                       #handle the spring constraints
            fixed = self._handle_fixed_constraints()                           #handle to fixed constraints

            #now loop over and update all of the constraints
            for i in range(0, len(self.objects)):
                #update these collision and optimization vectors
                #as well as any other constraints
                self.objects[i].update_constraints(self.time_step)
            #increment the itrations
            itrs += 1
            #print the results
##            print(itrs)
##            print(opt, col, fixed)

    def _set_gradient_inital_conditions(self):
        """ Defines the masks to specify the gradient intal conditions
        """
        cent = self.get_center()
        for i in range(0, len(self.gradients)):
            #make a mask the same size as the gradient
            shp = self.gradients[i].shape()
            #print(f"shp : {shp}")
            mask = np.zeros(shp)
            for j in range(0, len(self.objects)):
                agent = self.objects[j]
                #get the object position
                pos = agent.location
                #get the difference from the center
                dist_vec = SubtractVec(pos, cent)
                #convert this to a index value
                #for all of the gradients get the coeffients
                x, y = self.gradients[i].get_object_position_on_grid(dist_vec)
                #print(x,y)
                mask[int(x),int(y)] = 1
            #now pass this in to the set inital values function
            self.gradients[i].set_initial_conditions(mask)

    def _handle_spring_constraints(self):
        """
        """
        col = 0
        opt = 0
        edges = self.network.edges()
        cent=self.get_center()
        for edge in edges:
            #get the objects
            obj1 = edge[0]
            obj2 = edge[1]
            
            if(obj1.owner_ID != obj2.owner_ID):
                v1=obj1.location
                v2=obj2.location
                        
                v12=SubtractVec(v2, v1)                      
                
                dist = Mag(v12)
                #also compute the normal
                norm = NormVec(v12)
                #get the object radii
                r_sum = obj1.radius + obj2.radius
                #check for a collision
                if(r_sum >= dist):
                    #then apply the collision
                    d1=Distance(v1,cent)
                    d2=Distance(v2,cent)
                    
                    #Move cells away from colony 'central' point.
                    d = -norm*((r_sum-dist)*0.65)
                    if d1>d2:
                        obj2.add_fixed_constraint_vec(d)
                    else:
                        obj1.add_fixed_constraint_vec(-d)

                    #add this to the collision vec
                    col += Mag(d)*2
                #apply the spring
                #also get the normal interaction length
                l1 = obj1.get_interaction_length()
                l2 = obj2.get_interaction_length()
                #now figure out how far off the connection length  is
                #from that distance
                dist = dist - (l1 + l2)
                #now get the spring constant strength
                k1 = obj1.get_spring_constant(obj2)
                k2 = obj2.get_spring_constant(obj1)
                k = min(k1, k2)
                #now we can apply the spring constraint to this
                dist = (dist/2.0) * k
                #make sure it has the correct direction
                temp = ScaleVec(norm, dist)
                #add these vectors to the object vectors
                obj1.add_displacement_vec(temp)
                obj2.add_displacement_vec(-temp)
                
                #add to the global opt vec
                opt += Mag(temp)
        #return the average opt and col values
        opt = opt / (len(self.network.edges())*2.0)
        col = col / (len(self.network.edges())*2.0)
        return opt, col

    def _handle_fixed_constraints(self):
        """
        """
        error = 0
        edges = self._fixed_constraints.edges()
        for edge in edges:
            #get the objects
            obj1 = edge[0]
            obj2 = edge[1]
            dist_vec = SubtractVec(obj2.location, obj1.location)
            if obj1.z==0 and obj2.z==0: 
                dist_vec[2]=0
            ########  remove vertical displacement
            dist_vec[2]=0
            #############
            dist = Mag(dist_vec)
            #also compute the normal
            norm = NormVec(dist_vec)
            #get the object radii
            r_sum = obj1.radius + obj2.radius
            #then apply the collision
            d = -norm*((r_sum-dist)*0.5)
            obj1.add_fixed_constraint_vec(d)
            obj2.add_fixed_constraint_vec(-d)

            #add this to the collision vec
            error += Mag(d)*2
        #calculate the average error
        try:
            error = error / len(self._fixed_constraints.edges())
        except ZeroDivisionError:
            error = 0 
        return error

    def _define_sinks_and_sources(self):
        """ Defines the sinks and soruce terms from the simulation
            network by figuring out where each agent in the network lies
            and defining the subsequent diffusion procedures.
            returns - a list of (sink, source) as a tuple for each gradeint
        """
        #make a set of sink and source arrays which match
        #the size of each gradient
        sources = []
        sinks = []
        #print(f"length of gradient: {len(self.gradients)}")
        for i in range(0, len(self.gradients)):
            shp = self.gradients[i].shape()
            source = np.zeros(shp)
            sink = np.zeros(shp)
            
            sources.append(source)
            sinks.append(sink)
        #print(shp)
        #print(source)
        #print(sink.shape)

        #get the center of the network
        cent = self.get_center()
        #print(f"sim.center: {cent}")
        #now loop over all of the agents in the simulation
        for j in range(0, len(self.objects)):
            agent = self.objects[j]
            #get the object position
        
            pos = agent.location
            #get the difference from the center
            dist_vec = SubtractVec(pos, cent)
            #print(dist_vec)
            #convert this to a index value
            #for all of the gradients get the coeffients
            for i in range(0, len(self.gradients)):
                #print(f"agent location: {j, agent.location}")
                x, y = self.gradients[i].get_object_position_on_grid(dist_vec)
                #print(f"object position on grid: {j, x,y,z}")
                agent.set_gradient_location(self.gradients[i].name, (x,y))
                #see if the object conatins a value for this gradient
                src, snk = agent.get_gradient_source_sink_coeff(self.gradients[i].name)
                #print(f"sources and sinks: {sources[i][int(x),int(y),int(z)], sinks[i][int(x),int(y),int(z)]}")
                sources[i][int(x),int(y)] += src
                sinks[i][int(x),int(y)] += snk
                #print(sources[i].shape)
                #print(src)
                
        #and return the list
        return sources, sinks

    def get_center(self):
        """ Returns the center of the simulation
            return - point in the form of (x,y,z)
        """
        n=len(self.objects[1].location)
        if n==3:
            cent = (0,0,0)
        else:
            cent=(0,0)
                
        for i in range(0, len(self.objects)):            
            cent = AddVec(self.objects[i].location, cent)
        #then scale the vector
        cent = ScaleVec(cent, 1.0/ len(self.objects))
        #and return it
        return cent
    
    def _create_header_file(self):
        """ Creates a simulation header file which will save all the relevant
            information in text format to the simulation file.
        """
        #first create the file
        temp_path = self.path + self._sep + self.name + self._sep + "info.sim"
        self._header = open(temp_path, "w")
        #now save the simulation name
        self._header.write("Name:" + self.name + "\n")
        #write the path
        self._header.write("Base path:" + self.path + "\n")
        #write the sepeartor info
        self._header.write("Platform:" + platform.system() + "\n")
        #write the time info
        self._header.write("Time range:" + repr(self.start_time) + ",")
        self._header.write(repr(self.end_time) + "," + repr(self.time_step) + "\n")
        #Write the gradient names
        self._header.write("Gradient Species:")
        for i in range(0, len(self.gradients)):
            self._header.write(self.gradients[i].name)
            if(i > len(self.gradients) - 1):
                self._header.write(",")
        self._header.write("\n")
        self._header.write("InterC Species:")
        for i in range(0, len(self.gradients)):
            self._header.write(self.interC[i].name)
            if(i > len(self.interC) - 1):
                self._header.write(",") 
        #Now the header file is open and defined
        self._header.write("\n")  
        
    def save(self):
        """ Saves the simulation snapshot.
        """
        #Write the current itme into the header file
        self._header.write(repr(self.time))
        #get the base path
        base_path = self.path +self._sep +self.name + self._sep
        #First save the network files
        n_path = base_path + "network" + repr(self.time) + ".gpickle"
        with open(n_path, 'wb') as f:
            pickle.dump(self.network, f)
        #now write that path to the file
        self._header.write("," + n_path)
        #Then save the gradient files
        for i in range(0, len(self.gradients)):
            molecule = self.gradients[i].name
            grad_path = self.gradients[i].save(base_path, repr(self.time))
            self._header.write("," + grad_path)
        self._header.write(",")
        for i in range(0, len(self.interC)):
            grad_path = self.interC[i].save(base_path, repr(self.time))
            self._header.write("," + grad_path)
        #put the final new line character
        self._header.write("\n")
        #fluish the file
        self._header.flush()

