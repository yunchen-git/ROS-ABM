# cython: language_level=3
################################################################################
# File: IntC_diff.pyx
# Purpose:
#   Cython-accelerated intercellular transport / coupling module.
#   Manages per-cell intracellular concentrations and computes fluxes between
#   neighboring cells (e.g., gap-junction-like exchange), optionally applying
#   inhibition toggles (e.g., GJ/AC modes).
#
# Key public API:
#   - class IntC_Diff:
#       add_cell(), update_Cflux(), update_C0/C1/C2(), change_C(), save(), etc.
#
# How it fits in the codebase:
#   Registered into Simulation via Simulation.add_interC(...).
#   Couples to intracellular reaction model via imports from FBA_ode.py.
#
# Inputs/Outputs:
#   - Can serialize its internal state via save() into the run folder.
#
# Notes / gotchas:
#   - This is performance-critical; keep operations vectorized / Cython-friendly.
#   - Build/compile via setup.py (build_ext --inplace).
################################################################################

cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import scipy.integrate
import numpy as np
cimport numpy as np
import pickle
import time
import cython
cimport cython
import math

from cython.view cimport array as cvarray
from libc.stdlib cimport abs
from libc.math cimport fabs
#from libc.math cimport fmin

import FBA_ode
from FBA_ode import model, blap_ode_sys

import gc, atexit

SOLVER_METHOD = "Radau"
SOLVER_ATOL = 1e-10
SOLVER_RTOL_CFLUX = 1e-8
SOLVER_RTOL_SCFLUX = 1e-4

#-----------------------------------------------------------------------

"""
To change this code, first modify the code in this file then
save a copy of this script as IntC_FP.pyx.

To compile go to a command console and:
1. Change to this directory (type cd, press space then drag/drop this folder to the console. Press enter)
2. Type out: "python setup.py build_ext --inplace" without the quotation marks. Press enter.

"""


class IntC_Diff(object):
    """ Class which defines a gradient.
        name - the name of the species
        D - the diffusion coeffiecnt
        x, y, z - the spaital size of the grid
        dx, dy, dz - the spatial resolution of each grid step
        outside_c - the value fo the gradient outside the box
        unit_vol = the uinit vole in L (ul) m or um
    """

        
    def __init__(self, name, maxCells, D, dx, Pmax, open_prob, celltypes,
                 division_times,sync_time,vol_units = 1e-6):
        #save the unit volume
    
        self._vol_units = 1e-6
        self.Pmax=float(Pmax)
        
        self.p_n=(open_prob)
        #save the name
        self.name = name
        #set the diffusion coefficient
        self._D = float(D)
        self.division_times = division_times
        
        cdef int i,j,n1,n2
        cdef double div
        cdef int[:] sync_state = cvarray(shape=(2,), itemsize=sizeof(int),format="i")
        sync_state[0]=int(sync_time[0])
        sync_state[1]=int(sync_time[1])
        self.sync_t = sync_state
        #set the dimensions in each direction
        n1=np.max(division_times)
        n2=len(division_times)

        #cdef double[:,:] cell_cycle_effect = cvarray(shape=(n1,n2), itemsize=sizeof(double),format="d")
        cdef np.ndarray[np.float64_t, ndim=2] cell_cycle_effect = np.zeros((n1, n2), dtype=np.float64)
        cdef double[:,:] inhibition = cvarray(shape=(maxCells,2), itemsize=sizeof(double),format="d")
        self.inhibition=inhibition
        
        #for i in range(n2):
        #    for j in range(n1):
        #        div=abs(j-(division_times[i]/4.0))
        #        cell_cycle_effect[j,i]=1/(1+(div/(division_times[i]/4.0))**2)+0.69*j**6/(j**6+division_times[i]**6)
                #print(cell_cycle_effect[j,i])
          
        self.cce=cell_cycle_effect
        #print(self.cce.shape) (51,2)
        #set the spatial resolution in each direction
        self._dx = float(dx)

        #solve for the index of the array
        self.maxInd = maxCells
        #solve some of the more specific constraints
        dx2 = dx*dx

        #time step solution
        self.dt = 0.5 / (D * (1/dx2))
        
        #solve for the spatial correction constanta in each dimension
        self.vx = D * self.dt / dx2

        self.deg_prod = celltypes

        cdef double[:,:] C = cvarray(shape=(maxCells,2), itemsize=sizeof(double),format="d")
##        cdef int[:] state = np.zeros((5,), dtype=np.int)
        cdef long[:,:] state = cvarray(shape=(maxCells, 2), itemsize=sizeof(long), format="l")  
        self.state = state
        C[:,:]=0
        self.C=C

    def save(self, base_path, time_stamp):
        """ Saves the gradients to a binary numpy file
            Also outputs a string which represents
            import header information about the gradient
        """
        cdef np.ndarray temp = np.zeros([self.maxInd,2])
        cdef np.ndarray temp1 = np.zeros((self.maxInd,2),dtype=np.int64)
        cdef np.ndarray temp2 = np.zeros(self.cce.shape)
        cdef np.ndarray temp3 = np.zeros(self.inhibition.shape)
        cdef np.ndarray temp4 = np.zeros(self.sync_t.shape,dtype=np.int64)
        temp+=self.C
        temp1+=self.state
        temp2+=self.cce
        temp3+=self.inhibition
        temp4+=self.sync_t
        self.C=temp
        self.state=temp1
        self.cce=temp2
        self.inhibition=temp3
        self.sync_t=temp4
        path = base_path + self.name + "_" + time_stamp
        f = open(path, "wb")
        pickle.dump(self, f,2)
        f.close()
 
        return path
  
    def add_cell(self,ID,Ci,celltype, div_timer, inhib_gj,inhib_ac):
        cdef double[:,:] C = self.C
        cdef double[:,:] inh = self.inhibition
        cdef long[:,:] state = self.state
        cdef int i =ID
        cdef int ct = celltype
        cdef int div = div_timer
        state[i,0]=ct
        state[i,1]=div
        inh[i,0]=inhib_gj
        inh[i,1]=inhib_ac
        C[i,0]=Ci
        
    def get_C(self,ID):
        cdef double[:,:] C = self.C
        cdef int i =ID
        return C[i,0]

    def restart_div_cycle(self, ID):
        cdef long[:,:] state = self.state
        cdef int i =ID
        state[i,1]=0
        
    def change_state(self,ID,celltype):
        cdef double[:,:] C = self.C
        cdef long[:,:] state = self.state
        cdef int i =ID
        cdef int ct = celltype
        state[i,0]=ct

    def change_C(self,ID,Cnew):
        cdef double[:,:] C = self.C
        cdef int i =ID
        C[i,0]=Cnew

    def change_inhib_GJ(self,ID,inhib_value):
        cdef double[:,:] inh = self.inhibition
        cdef int i =ID
        inh[i,0]=inhib_value
        
    def change_inhib_AC(self,ID,inhib_value):
        cdef double[:,:] inh = self.inhibition
        cdef int i =ID
        inh[i,1]=inhib_value
        
    def add_to_dC(self,ID,Cmod):
        cdef double[:,:] C = self.C
        cdef int i =ID
        #print(C[i,0], Cmod, C[i,0]+Cmod)
        C[i,0]=C[i,0]+Cmod

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)  
    def update_Cflux(self, n, x0, te): #GSH, extraC, te):
        cdef double[:,:] C = self.C
        cdef long[:,:] state = self.state
        cdef int tend = te
        
        #cdef np.ndarray x0 = np.copy(FBA_ode.get_x0())
        cdef np.ndarray k = FBA_ode.get_k()
        cdef np.ndarray x_0 = x0
        
        # Initialize x0 with extraC and self.C
        #x0[0] = extraC
        x_0[1] = C[n, 0]
        #x0[5] = GSH
        
        #h2o2 scavenging rate constants
        #k[2]=0          #
        #k[5]=0
        #k[7]=0
        #k[8]=0
        #k[13]=0
        #k[17]=0
        #k[22]=0

        #h2o2 production
        #k[1]=0

        #if x_0[1] > 1e-7 or x_0[29] > 1e-19:
        #k[32] = x_0[1] * 1e-6
        if x_0[1] > 1e-7:
            k[31] = 0#1e-4                #nox
            #k[1] = 1.1e-5               #internal production
        else:
            k[31] = 0
            #k[1] = k[1]
            
        # Create time points for integration (0 to te with te+1 points)
        t = np.linspace(0, tend, tend+1)  # With te=3600, this gives 3601 points
        
        # Solve the ODE system across the whole interval
        # result = scipy.integrate.odeint(model, x0, t, args=(k,x0))

        try:
            # Use more robust integration method with error control
            result = scipy.integrate.solve_ivp(
                lambda t, y: model(y, t, k, x_0),
                [0, tend],
                x_0,
                method=SOLVER_METHOD,
                t_eval=t,
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL_CFLUX
            )
            final_values = result.y[:, -1]  # Get the last time point values
            #print(result.y.shape)  # Debugging: print the shape of the result
            
            # Update the cell's internal concentration with only the final value
            C[n, 0] = final_values[1]  # Last time point, H2O2 cytosol concentration
            #print(final_values[1], final_values[0])  # Debugging: print final concentrations
            
            # Return final internal and external concentrations
            return final_values

        except ValueError as e:
            print(f"Error in cell {n}: {e}")
            print(f"x0 = {x0}")
            # Return current values if integration fails
            return x_0

        # Return final internal and external concentrations
        #return final_values[5], final_values[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)  
    def update_scflux(self, n, extraC, te, protein_vals, GSH):
        cdef double[:,:] C = self.C
        cdef long[:,:] state = self.state
        cdef int tend = te
        cdef int timestep = 1
        cdef double cell_density = 1.0e9
        extraC = extraC

        cdef np.ndarray x0 = np.copy(FBA_ode.get_blap_x(protein_vals, 3, extraC, C[n,0]))
        cdef np.ndarray k = FBA_ode.get_blap_k(protein_vals, 3, extraC, C[n,0])
        
        t = np.linspace(0, tend, tend+1)  # With te=3600, this gives 3601 points

        try:
            result = scipy.integrate.solve_ivp(
                lambda t, y: blap_ode_sys(y, t, k, x0, cell_density),
                [0, tend],
                x0,
                method=SOLVER_METHOD,
                t_eval=t,
                atol=SOLVER_ATOL,
                rtol=SOLVER_RTOL_SCFLUX
            )
            final_values = result.y[:, -1]  # Get the last time point values
            C[n, 0] = final_values[1]  # Last time point, H2O2 cytosol concentration
            return final_values[5], final_values[0]
        except ValueError as e:
            print(f"Error in cell {n}: {e}")
            print(f"x0 = {x0}")
            # Check which values might be causing issues in the result
            if 'result' in locals():
                if hasattr(result, 'y'):
                    print("Result shape:", result.y.shape)
                    for i in range(min(5, result.y.shape[0])):  # Print first few rows
                        print(f"result.y[{i},:] = {result.y[i,:]}")
                    
                    # Find any NaN or infinity values in the result
                    if np.any(~np.isfinite(result.y)):
                        nan_indices = np.where(~np.isfinite(result.y))
                        print(f"Non-finite values at indices: {nan_indices}")
                        print(f"Values: {result.y[nan_indices]}")
                        
            # Return current values if integration fails
            return x0[5], x0[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)  
    def update_C0(self,cell_nbs,num,te):
        cdef int dim = num
        cdef int[:,:] nbs = cell_nbs
        cdef int edges =nbs.shape[0]
        cdef double[:,:] dp = self.deg_prod
        cdef double[:,:] C = self.C
        cdef double[:,:] cce = self.cce
        cdef double[:] p_n =self.p_n
        cdef int[:] division_times = self.division_times
        
        cdef double[:,:] inh = self.inhibition
        cdef double[:] Peff = cvarray(shape=(edges,), itemsize=sizeof(double),format="d")
        cdef long[:,:] state = self.state  
        cdef int i,j,x,ni,nj,mi,mj
        
        cdef double t=0.0
        cdef double tend=te
        cdef double Pmax=self.Pmax
        cdef double vx=self.vx
        
        cdef double dt =self.dt
        cdef double prod, deg, qni, qnj
        #print(np.array(C))
        for x in range(edges):
            i=nbs[x,0]
            j=nbs[x,1]
            ni=state[i,0]
            mi=state[i,1]
            nj=state[j,0]
            mj=state[j,1]
            ccei = 1/(1+(fabs(mi-(division_times[ni]/4.0))/(division_times[ni]/4.0))**2)+0.69*mi**6/(mi**6+division_times[ni]**6)
            ccej = 1/(1+(fabs(mj-(division_times[nj]/4.0))/(division_times[nj]/4.0))**2)+0.69*mj**6/(mj**6+division_times[nj]**6)
            Peff[x]=vx*Pmax#*(p_n[ni]*ccei)*(p_n[nj]*ccej)

        #print(np.array(np.min(Peff)))       #min is 1.187e-05
        #print(np.array(np.max(Peff)))       #max is 0.0002
        while t<tend:
            for x in range(edges):
                i=nbs[x,0]
                j=nbs[x,1]
                ni =state[i,0]
                C[i,1]=C[i,1]+Peff[x]*(C[j,0]-C[i,0])
                C[j,1]=C[j,1]+Peff[x]*(C[i,0]-C[j,0])
                #C[i,1]=C[i,1]+(C[j,0]-C[i,0])*vx
                #C[j,1]=C[j,1]+(C[i,0]-C[j,0])*vx

            for i in range(dim):
                ni =state[i,0]  
                if C[i,0]<0:
                    C[i,0]=0    
                elif math.isnan(C[i,0]):
                    C[i,0]=0
                C[i,0]=C[i,0]+C[i,1]

                #print(C[i,1]*tend/dt, C[i,0], dt, vx, Pmax, self.name)
                 # Debugging: print the change in concentration
                C[i,1]=0.0
            t=t+dt

        for i in range(dim):
            state[i,1]=state[i,1] + te

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)  
    def update_C1(self,cell_nbs,num,te):
        cdef int dim = num
        cdef int[:,:] nbs = cell_nbs
        cdef int edges =nbs.shape[0]
        cdef double[:,:] dp = self.deg_prod
        cdef double[:,:] C = self.C
        cdef double[:,:] cce = self.cce
        cdef double[:] p_n =self.p_n
        cdef int[:] division_times = self.division_times
        
        cdef double[:,:] inh = self.inhibition
        cdef double[:] Peff = cvarray(shape=(edges,), itemsize=sizeof(double),format="d")
        cdef long[:,:] state = self.state  
        cdef int i,j,x,ni,nj,mi,mj
        
        cdef double t=0.0
        cdef double tend=te
        cdef double Pmax=self.Pmax
        cdef double vx=self.vx
        
        cdef double dt =self.dt
        cdef double prod, deg, qni, qnj
        #print(np.array(C))
        for x in range(edges):
            i=nbs[x,0]
            j=nbs[x,1]
            ni=state[i,0]
            mi=state[i,1]
            nj=state[j,0]
            mj=state[j,1]
            ccei = 1/(1+(fabs(mi-(division_times[ni]/4.0))/(division_times[ni]/4.0))**2)+0.69*mi**6/(mi**6+division_times[ni]**6)
            ccej = 1/(1+(fabs(mj-(division_times[nj]/4.0))/(division_times[nj]/4.0))**2)+0.69*mj**6/(mj**6+division_times[nj]**6)
            Peff[x]=vx*Pmax*(p_n[ni]*ccei)*(p_n[nj]*ccej)

        #print(np.array(np.min(Peff)))       #min is 1.187e-05
        #print(np.array(np.max(Peff)))       #max is 0.0002
        while t<tend:
            for x in range(edges):
                i=nbs[x,0]
                j=nbs[x,1]
                ni =state[i,0]
                C[i,1]=C[i,1]+Peff[x]*(C[j,0]-C[i,0])
                C[j,1]=C[j,1]+Peff[x]*(C[i,0]-C[j,0])
                #C[i,1]=C[i,1]+(C[j,0]-C[i,0])*vx
                #C[j,1]=C[j,1]+(C[i,0]-C[j,0])*vx

            for i in range(dim):
                ni =state[i,0]  
                if C[i,0]<0:
                    C[i,0]=0    
                elif math.isnan(C[i,0]):
                    C[i,0]=0
                
                C[i,0]=C[i,0]+C[i,1]
                #print(C[i,1]*tend/dt, C[i,0], dt, vx, Pmax)
                C[i,1]=0.0
            t=t+dt

        for i in range(dim):
            state[i,1]=state[i,1] + te

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)  
    def update_C2(self,cell_nbs,num,te):
        cdef int dim = num
        cdef int[:,:] nbs = cell_nbs
        cdef int edges =nbs.shape[0]
        cdef double[:,:] dp = self.deg_prod
        cdef double[:,:] C = self.C
        cdef double[:,:] cce = self.cce
        cdef double[:] p_n =self.p_n
        cdef int[:] division_times = self.division_times
        
        cdef double[:,:] inh = self.inhibition
        cdef double[:] Peff = cvarray(shape=(edges,), itemsize=sizeof(double),format="d")
        cdef long[:,:] state = self.state  
        cdef int i,j,x,ni,nj,mi,mj
        
        cdef double t=0.0
        cdef double tend=te
        cdef double Pmax=self.Pmax
        cdef double vx=self.vx
        
        cdef double dt =self.dt
        cdef double prod, deg, qni, qnj
        #print(np.array(C))
        for x in range(edges):
            i=nbs[x,0]
            j=nbs[x,1]
            ni=state[i,0]
            mi=state[i,1]
            nj=state[j,0]
            mj=state[j,1]
            ccei = 1/(1+(fabs(mi-(division_times[ni]/4.0))/(division_times[ni]/4.0))**2)+0.69*mi**6/(mi**6+division_times[ni]**6)
            ccej = 1/(1+(fabs(mj-(division_times[nj]/4.0))/(division_times[nj]/4.0))**2)+0.69*mj**6/(mj**6+division_times[nj]**6)
            Peff[x]=vx*Pmax#*(p_n[ni]*ccei)*(p_n[nj]*ccej)
            #print(vx, Pmax, p_n[ni], cce[mi,ni], inh[i,0], p_n[nj], cce[mj,nj], inh[j,0])

        #print(np.array(np.min(Peff)))       #min is 1.187e-05
        #print(np.array(np.max(Peff)))       #max is 0.0002
        while t<tend:
            for x in range(edges):
                i=nbs[x,0]
                j=nbs[x,1]
                ni =state[i,0]
                C[i,1]=C[i,1]+Peff[x]*(C[j,0]-C[i,0])
                C[j,1]=C[j,1]+Peff[x]*(C[i,0]-C[j,0])

            for i in range(dim):
                ni =state[i,0]  
                if C[i,0]<0:
                    C[i,0]=0    
                elif math.isnan(C[i,0]):
                    C[i,0]=0
                
                prod=inh[i,1]*vx*dp[ni,0]/(1.+(C[i,0]/dp[ni,1])**dp[ni,2])
                #print(prod)
                deg=-vx*dp[ni,3]*C[i,0]

                C[i,0]=C[i,0]+C[i,1]+prod+deg
                C[i,1]=0.0
            t=t+dt

        for i in range(dim):
            state[i,1]=state[i,1] + te

    def __repr__(self):
        """ Returns a string representation of the gradient
        """
        return self.name + ": "  + " " + repr(np.min(self.C)) + " " + repr(np.max(self.C))
    
    @classmethod   
    def garbage_collect(cls):
        gc.collect()
                         
# Register garbage_collect with atexit
atexit.register(lambda: IntC_Diff.garbage_collect())
