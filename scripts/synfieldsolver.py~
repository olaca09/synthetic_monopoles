import numpy as np
import magfield as mg
import synfieldtools as sn
import pickle
from numpy import pi as pi

if __name__ == '__main__':
    #Set field and modes
    availablefieldtypes = ['simplewire', 'oppositecoils']
    fieldtype = 'oppositecoils'
    I = 100 #Current parameter for the field
    norot = False #Whether to ignore rotational degrees of freedom
    nosyn = True #Whether to ignore synthetic fields
    overwriteresult = False #Whether to overwrite previous ODE results
    
    #Define parameters

    nr = 101 #Number of points in field lattice
    lablength = 1e-3 #Cube side of lab in m
    tmax = 0.1 #Trajectory time in s
    J = 1e2 #Spin-spin coupling strength
    Gamma = 1e9 #Spin-field coupling strength
    
    mass0 = 3.58e-25 #The total mass of the dumbbell in kg, as a placeholder this is the mass of
                     #two silver atoms
    len0 = 5e-5 #The distance between dumbbell edges in m
    mass = np.repeat(mass0, 5) #Full mass vector
    mass[3] = mass0*len0**2/4
    mass[4] = mass[3]

    #Set parameters
    sn.nr = nr
    sn.lablength = lablength
    sn.tmax = tmax
    sn.J = J
    sn.Gamma = Gamma
    sn.mass0 = mass0
    sn.len0 = len0
    sn.mass = mass

    #Set initial position, velocity and the fast eigenstate to consider
    step = lablength/nr
    initposarray = np.array((10*step, 10*step, 10*step, 0, 0))
    swarmnum = 4 #Square root of number of streams in swarm
    swarmgrid = np.mgrid[0:lablength-20*step:swarmnum*1j,0:lablength-20*step:swarmnum*1j] #Grid to swarm the initial positions
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 0, np.zeros(swarmgrid.shape[1:3]), axis=0)
    initposarray = initposarray[:,None,None] + swarmgrid
    initvel = (1e-2, 0, 0, 0, 0)
    eigenstate = 2

    try: #Try to load pregenerated result
        with open(f'saves/odesols/resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}.bin', 'rb') as file:
            sol = pickle.load(file)
    except FileNotFoundError:
        overwriteresult = True

    #Generate a field
    if fieldtype == 'simplewire':
        field = mg.simplewire(nr, lablength, I)
    if fieldtype == 'oppositecoils':
        field = mg.oppositecoils(nr, lablength, I, overwrite=False)
        
    if overwriteresult: #Only perform calculations if result not avaiable on save:
    
        #Integrate paths
        print('Integrating dynamics . . . Please wait')
        sol = []
        for i in range(initposarray.shape[1]):
            for j in range(initposarray.shape[2]):
                print(tuple(initposarray[:,i,j]))
                try:
                    sol.append(sn.solvedyn(tuple(initposarray[:,i,j]), initvel, field, eigenstate, norot, nosyn))
                except Exception:
                    print(f'A stream has failed to generate, most probably due to crossing the centre')
        
        #Save the result
        print(f'Saving result to resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}norot{norot}nosyn{nosyn}.bin')
        with open(f'saves/odesols/resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}.bin', 'wb') as file:
            pickle.dump(sol, file)

    else:
        print('Loading previously generated result')
    #Extract positions and orientations
    pos = sol[0].y[0:3,:]
    vel = sol[0].y[5:8,:]
    ori = sol[0].y[3:5,:]
    #Print the result
    print('Times sampled:')
    print(sol[0].t)
#    print('Path integrated:')
#    print(pos)
    print('Rotation integrated:')
    print(ori)
#    print('Velocity integrated:')
#    print(vel)


    
    sn.lineplot(sol, field, I, initvel, swarmnum, eigenstate, norot, nosyn)
