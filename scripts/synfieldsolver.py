import numpy as np
import magfield as mg
import synfieldtools as sn

if __name__ == '__main__':
    #Set field and modes
    availablefieldtypes = ['simplewire', 'oppositecoils']
    fieldtype = 'oppositecoils'
    I = 10000 #Current parameter for the field
    norot = False #Whether to ignore rotational degrees of freedom
    nosyn = False #Whether to ignore synthetic fields
    
    #Define parameters

    nr = 100 #Number of points in field lattice
    lablength = 1e-3 #Cube side of lab in m
    tmax = 1 #Trajectory time in s
    J = 1e9 #Spin-spin coupling strength
    Gamma = 1e9 #Spin-field coupling strength
    
    mass0 = 3.58e-25 #The total mass of the dumbbell in kg, as a placeholder this is the mass of
                     #two silver atoms
    len0 = 5e-3 #The distance between dumbbell edges in m #This is now stupidly large
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
    initpos = (lablength/10, lablength/10, lablength/10, 2, 2)
    step = lablength/nr
    initposarray = np.array((5*step, 5*step, 5*step, 0, 0))
    swarmgrid = np.mgrid[0:lablength-7*step:4j,0:lablength-7*step:4j] #Grid to swarm the initial positions
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 0, np.zeros(swarmgrid.shape[1:3]), axis=0)
    initposarray = initposarray[:,None,None] + swarmgrid
    initvel = (lablength*100/(tmax), 0, 0, 0, 0)
    eigenstate = 1

    #Generate a field
    if fieldtype == 'simplewire':
        field = mg.simplewire(nr, lablength, I)
    if fieldtype == 'oppositecoils':
        field = mg.oppositecoils(nr, lablength, I, overwrite=False)
    
    #Integrate paths
    print('Integrating dynamics . . . Please wait')
    sol = []
    for i in range(initposarray.shape[1]):
        for j in range(initposarray.shape[2]):
            print(tuple(initposarray[:,i,j]))
            sol.append(sn.solvedyn(tuple(initposarray[:,i,j]), initvel, field, eigenstate, norot, nosyn))
    
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

    #Save the result
    with open(f'saves/resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}pos{initpos}vel{initvel}norot{norot}nosyn{nosyn}.bin', 'w') as file:
        file.write(repr(sol))

    
    sn.lineplot(sol, field)
