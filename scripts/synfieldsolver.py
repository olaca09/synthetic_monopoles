import numpy as np
import magfield as mg
import synfieldtools as sn
import pickle
from numpy import pi as pi

if __name__ == '__main__':
    #Set field and modes
    availablefieldtypes = ['simplewire', 'oppositecoils']
    fieldtype = 'oppositecoils'
    I = 10 #Current parameter for the field
    norot = False #Whether to ignore rotational degrees of freedom
    nosyn = 'False' #Whether to ignore synthetic fields, accepts the strings 'False',
                    #'True', 'Nomag' and 'Noscalar'
    overwriteresult = False #Whether to overwrite previous ODE results
    alternatestreams = False #Whether to use alternate swarming scheme
    
    #Define parameters

    nr = 101 #Number of points in field lattice
    lablength = 1e-3 #Cube side of lab in m
    tmax = 0.1 #Trajectory time in s
    J = 1e5 #Spin-spin coupling strength
    Gamma = 1e10 #Spin-field coupling strength
    
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
    altinitpos = initposarray[:,0,2] #Starting position for alternate swarming method
    initvel = (1e-2, 0, 0, 0, 0)
    eigenstate = 2

    try: #Try to load pregenerated result
        with open(f'saves/odesols/resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin', 'rb') as file:
            sol = pickle.load(file)
            print(f'Loading result from file resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin')
    except FileNotFoundError:
        overwriteresult = True

    #Generate a field
    if fieldtype == 'simplewire':
        field = mg.simplewire(nr, lablength, I)
    if fieldtype == 'oppositecoils':
        field = mg.oppositecoils(nr, lablength, I)
        
    if overwriteresult: #Only perform calculations if result not avaiable on save:
    
        #Integrate paths
        print('Integrating dynamics . . . Please wait')
        sol = []
        if alternatestreams:
            print(f'Simulating start {tuple(altinitpos)} with synthetic fields')
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, field, eigenstate, norot, 'False')
                stream.color = 'red'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print(f'A stream has failed to generate, most probably due to crossing the centre')
            print(f'Simulating start {tuple(altinitpos)} without synthetic fields')
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, field, eigenstate, norot, 'True')
                stream.color = 'blue'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print(f'A stream has failed to generate, most probably due to crossing the centre')
            print(f'Simulating start {tuple(altinitpos)} without the magnetic synthetic field')
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, field, eigenstate, norot, 'Nomag')
                stream.color = 'green'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print(f'A stream has failed to generate, most probably due to crossing the centre')
            print(f'Simulating start {tuple(altinitpos)} without the scalar synthetic field')
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, field, eigenstate, norot, 'Noscalar')
                stream.color = 'orange'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print(f'A stream has failed to generate, most probably due to crossing the centre')

        else:
            for i in range(initposarray.shape[1]):
                for j in range(initposarray.shape[2]):
                    print(tuple(initposarray[:,i,j]))
                    try:
                        sol.append(sn.solvedyn(tuple(initposarray[:,i,j]), initvel, field, eigenstate, norot, nosyn))
                        print(f'Stream number {len(sol)} has been integrated!')
                    except Exception as e:
                        print(e)
                        print(f'A stream has failed to generate, most probably due to crossing the centre')
        
        #Save the result
        print(f'Saving result to file resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin')
        with open(f'saves/odesols/resultF{fieldtype}I{I}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin', 'wb') as file:
            pickle.dump(sol, file)

    else:
        print('Loading previously generated result')
    #Extract positions and orientations of the first stream
    pos = sol[0].y[0:3,:]
    vel = sol[0].y[5:8,:]
    ori = sol[0].y[3:5,:]
    #Print the result
    print('Times sampled:')
    print(sol[0].t)
    print('Rotation integrated:')
    print(ori)
    
    sn.lineplot(sol, field, I, initvel, swarmnum, eigenstate, norot, nosyn, alternatestreams)
