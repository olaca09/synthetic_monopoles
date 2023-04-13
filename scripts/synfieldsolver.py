import numpy as np
import synfieldtools as sn
import pickle
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main():
    # Parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--norot', action='store_true',
                        help='Ignores rotation if set')
    parser.add_argument('-s', '--nosyn', default='False', type=str,
                        help='Ignores synthetic fields if set to True')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Always generate new data if set')
    parser.add_argument('-a', '--alternatestreams', action='store_false',
                        help='Overrides "nosyn", integrates a'
                        + ' single starting point for all combinations of'
                        + ' fields. Is on by default, turns off if set.')
    parser.add_argument('-c', '--current', default=10, type=float,
                        help='Current in amperes flowing through coils')
    parser.add_argument('--br', default=10, type=int, help='Integration'
                        + ' granularity along coil side')
    parser.add_argument('--nr', default=1e5, type=int, help='Differentiation'
                        + ' step is lab length divided by this number')
    parser.add_argument('--lablength', default=1e-3, type=float, help='Size'
                        + ' of lab cube side in meters')
    parser.add_argument('--tmax', default=0.5, type=float, help='Maximum'
                        + ' flight time in seconds')
    parser.add_argument('-j', '--jcouple', default=1e5, type=float,
                        help='Spin-spin coupling strength')
    parser.add_argument('-g', '--gammacouple', default=1e8, type=float,
                        help='Spin-field coupling strength')
    parser.add_argument('-m', '--mass', default=3.58e-27, type=float,
                        help='The total mass of dumbbell in kg')
    parser.add_argument('-l', '--dumbbellength', default=5e-5, type=float,
                        help='The distance between dumbbell edges in m')
    args = vars(parser.parse_args())

    # Set field and modes
    current = args['current']  # Current parameter for the field
    # Number of points to consider along each wire when integrating external
    # field
    br = args['br']
    norot = args['norot']  # Whether to ignore rotational degrees of freedom
    # Whether to ignore synthetic fields, accepts the strings 'False',
    # 'True', 'Nomag' and 'Noscalar'
    nosyn = args['nosyn']  # Must be a string.
    # Whether to overwrite previous ODE results
    overwriteresult = args['overwrite']
    # Whether to use alternate swarming scheme
    alternatestreams = args['alternatestreams']

    # Define parameters

    # Number of points to divide the lab length by for numerical
    # differentiation step size
    nr = args['nr']
    # Cube side of lab in m
    lablength = args['lablength']
    # Trajectory time in s
    tmax = args['tmax']
    # Spin-spin coupling strength
    J = args['jcouple']
    # Spin-field coupling strength
    Gamma = args['gammacouple']
    # The total mass of the dumbbell in kg, as a placeholder this is the mass
    # of two silver atoms
    mass0 = args['mass']
    # The distance between dumbbell edges in m
    len0 = args['dumbbellength']
    # Full mass vector
    mass = np.repeat(mass0, 5)
    mass[3] = mass0*len0**2/4
    mass[4] = mass[3]

    # Tuple collecting field parameters
    field = (br, lablength, current)

    # Set parameters
    sn.nr = nr
    sn.lablength = lablength
    sn.tmax = tmax
    sn.J = J
    sn.Gamma = Gamma
    sn.mass0 = mass0
    sn.len0 = len0
    sn.mass = mass
    sn.field = field

    # Set initial position, velocity and the fast eigenstate to consider
    # step = lablength/nr
    initposarray = np.array((0, 0, 0, 0, 0))
    # Square root of number of streams in swarm
    swarmnum = 4
    # Grid to swarm the initial positions
    swarmgrid = np.mgrid[0:lablength:swarmnum*1j, 0:lablength:swarmnum*1j]
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 2, np.zeros(swarmgrid.shape[1:3]), axis=0)
    swarmgrid = np.insert(swarmgrid, 0, np.zeros(swarmgrid.shape[1:3]), axis=0)
    initposarray = initposarray[:, None, None] + swarmgrid
    # Starting position for alternate swarming method
    altinitpos = initposarray[:, 1, 1]
    initvel = (1e-2, 0, 0, 0, 0)
    eigenstate = 2

    # Try to load pregenerated result
    try:
        with open(f'''saves/odesols/result_field{field}nr{nr}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin''', 'rb') as file:
            sol = pickle.load(file)
            print(f'''Loaded result from file result_field{field}nr{nr}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin''')
    except FileNotFoundError:
        overwriteresult = True

    # Only perform calculations if result not avaiable on save:
    if overwriteresult:
        # Integrate paths
        print('Integrating dynamics . . . Please wait')
        sol = []
        if alternatestreams:
            print(f'Simulating start {tuple(altinitpos)} with synthetic'
                  + ' fields')
            start = time.time()
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, eigenstate,
                                     norot, 'False')
                stream.color = 'red'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print('A stream has failed to generate, most probably due to'
                      + ' crossing the centre')
            end = time.time()
            print(f'This took {end - start} seconds')

            print(f'Simulating start {tuple(altinitpos)} without synthetic'
                  + ' fields')
            start = time.time()
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, eigenstate,
                                     norot, 'True')
                stream.color = 'blue'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print('A stream has failed to generate, most probably due to'
                      + ' crossing the centre')
            end = time.time()
            print(f'This took {end - start} seconds')

            print(f'Simulating start {tuple(altinitpos)} without the magnetic'
                  + ' synthetic field')
            start = time.time()
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, eigenstate,
                                     norot, 'Nomag')
                stream.color = 'green'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print('A stream has failed to generate, most probably due to'
                      + ' crossing the centre')
            end = time.time()
            print(f'This took {end - start} seconds')

            print(f'Simulating start {tuple(altinitpos)} without the scalar' 
                  + ' synthetic field')
            start = time.time()
            try:
                stream = sn.solvedyn(tuple(altinitpos[:]), initvel, eigenstate,
                                     norot, 'Noscalar')
                stream.color = 'orange'
                sol.append(stream)
                print(f'Stream number {len(sol)} has been integrated!')
            except Exception as e:
                print(e)
                print('A stream has failed to generate, most probably due to'
                      + ' crossing the centre')
            end = time.time()
            print(f'This took {end - start} seconds')

        else:
            for i in range(initposarray.shape[1]):
                for j in range(initposarray.shape[2]):
                    print(tuple(initposarray[:, i, j]))
                    start = time()
                    try:
                        sol.append(sn.solvedyn(
                            tuple(initposarray[:, i, j]), initvel, eigenstate,
                            norot, nosyn))
                        print(f'Stream number {len(sol)} has been integrated!')
                    except Exception as e:
                        raise e
                        print(e)
                        print('A stream has failed to generate, most probably'
                              + ' due to crossing the centre')
                    end = time.time()
                    print(f'This took {end - start} seconds')

        # Save the result
        print(f'''Saving result to file resultfield{field}nr{nr}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin''')
        with open(f'''saves/odesols/result_field{field}nr{nr}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{eigenstate}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}altstream{alternatestreams}.bin''', 'wb') as file:
            pickle.dump(sol, file)

    else:
        print('Loading previously generated result')
    # Extract positions and orientations of the some stream
    # pos = sol[0].y[0:3, :]
    # vel = sol[0].y[5:8, :]
    ori = sol[0].y[3:5, :]
    # Print the result
    print('Rotation integrated:')
    print(ori)
    # print('Position integrated:')
    # print(pos)

    sn.lineplot(sol, initvel, swarmnum, eigenstate, norot, nosyn,
                alternatestreams)


if __name__ == '__main__':
    main()
