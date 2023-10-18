import synfieldsolver as sfs
import multiprocessing
import numpy as np
from itertools import product


# Iterates sunfieldsolver through ranges of parameters
def main():
    norot = True
    nosyn = 'False'

    difference = False
    noclass = True
    overwriteresult = False
    br = 10
    nr = 1e5
    lablength = 1e-3
    tmax = 2
    noplot = False
    quiver = False
    alternatestreams = True
    parameter = False
    # Ranges of looped par. of the form (start, stop, step)
    currentr = (10, 10, 1)
    jr = (0, 0, 1)
    gammar = (1e8, 1e8, 1)  # Standard 1e8
    massr = (3.58e-29, 3.58e-29, 1)  # Standard 3.58e-27
    dlr = (5e-5, 5e-5, 1)  # dumbellength
    vxr = (0, 0, 1)
    vyr = (0, 0, 1)
    vzr = (1.5e-3, 1.5e-3, 1)
    vthetar = (0, 0, 1)
    vphir = (0, 0, 1)
    xr = (13/30, 14/30, 3)
    yr = (1/2, 1/2, 1)
    zr = (0, 0, 1)
    thetar = (0, 0, 1)
    phir = (0, 0, 1)

    for current, J, Gamma, mass0, len0 in product(
                                          np.linspace(currentr[0], currentr[1],
                                                      currentr[2]),
                                          np.linspace(jr[0], jr[1], jr[2]),
                                          np.linspace(gammar[0], gammar[1],
                                                      gammar[2]),
                                          np.linspace(massr[0], massr[1],
                                                      massr[2]),
                                          np.linspace(dlr[0], dlr[1], dlr[2])):
        for velx, vely, velz, velth, velphi in product(np.linspace(vxr[0],
                                                                   vxr[1],
                                                                   vxr[2]),
                                                       np.linspace(vyr[0],
                                                                   vyr[1],
                                                                   vyr[2]),
                                                       np.linspace(vzr[0],
                                                                   vzr[1],
                                                                   vzr[2]),
                                                       np.linspace(vthetar[0],
                                                                   vthetar[1],
                                                                   vthetar[2]),
                                                       np.linspace(vphir[0],
                                                                   vphir[1],
                                                                   vphir[2])):
            for x, y, z, th, phi in product(np.linspace(xr[0], xr[1], xr[2]),
                                            np.linspace(yr[0], yr[1], yr[2]),
                                            np.linspace(zr[0], zr[1], zr[2]),
                                            np.linspace(thetar[0], thetar[1],
                                                        thetar[2]),
                                            np.linspace(phir[0], phir[1],
                                                        phir[2])):
                initpos = np.array((x, y, z, th, phi))
                initvel = (velx, vely, velz, velth, velphi)
                p = multiprocessing.Process(target=sfs.main, name="sfs",
                                            args=(current, br, norot, nosyn,
                                                  noclass,
                                                  difference, noplot,
                                                  overwriteresult,
                                                  alternatestreams, quiver,
                                                  parameter, nr, lablength,
                                                  tmax, initvel,
                                                  initpos, J, Gamma,
                                                  mass0, len0))
                p.start()

                # Terminate process after timeout seconds
                timeout = 9000
                p.join(timeout)
                if p.is_alive():
                    print('Synfieldsolver '
                          + f'current{current}vel{initvel}pos{initpos}J{J}Gamma{Gamma}mass{mass0}len0{len0}'
                          + ' took more than {timeout}'
                          + 'seconds to finish, skipping parameter selection')
                    p.terminate()
                    p.join()


if __name__ == '__main__':
    main()
