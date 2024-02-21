import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits import mplot3d
from scipy.constants import hbar
from scipy.constants import mu_0 as mu
from scipy.constants import pi as pi
from scipy.integrate import solve_ivp
import numba
from numba import njit
from numba import guvectorize

# This is a collection of scripts related to the finding of eigenstates and
# thence the derivation of the synthetic fields.

# Common variables defined below, are adjusted by synfieldsolver.py:
# Step size of theta
steptheta = np.pi/1e5
# Step size of phi
stepphi = 2*np.pi/1e5
# Number of points to divide the lab length by for numerical differentiation
nr = 1e5
# Side length of environment cube in meters
lablength = 1e-3
# Maximum simulated time in seconds
tmax = 1
# Spin-spin coupling strength
J = 1e9
# Spin-field coupling strength
Gamma = 1e9
# The total mass of the dumbbell in kg, as a placeholder this is the mass of
# two silver atoms
mass0 = 3.58e-25
# The distance between dumbbell edges in m
len0 = 5e-10
# Full mass vector
mass = np.repeat(mass0, 5)
mass[3] = mass0*len0**2/4
mass[4] = mass[3]

field = (10, 1e-3, 10)

# Define statistics variables
dscalaravg = 0
Faavg = 0
denergyavg = 0
energyavg = 0


# Returns the differentiated Hamiltonian w.r.t. the specified par. diffpar=r,
# th_r, ph_r at the point given in point=(x, y, z, th_r, ph_r ). If diffpar=r
# a list of matrices for derivatives w.r.t. x, y, z are returned, otherwise a
# list with the correct matrix in all three positions is returned.
@njit(cache=False, parallel=True, fastmath=False)
def diffhamiltonian(diffpar, pos):

    # Check if the differentiated Hamiltonian has already been calculated here
    #    global diffhamsave
    #    if (diffpar, pos) in diffhamsave:
    #        #print('diffhamsave used')
    #        return diffhamsave[(diffpar, pos)]

    # Differentiation step size
    stepr = lablength/nr
    # Select derivative to return
    if (diffpar == "r"):

        # Initialize return matrices
        diff_hamx = np.zeros((3, 3), dtype=numba.complex64)
        diff_hamy = np.zeros((3, 3), dtype=numba.complex64)
        diff_hamz = np.zeros((3, 3), dtype=numba.complex64)

        # Meshgrid to generate neighbours
        neighgrid = meshgrid(np.arange(-1, 2), np.arange(-1, 2),
                             np.arange(-1, 2))
        # Uses broadcasting to duplicate x,y,z into each point on the grid.
        # The result has first dimension determining which coordinate is given
        # and the remaining specifying position related to the point.

        # Numba does not allow magic indexing, manual expansion of pos is
        # instead necessary
        pos1 = np.stack((pos[0:3], pos[0:3], pos[0:3]), axis=1)
        pos2 = np.stack((pos1, pos1, pos1), axis=2)
        pos3 = np.stack((pos2, pos2, pos2), axis=3)
        neighgrid = pos3 + neighgrid*stepr

        # Find neighbouring external fields
        Bgrid = np.zeros((6, 3, 3, 3))
        Bgrid[:, 1, 1, 1] = oppositecoils(field, neighgrid[:, 1, 1, 1])
        Bgrid[:, 2, 1, 1] = oppositecoils(field, neighgrid[:, 2, 1, 1])
        Bgrid[:, 0, 1, 1] = oppositecoils(field, neighgrid[:, 0, 1, 1])
        Bgrid[:, 1, 2, 1] = oppositecoils(field, neighgrid[:, 1, 2, 1])
        Bgrid[:, 1, 0, 1] = oppositecoils(field, neighgrid[:, 1, 0, 1])
        Bgrid[:, 1, 1, 2] = oppositecoils(field, neighgrid[:, 1, 1, 2])
        Bgrid[:, 1, 1, 0] = oppositecoils(field, neighgrid[:, 1, 1, 0])
        # Magnetic field strength at pos
        B = Bgrid[3, 1, 1, 1]
        # Value of theta_B at pos
        theta = Bgrid[4, 1, 1, 1]
        # Value of phi_B at pos
        phi = Bgrid[5, 1, 1, 1]

        # Approximate the derivatives of B
        dBdr = np.zeros(3)
        dBdr[0] = 0.5*(Bgrid[3, 2, 1, 1]-Bgrid[3, 0, 1, 1])/stepr
        dBdr[1] = 0.5*(Bgrid[3, 1, 2, 1]-Bgrid[3, 1, 0, 1])/stepr
        dBdr[2] = 0.5*(Bgrid[3, 1, 1, 2]-Bgrid[3, 1, 1, 0])/stepr

        # Approximate the derivatives of theta_B
        dthdr = np.zeros(3)
        dthdr[0] = 0.5*(Bgrid[4, 2, 1, 1]-Bgrid[4, 0, 1, 1])/stepr
        dthdr[1] = 0.5*(Bgrid[4, 1, 2, 1]-Bgrid[4, 1, 0, 1])/stepr
        dthdr[2] = 0.5*(Bgrid[4, 1, 1, 2]-Bgrid[4, 1, 1, 0])/stepr

        # Approximate the derivatives of phi_B
        dphdr = np.zeros(3)
        dphdr[0] = 0.5*(Bgrid[5, 2, 1, 1]-Bgrid[5, 0, 1, 1])/stepr
        dphdr[1] = 0.5*(Bgrid[5, 1, 2, 1]-Bgrid[5, 1, 0, 1])/stepr
        dphdr[2] = 0.5*(Bgrid[5, 1, 1, 2]-Bgrid[5, 1, 1, 0])/stepr

        # Assign matrix elements
        # Calculate the sine
        sin = np.sin(theta)
        # Calculate the cosine
        cos = np.cos(theta)
        # Calculates offdiagonals
        omega = ((- dBdr*sin + 1j*B*dphdr*sin -
                  B*dthdr*cos)*np.exp(-1j*phi)/np.sqrt(2))
        diff_hamx[0, 0], diff_hamy[0, 0], diff_hamz[0, 0] = (dBdr*cos
                                                             - B*dthdr*sin)
        diff_hamx[0, 1], diff_hamy[0, 1], diff_hamz[0, 1] = omega
        diff_hamx[1, 0], diff_hamy[1, 0], diff_hamz[1, 0] = np.conjugate(omega)
        diff_hamx[1, 2], diff_hamy[1, 2], diff_hamz[1, 2] = omega
        diff_hamx[2, 1], diff_hamy[2, 1], diff_hamz[2, 1] = np.conjugate(omega)
        diff_hamx[2, 2], diff_hamy[2, 2], diff_hamz[2, 2] = (-dBdr*cos
                                                             + B*dthdr*sin)

        # Return the matrices with correct prefactors
        diff_hamx = Gamma*hbar*diff_hamx
        diff_hamy = Gamma*hbar*diff_hamy
        diff_hamz = Gamma*hbar*diff_hamz

        # Save the result
        # diffhamsave[(diffpar, pos)] = diff_hamx, diff_hamy, diff_hamz

        return diff_hamx, diff_hamy, diff_hamz

    elif (diffpar == "th_r"):

        # Initialize return matrix
        diff_ham = np.zeros((3, 3), dtype=numba.complex64)

        # Assign matrix elements
        # Calculate the sine of twice theta_r
        sin = np.sin(2*pos[3])
        # Calculate the cosine of twice theta_r
        cos = np.cos(2*pos[3])
        # Calculate the complex exponential of phi_r
        exp = np.exp(1j*pos[4])
        # Calculate the square root of two used
        sq = np.sqrt(2)
        # Calculate an often occuring element
        omega = sq*exp*cos

        diff_ham[0, 0] = -sin
        diff_ham[0, 1] = -1*np.conjugate(omega)
        diff_ham[0, 2] = sin/exp**2
        diff_ham[1, 0] = -1*omega
        diff_ham[1, 1] = 2*sin
        diff_ham[1, 2] = np.conjugate(omega)
        diff_ham[2, 0] = exp**2*sin
        diff_ham[2, 1] = omega
        diff_ham[2, 2] = -sin

        # Return the matrix with correct prefactors
        diff_ham = J*hbar*diff_ham

        # diffhamsave[(diffpar, pos)] = diff_ham #Save the result

        return diff_ham, diff_ham, diff_ham

    elif (diffpar == "ph_r"):

        # Initialize return matrix
        diff_ham = np.zeros((3, 3), dtype=numba.complex64)

        # Assign matrix elements
        # Calculate the sine of twice theta_r
        sin = np.sin(2*pos[3])
        # Calculate the square of sine of theta_r
        sin2 = np.sin(pos[3])**2
        # Calculate the complex exponential of phi_r
        exp = np.exp(1j*pos[4])
        # Calculate the square root of two used
        sq = np.sqrt(2)
        # Calculate an often occuring element
        omega = 1j*exp*sin/sq

        diff_ham[0, 1] = -np.conjugate(omega)
        diff_ham[0, 2] = -2j*sin2/exp**2
        diff_ham[1, 0] = -omega
        diff_ham[1, 2] = np.conjugate(omega)
        diff_ham[2, 0] = 2j*sin2*exp**2
        diff_ham[2, 1] = omega

        # Return the matrix with correct prefactors
        diff_ham = J*hbar*diff_ham

        # diffhamsave[(diffpar, pos)] = diff_ham #Save the result

        return diff_ham, diff_ham, diff_ham

    else:
        raise ValueError('Invalid string for differentiation coordinate'
                         + ' passed to diffhamiltonian')


# Solves the eigenvalue problem of the fast Hamiltonian at position pos for
# field of par. field. The position is taken to be of shape
# (x, y, z, theta_r, phi_r). Returns the energies and eigenvectors in pairs
# with ascending energies. The vectors are normalized column vectors in the
# singlet-triplet basis.
@njit(cache=False, parallel=True, fastmath=False)
def eigensolver(pos):

    # warnings.filterwarnings("error")

    # First calculate the fast Hamiltonian at point
    # Initialize empty matrix
    ham = np.zeros((3, 3), dtype=numba.complex64)

    # Find the values of B, theta_B and phi_B at point
    Barray = oppositecoils(field, pos)
    B = Barray[3]
    thetaB = Barray[4]
    phiB = Barray[5]
    xi = J/(Gamma*B)
    if xi == np.inf or np.isnan(xi):
        print('External field is zero at stream! Please correct the field'
              + ' or the streams')

    # Extract theta_r and phi_r at point
    thetar = pos[3]
    phir = pos[4]

    # Assign matrix elements
    # Calculate the cosine of theta_r
    # cosr = np.cos(thetar)
    # Calculate the sine of twice theta_r
    sin2r = np.sin(2*thetar)
    # Calculate the cosine of twice theta_r
    cos2r = np.cos(2*thetar)
    # Calculate the square of sine of theta_r
    sinrsq = np.sin(thetar)**2
    # Calculate the square of cosine of theta_r
    cosrsq = np.cos(thetar)**2
    # Calculate the cosine of theta_B
    cosB = np.cos(thetaB)
    # Calculate the sine of theta_B
    sinB = np.sin(thetaB)
    # Calculate the complex exponential of phi_r
    expr = np.exp(1j*phir)
    # Calculate the complex exponential of phi_B
    expB = np.exp(1j*phiB)
    # Calculate the square root of two often used
    sq = np.sqrt(2)

    ham[0, 0] = xi*cosrsq + cosB
    ham[0, 1] = -sinB / (expB*sq) - xi*sin2r/(expr*sq)
    ham[0, 2] = xi*sinrsq/expr**2
    ham[1, 0] = -expB*sinB/sq - xi*expr*sin2r/sq
    ham[1, 1] = -xi*cos2r
    ham[1, 2] = -sinB/(expB*sq) + xi*sin2r/(expr*sq)
    ham[2, 0] = xi*expr**2*sinrsq
    ham[2, 1] = -expB*sinB/sq + xi*expr*sin2r/sq
    ham[2, 2] = xi*cosrsq-cosB

    # Fix matrix prefactors
    ham = Gamma*B*hbar*ham

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    # Return the result
    return eigenvalues, eigenvectors


# Calculates the synthetic scalar field at position pos for field par. field
# for state number n. The position is taken to be of shape
# (x, y, z, theta_r, phi_r). Returns a tuple of the scalar field value followed
# by the fast energy.
@njit(cache=False, parallel=True, fastmath=False)
def scalarcalc(pos, n):
    # First check if the scalar field has been calculated here before
    # global scalarsave
    # if (pos, n) in scalarsave:
    #     #print('scalarsave used')
    #     return scalarsave[(pos, n)]

    # First retrieve the energies and eigenstates
    energies, eigvec = eigensolver(pos)

    # Differentiate the Hamiltonian w.r.t. each coordinate
    dummyarray = np.empty((3, 3), dtype=numba.complex128)
    dHam = [dummyarray, dummyarray, dummyarray, dummyarray, dummyarray]
    dHam[0], dHam[1], dHam[2] = diffhamiltonian('r', pos)
    dHam[3] = diffhamiltonian('th_r', pos)[0]
    dHam[4] = diffhamiltonian('ph_r', pos)[0]

    # Initialize synthetic scalar
    Phi = 0

    for i in range(5):
        for k in range(3):
            # Remove diagonals
            if not k == n:
                # Braket for formula
                braket = np.vdot(eigvec[:, n], np.dot(dHam[i], eigvec[:, k]))
                # Add up contributions to the synthetic scalar
                Phi += (hbar**2 / (2*mass[i]) *
                        # Note the discard of the imaginary part, numerical
                        # errors otherwise arise.
                        braket*np.conjugate(braket) /
                        (energies[n] - energies[k])**2).real

    returnlist = (Phi, energies[n])
    # scalarsave[(pos, n)] = returnlist

    return returnlist


# Calculates the acceleration due to the synthetic magnetic field and
# summarizes all acceleration contributions. This is done for the position pos,
# the velocity vel as a tuple with the field field for state number n.
# The position and velocity is taken to be of shape (x, y, z, theta_r, phi_r).
# Returns the velocity in m/s (for integration purposes) followed by the
# acceleration of the system in m/s^2. Note that the t argument is a dummy.
@njit(cache=False, parallel=True, fastmath=False)
def acc(t, posvel, n, norot, nosyn, noclass):
    # Extract pos and vel:
    pos = posvel[0:5]
    vel = posvel[5:10]

    stepr = lablength/nr

    # Initialize forces
    Fa = np.zeros(5)
    dscalar = np.zeros(5)
    denergy = np.zeros(5)

    if nosyn == "False" or nosyn == "Noscalar":
        # Find energies and eigenstates at point
        energies, eigvec = eigensolver(pos)

        # Differentiate the Hamiltonian w.r.t. each coordinate
        dummyarray = np.empty((3, 3), dtype=numba.complex128)
        dHam = [dummyarray, dummyarray, dummyarray, dummyarray, dummyarray]
        dHam[0], dHam[1], dHam[2] = diffhamiltonian('r', pos)
        dHam[3] = diffhamiltonian('th_r', pos)[0]
        dHam[4] = diffhamiltonian('ph_r', pos)[0]

        # Calculate the acceleration due to the syn. magnetic field
        for i in range(5):
            for j in range(5):
                # Remove diagonals
                if not j == i:
                    for k in range(3):
                        # Remove diagonals
                        if not k == n:
                            if energies[n] == energies[k]:
                                print('Degenerate fast eigenvalues!')
                            Fa[i] += (-2*hbar * vel[j] /
                                      (energies[n]-energies[k])**2 *
                                      np.imag(np.vdot(eigvec[:, n],
                                                      np.dot(dHam[i],
                                                             eigvec[:, k])) *
                                              np.vdot(eigvec[:, k],
                                                      np.dot(dHam[j],
                                                             eigvec[:, n]))))

        # global Faavg
        # Faavg = (Faavg + np.linalg.norm(Fa))/2

    if nosyn == "False" or nosyn == "Nomag":

        # To get the derivatives of the scalar fields find coordinate values of
        # all neighbouring sites

        # meshgrid to generate neighbours
        neighgrid = meshgridlarge(np.arange(-1, 2), np.arange(-1, 2),
                             np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
        # Fix step sizes
        neighgrid[0:3, :, :, :, :, :] *= stepr
        neighgrid[3, :, :, :, :, :] *= steptheta
        neighgrid[4, :, :, :, :, :] *= stepphi
        # uses broadcasting to duplicate
        # x,y,z into each point on the grid. the result has first dimension
        # determining which coordinate is given and the remaining specifying
        # position related to the point

        # Numba does not allow magic indexing, manual expansion of pos is
        # instead necessary
        pos1 = np.stack((pos, pos, pos), axis=1)
        pos2 = np.stack((pos1, pos1, pos1), axis=2)
        pos3 = np.stack((pos2, pos2, pos2), axis=3)
        pos4 = np.stack((pos3, pos3, pos3), axis=4)
        pos5 = np.stack((pos4, pos4, pos4), axis=5)
        neighgrid = pos5 + neighgrid

        # Find neighbouring scalar field values and eigenstate energies
        scalar = np.zeros((3, 3, 3, 3, 3))
        energy = np.zeros((3, 3, 3, 3, 3))
        scalar[2, 1, 1, 1, 1], energy[2, 1, 1, 1, 1] = (
                scalarcalc(neighgrid[:, 2, 1, 1, 1, 1], n))
        scalar[0, 1, 1, 1, 1], energy[0, 1, 1, 1, 1] = (
                scalarcalc(neighgrid[:, 0, 1, 1, 1, 1], n))
        scalar[1, 2, 1, 1, 1], energy[1, 2, 1, 1, 1] = (
                scalarcalc(neighgrid[:, 1, 2, 1, 1, 1], n))
        scalar[1, 0, 1, 1, 1], energy[1, 0, 1, 1, 1] = (
                scalarcalc(neighgrid[:, 1, 0, 1, 1, 1], n))
        scalar[1, 1, 2, 1, 1], energy[1, 1, 2, 1, 1] = (
                scalarcalc(neighgrid[:, 1, 1, 2, 1, 1], n))
        scalar[1, 1, 0, 1, 1], energy[1, 1, 0, 1, 1] = (
                scalarcalc(neighgrid[:, 1, 1, 0, 1, 1], n))
        scalar[1, 1, 1, 2, 1], energy[1, 1, 1, 2, 1] = (
                scalarcalc(neighgrid[:, 1, 1, 1, 2, 1], n))
        scalar[1, 1, 1, 0, 1], energy[1, 1, 1, 0, 1] = (
                scalarcalc(neighgrid[:, 1, 1, 1, 0, 1], n))
        scalar[1, 1, 1, 1, 2], energy[1, 1, 1, 1, 2] = (
                scalarcalc(neighgrid[:, 1, 1, 1, 1, 2], n))
        scalar[1, 1, 1, 1, 0], energy[1, 1, 1, 1, 0] = (
                scalarcalc(neighgrid[:, 1, 1, 1, 1, 0], n))

        # Differentiate the scalar field
        dscalar[0] = (scalar[2, 1, 1, 1, 1] - scalar[0, 1, 1, 1, 1])/(2*stepr)
        dscalar[1] = (scalar[1, 2, 1, 1, 1] - scalar[1, 0, 1, 1, 1])/(2*stepr)
        dscalar[2] = (scalar[1, 1, 2, 1, 1] - scalar[1, 1, 0, 1, 1])/(2*stepr)
        dscalar[3] = (scalar[1, 1, 1, 2, 1] -
                      scalar[1, 1, 1, 0, 1])/(2*steptheta)
        dscalar[4] = (scalar[1, 1, 1, 1, 2] -
                      scalar[1, 1, 1, 1, 0])/(2*stepphi)

        # global dscalaravg
        # dscalaravg = (dscalaravg + np.linalg.norm(dscalar))/2

    elif nosyn == "True" or nosyn == "Noscalar":
        # To get the derivatives of the energies find coordinate values of all
        # neighbouring sites.

        # meshgrid to generate neighbours
        neighgrid = meshgridlarge(np.arange(-1, 2), np.arange(-1, 2),
                                  np.arange(-1, 2), np.arange(-1, 2),
                                  np.arange(-1, 2))
        # Fix step sizes
        neighgrid[0:3, :, :, :, :, :] *= stepr
        neighgrid[3, :, :, :, :, :] *= steptheta
        neighgrid[4, :, :, :, :, :] *= stepphi
        # Uses broadcasting to duplicate x,y,z into each point on the grid.
        # The result has first dimension determining which coordinate is given
        # and the remaining specifying position related to the point.

        # Numba does not allow magic indexing, manual expansion of pos is
        # instead necessary
        pos1 = np.stack((pos, pos, pos), axis=1)
        pos2 = np.stack((pos1, pos1, pos1), axis=2)
        pos3 = np.stack((pos2, pos2, pos2), axis=3)
        pos4 = np.stack((pos3, pos3, pos3), axis=4)
        pos5 = np.stack((pos4, pos4, pos4), axis=5)
        neighgrid = pos5 + neighgrid

        # Find energies at neighbouring points
        energy = np.zeros((3, 3, 3, 3, 3))
        energy[2, 1, 1, 1, 1] = eigensolver(neighgrid[:, 2, 1, 1, 1, 1])[0][n]
        energy[0, 1, 1, 1, 1] = eigensolver(neighgrid[:, 0, 1, 1, 1, 1])[0][n]
        energy[1, 2, 1, 1, 1] = eigensolver(neighgrid[:, 1, 2, 1, 1, 1])[0][n]
        energy[1, 0, 1, 1, 1] = eigensolver(neighgrid[:, 1, 0, 1, 1, 1])[0][n]
        energy[1, 1, 2, 1, 1] = eigensolver(neighgrid[:, 1, 1, 2, 1, 1])[0][n]
        energy[1, 1, 0, 1, 1] = eigensolver(neighgrid[:, 1, 1, 0, 1, 1])[0][n]
        energy[1, 1, 1, 2, 1] = eigensolver(neighgrid[:, 1, 1, 1, 2, 1])[0][n]
        energy[1, 1, 1, 0, 1] = eigensolver(neighgrid[:, 1, 1, 1, 0, 1])[0][n]
        energy[1, 1, 1, 1, 2] = eigensolver(neighgrid[:, 1, 1, 1, 1, 2])[0][n]
        energy[1, 1, 1, 1, 0] = eigensolver(neighgrid[:, 1, 1, 1, 1, 0])[0][n]
        energy[1, 1, 1, 1, 1] = eigensolver(neighgrid[:, 1, 1, 1, 1, 1])[0][n]

    else:
        print(f'Incorrect parameter "nosyn" = {nosyn}')

    # global energyavg
    # energyavg = (energyavg + energy[1, 1, 1, 1, 1])/2

    # Differentiate the energies if noclass(ical) is false
    if not noclass:
        denergy[0] = (energy[2, 1, 1, 1, 1] - energy[0, 1, 1, 1, 1])/(2*stepr)
        denergy[1] = (energy[1, 2, 1, 1, 1] - energy[1, 0, 1, 1, 1])/(2*stepr)
        denergy[2] = (energy[1, 1, 2, 1, 1] - energy[1, 1, 0, 1, 1])/(2*stepr)
        denergy[3] = (energy[1, 1, 1, 2, 1] - energy[1, 1, 1, 0, 1])/(2*steptheta)
        denergy[4] = (energy[1, 1, 1, 1, 2] - energy[1, 1, 1, 1, 0])/(2*stepphi)

    # Summarize forces
    acc = Fa - dscalar - denergy

    # global denergyavg
    # denergyavg = (denergyavg + np.linalg.norm(denergy))/2

    for i in range(5):
        # Divide by mass to get acceleration
        acc[i] = acc[i]/mass[i]

    # Freeze rotational axes if norot is turned on
    if norot:
        vel[3:5] = 0
        acc[3:5] = 0

    if acc[3] > 1e10:
        print('The rotation is out of control!')

    return np.concatenate((vel, acc))


# Solves the ODE and returns the solution as per sscipy.integrate.solve_ivp.
# The external magnetic field is given as field. The
# dumbbell is placed initially at position pos and with velocity vel of the
# shape (x, y, z, theta_r, phi_r). Note that cartesian position here is in m.
# The spin subsystem is assumed to remain in the fast
# eigenstate labeled n. Runs until the time reaches tmax.
# The given initial conditions must be tuples.
def solvedyn(pos, vel, n, norot=False, nosyn='False', noclass=False):
    posvel = np.array(pos + vel)

    # Reset average force counters
    # global Faavg
    # global dscalaravg
    # global denergyavg
    # global energyavg
    # Faavg = 0
    # dscalaravg = 0
    # denergyavg = 0
    # energyavg = 0

    # Find times to require ODE evaluation
    t_eval = np.linspace(0, tmax, 100000)

    # Set error tolerances
    edgedistance.terminal = True
    sol = solve_ivp(acc, (0, tmax), posvel, events=edgedistance,
                    args=(n, norot, nosyn, noclass), t_eval=t_eval)

    # sol.Faavg = Faavg
    # sol.denergyavg = denergyavg
    # sol.dscalaravg = dscalaravg
    # sol.energyavg = energyavg

    return sol


# Meshgrid function to replace np.mgrid not supported by Numba
@numba.njit(cache=False, fastmath=False)
def meshgrid(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=numba.float64)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=numba.float64)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=numba.float64)
    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                xx[i, j, k] = x[i]
                yy[i, j, k] = y[j]
                zz[i, j, k] = z[k]
    return np.stack((xx, yy, zz), axis=0)


# Meshgrid function to replace np.mgrid not supported by Numba, but larger
@numba.njit(cache=False, fastmath=False)
def meshgridlarge(x, y, z, p, q):
    xx = np.empty(shape=(x.size, y.size, z.size, p.size, q.size),
                  dtype=numba.float64)
    yy = np.empty(shape=(x.size, y.size, z.size, p.size, q.size),
                  dtype=numba.float64)
    zz = np.empty(shape=(x.size, y.size, z.size, p.size, q.size),
                  dtype=numba.float64)
    pp = np.empty(shape=(x.size, y.size, z.size, p.size, q.size),
                  dtype=numba.float64)
    qq = np.empty(shape=(x.size, y.size, z.size, p.size, q.size),
                  dtype=numba.float64)
    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                for l in range(p.size):
                    for m in range(q.size):
                        xx[i, j, k, l, m] = x[i]
                        yy[i, j, k, l, m] = y[j]
                        zz[i, j, k, l, m] = z[k]
                        pp[i, j, k, l, m] = p[l]
                        qq[i, j, k, l, m] = q[m]
    return np.stack((xx, yy, zz, pp, qq), axis=0)


# Event to terminate integration, returns distance to the closest edge minus a
# small correction to avoid hitting the edge.
@njit(cache=False, parallel=True, fastmath=False)
def edgedistance(t, posvel, n, norot, nosyn, noclass):
    pos = posvel[0:3]
    mindist = pos.min()
    maxdist = pos.max()
    distancetoedge = min(mindist+lablength/1000,
                         lablength - maxdist+lablength/1000)

    return distancetoedge


# Returns the field at position pos corresponding to two currents I of
# opposing directions through circular coils
# placed orthogonally to the z-axis centred 1/3th from the edges of the
# lab. The coil diameter is 5/3 times the lab size.
@njit(cache=False, parallel=True, fastmath=False)
def oppositecoils(field, pos):

    # Initiate variables:
    br = int(field[0])
    lablength = field[1]
    current = field[2]
    # Length of each wire segment
    stepr = 5/3*lablength*pi/br
    # Radius of coils
    crad = 5/6*lablength
    # Positions and directions of all flowing currents:
    currentpos = np.array([t for t in np.linspace(0.0, 2*pi, br+1)[0:-1]])
    currentpos = crad*np.vstack((np.cos(currentpos), np.sin(currentpos)))
    centerpos = np.array([lablength/2]*br)
    currentpos += np.vstack((centerpos, centerpos))
    currentdir = np.array([t for t in np.linspace(0.0, 2*pi, br+1)[0:-1]])
    currentdir = stepr*np.vstack((-1*np.sin(currentdir), np.cos(currentdir)))

    # Generate field at pos
    # Integrate the field per Biot-Savart along all currents
    B = np.array((0, 0, 0), dtype=numba.float64)
    for i in range(br):
        distance1 = np.array((pos[0]-currentpos[0, i], pos[1]-currentpos[1, i],
                             pos[2]-(4/3*lablength)))
        distance2 = np.array((pos[0]-currentpos[0, i], pos[1]-currentpos[1, i],
                             pos[2]+1/3*lablength))
        dB = (mu*current/(4*pi) * (
            np.cross(currentdir[:, i], distance1)/(np.linalg.norm(distance1)**3)
            + np.cross(-1*currentdir[:, i], distance2)/(np.linalg.norm(distance2)**3))
              )
        B += dB
    Bsph = cart_to_sph(B)  # Express as spherical coordinates
    retfield = np.append(B, Bsph)

    return retfield


@njit(cache=True, parallel=True, fastmath=False)
def griddot(a, b):
    ##Returns the dot product for each point in the supplied grids a, b. Contracts the
    ##first dimension.
    result = np.zeros(a[0].shape)
    for i in range(len(a)):
        result += a[i]*b[i]

    #result = np.where(result == 0, 1e-20, result)
    return result


@njit(cache=True, parallel=True, fastmath=False)
def cart_to_sph(cart):
    ##Returns spherical coordinates of the form(r, polar, azimuthal) for the given cartesian 
    ##coordinates of the form (x,y,z) takes an array with coords as the first dimension.
    sph = np.zeros(cart.shape) #Initialize array
    xsqysq = cart[0]**2 + cart[1]**2 #Value of x^2 + y^2
    sph[0] = np.sqrt(xsqysq + cart[2]**2) #Radius r
    sph[1] = np.arctan2(np.sqrt(xsqysq), cart[2]) #Polar angle theta
    sph[2] = np.arctan2(cart[1],cart[0]) #Azimuthal angle phi
    return sph


# Takes an array of positions of shape (timestamp, pos) and returns an array of
# corresponding external field values of shape (timestamp, fields)
@guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:])],
             '(n),(f)->(n)', nopython=True, cache=False)
def parameterspace(pos, f, parpos):
    # Find stream coordinates in parameter space
    parpos[0:3] = oppositecoils(f, pos)[0:3]


# Plotting function, takes a list of solutions sol from solve_ivp, a field
# field and displays an interactive 3D swarm plot. Uses matplotlib.
def lineplot(sol, timestamp, quiver, parameter, difference, noplot):

    # Print average acceleration components
    # for stream in sol:
    #     print(f'Statistics for stream starting at {stream.y[0:3,0]}')
    #     print(f'Faavg = {stream.Faavg}')
    #     print(f'dscalaravg = {stream.dscalaravg}')
    #     print(f'denergyavg = {stream.denergyavg}')
    #     print(f'energyavg = {stream.energyavg}')

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')

    if not parameter:
        ax.set_xlim((0, lablength*1000))
        ax.set_ylim((0, lablength*1000))
        ax.set_zlim((0, lablength*1000))
        ax.set_xlabel('x (mm)', fontsize=10, color='blue')
        ax.set_ylabel('y (mm)', fontsize=10, color='blue')
        ax.set_zlabel('z (mm)', fontsize=10, color='blue')
        if difference:
            boxmax = 0
            for stream in sol:
                smax = stream.y[0:3, :].max()
                if smax > boxmax:
                    boxmax = smax
                smin = stream.y[0:3, :].min()
                if -smin > boxmax:
                    boxmax = -smin
            ax.set_xlim((-boxmax*1000, boxmax*1000))
            ax.set_ylim((-boxmax*1000, boxmax*1000))
            ax.set_zlim((-boxmax*1000, boxmax*1000))
    else:
        for stream in sol:
            stream.p = parameterspace(stream.y[0:3, :].swapaxes(0, 1), field)
            stream.p = stream.p.swapaxes(0, 1)
        print('Fitting done!')
        boxmax = 0
        for stream in sol:
            smax = stream.p[0:3, :].max()
            if smax > boxmax:
                boxmax = smax
            smin = stream.y[0:3, :].min()
            if -smin > boxmax:
                boxmax = -smax
        ax.set_xlim((-boxmax, boxmax))
        ax.set_ylim((-boxmax, boxmax))
        ax.set_zlim((-boxmax, boxmax))
        ax.set_xlabel('x (T)', fontsize=10, color='blue')
        ax.set_ylabel('y (T)', fontsize=10, color='blue')
        ax.set_zlabel('z (T)', fontsize=10, color='blue')

    # Display external field if quiver is True
    if quiver:
        # Create grids for the quiver:
        qgran = 11
        xx, yy, zz = np.mgrid[0:qgran, 0:qgran, 0:qgran]
        magarray = np.empty((3, qgran, qgran, qgran))
        xx = lablength/(qgran - 1) * xx
        yy = lablength/(qgran - 1) * yy
        zz = lablength/(qgran - 1) * zz
        posarray = np.stack((xx, yy, zz), axis=0)
        for i in np.ndindex(magarray[0, :, :, :].shape):
            magarray[:, i[0], i[1], i[2]] = oppositecoils(field,
                                            posarray[:, i[0], i[1], i[2]])[0:3]
        # Display positions in mm
        xx = xx*1000
        yy = yy*1000
        zz = zz*1000
        ax.quiver(xx, yy, zz, magarray[0, :, :, :], magarray[1, :, :, :],
                  magarray[2, :, :, :], length=0.1, normalize=True)

    for stream in sol:
        # Extract pos
        # Set stream colour
        try:
            color = stream.color
        except AttributeError:
            color = 'red'
        if not parameter:
            pos = stream.y[0:3, :]
            # Plot in mm
            pos = pos*1000
        else:
            pos = stream.p[0:3, :]
        # Plot the integrated path
        ax.plot(pos[0, :], pos[1, :], pos[2, :], color=color)
        # ax.plot(pos[0, -1], pos[1, -1], pos[2, -1], color=color, marker='o')

    plt.savefig(f'''saves/graphs/{timestamp}.png''', bbox_inches='tight')

    if not noplot:
        plt.show()


# Prints magnetic field of the box, for debugging
# Need fixing
def debugquiver():
    lineplot((), "test", True, False, False, False)

# Calculates velocity towards centre of par.space for position pos


def towardmonopole(pos, field):

    stepr = lablength/nr

    # Meshgrid to generate neighbours
    neighgrid = meshgrid(np.arange(-1, 2), np.arange(-1, 2),
                         np.arange(-1, 2))
    # Uses broadcasting to duplicate x,y,z into each point on the grid.
    # The result has first dimension determining which coordinate is given
    # and the remaining specifying position related to the point.

    # Numba does not allow magic indexing, manual expansion of pos is
    # instead necessary
    pos1 = np.stack((pos[0:3], pos[0:3], pos[0:3]), axis=1)
    pos2 = np.stack((pos1, pos1, pos1), axis=2)
    pos3 = np.stack((pos2, pos2, pos2), axis=3)
    neighgrid = pos3 + neighgrid*stepr
    # Find neighbouring external fields
    Bgrid = np.zeros((6, 3, 3, 3))
    Bgrid[:, 1, 1, 1] = oppositecoils(field, neighgrid[:, 1, 1, 1])
    Bgrid[:, 2, 1, 1] = oppositecoils(field, neighgrid[:, 2, 1, 1])
    Bgrid[:, 0, 1, 1] = oppositecoils(field, neighgrid[:, 0, 1, 1])
    Bgrid[:, 1, 2, 1] = oppositecoils(field, neighgrid[:, 1, 2, 1])
    Bgrid[:, 1, 0, 1] = oppositecoils(field, neighgrid[:, 1, 0, 1])
    Bgrid[:, 1, 1, 2] = oppositecoils(field, neighgrid[:, 1, 1, 2])
    Bgrid[:, 1, 1, 0] = oppositecoils(field, neighgrid[:, 1, 1, 0])

    # Approximate the Jacobian of B
    dBdr = np.zeros((3, 3))
    dBdr[0, 0] = 0.5*(Bgrid[0, 2, 1, 1]-Bgrid[0, 0, 1, 1])/stepr
    dBdr[0, 1] = 0.5*(Bgrid[0, 1, 2, 1]-Bgrid[0, 1, 0, 1])/stepr
    dBdr[0, 2] = 0.5*(Bgrid[0, 1, 1, 2]-Bgrid[0, 1, 1, 0])/stepr
    dBdr[1, 0] = 0.5*(Bgrid[1, 2, 1, 1]-Bgrid[1, 0, 1, 1])/stepr
    dBdr[1, 0] = 0.5*(Bgrid[1, 2, 1, 1]-Bgrid[1, 0, 1, 1])/stepr
    dBdr[1, 1] = 0.5*(Bgrid[1, 1, 2, 1]-Bgrid[1, 1, 0, 1])/stepr
    dBdr[2, 2] = 0.5*(Bgrid[2, 1, 1, 2]-Bgrid[2, 1, 1, 0])/stepr
    dBdr[2, 1] = 0.5*(Bgrid[2, 1, 2, 1]-Bgrid[2, 1, 0, 1])/stepr
    dBdr[2, 2] = 0.5*(Bgrid[2, 1, 1, 2]-Bgrid[2, 1, 1, 0])/stepr

    # Solve the lin sys giving the direction
    v = np.linalg.solve(dBdr, Bgrid[0:3, 1, 1, 1])
    return v
