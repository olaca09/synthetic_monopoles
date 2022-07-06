import pandas as pd
import numpy as np
import scipy
import magfield as mg
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from scipy.constants import hbar
from scipy.integrate import solve_ivp
import warnings

##This is a collection of scripts related to the finding of eigenstates and thence the
##derivation of the synthetic fields.

#Common variables defined below, are adjusted by synfieldsolver.py:
steptheta = np.pi/1e5 #Step size of theta
stepphi = 2*np.pi/1e5 #Step size of phi
nr = 1e5 #Number of points to divide the lab length by for numerical differentiation
         #step size
lablength = 1e-3 #Side length of environment cube in meters
tmax = 1 #Maximum simulated time in seconds

J = 1e9 #Spin-spin coupling strength
Gamma = 1e9 #Spin-field coupling strength

mass0 = 3.58e-25 #The total mass of the dumbbell in kg, as a placeholder this is the mass of two
                 #silver atoms
len0 = 5e-10 #The distance between dumbbell edges in m
mass = np.repeat(mass0, 5) #Full mass vector
mass[3] = mass0*len0**2/4
mass[4] = mass[3]

field = (0,0,0)

#Define statistics variables
dscalaravg=0
Faavg=0
denergyavg=0
energyavg=0


##Returns the differentiated Hamiltonian w.r.t. the specified par. diffpar=r, th_r, ph_r at
##the point given in point=(x, y, z, th_r, ph_r ) in the external field field.
##If diffpar=r a list of matrices for derivatives w.r.t. x, y, z are returned
#diffhamsave = {}
def diffhamiltonian(diffpar, pos):

    #Check if the differentiated Hamiltonian has already been calculated here
#    global diffhamsave
#    if (diffpar, pos) in diffhamsave:
#        #print('diffhamsave used')
#        return diffhamsave[(diffpar, pos)]

    #Find nr and stepr
    stepr = lablength/nr #Differentiation step size
    #Select derivative to return
    if(diffpar == "r"):

        #Initialize return matrices
        diff_hamx = np.zeros([3, 3], dtype='complex_')
        diff_hamy = np.zeros([3, 3], dtype='complex_')
        diff_hamz = np.zeros([3, 3], dtype='complex_')

        neighgrid = np.mgrid[-1:2,-1:2,-1:2].astype('float') ##Meshgrid to generate neighbours
        neighgrid = np.array(pos[0:3])[:,None,None,None] + neighgrid*stepr ##Uses broadcasting to duplicate
	    #x,y,z into each point on the grid. The result has first dimension determining which 
	    #coordinate is given and the remaining specifying position related to the point

        #Find neighbouring external fields
        Bgrid = np.zeros([6,3,3,3])
        Bgrid[:,1,1,1] = mg.oppositecoilspos(field, neighgrid[:,1,1,1])
        Bgrid[:,2,1,1] = mg.oppositecoilspos(field, neighgrid[:,2,1,1])
        Bgrid[:,0,1,1] = mg.oppositecoilspos(field, neighgrid[:,0,1,1])
        Bgrid[:,1,2,1] = mg.oppositecoilspos(field, neighgrid[:,1,2,1])
        Bgrid[:,1,0,1] = mg.oppositecoilspos(field, neighgrid[:,1,0,1])
        Bgrid[:,1,1,2] = mg.oppositecoilspos(field, neighgrid[:,1,1,2])
        Bgrid[:,1,1,0] = mg.oppositecoilspos(field, neighgrid[:,1,1,0])
        B = Bgrid[3,1,1,1] #Magnetic field strength at pos
        theta = Bgrid[4,1,1,1] #Value of theta_B at pos
        phi = Bgrid[5,1,1,1] #Value of phi_B at pos

        #Approximate the derivatives of B
        dBdr = np.zeros(3)
        dBdr[0] = 0.5*(Bgrid[3,2,1,1]-Bgrid[3,0,1,1])/stepr
        dBdr[1] = 0.5*(Bgrid[3,1,2,1]-Bgrid[3,1,0,1])/stepr
        dBdr[2] = 0.5*(Bgrid[3,1,1,2]-Bgrid[3,1,1,0])/stepr

        #Approximate the derivatives of theta_B
        dthdr = np.zeros(3)
        dthdr[0] = 0.5*(Bgrid[4,2,1,1]-Bgrid[4,0,1,1])/stepr
        dthdr[1] = 0.5*(Bgrid[4,1,2,1]-Bgrid[4,1,0,1])/stepr
        dthdr[2] = 0.5*(Bgrid[4,1,1,2]-Bgrid[4,1,1,0])/stepr

        #Approximate the derivatives of phi_B
        dphdr = np.zeros(3)
        dphdr[0] = 0.5*(Bgrid[5,2,1,1]-Bgrid[5,0,1,1])/stepr
        dphdr[1] = 0.5*(Bgrid[5,1,2,1]-Bgrid[5,1,0,1])/stepr
        dphdr[2] = 0.5*(Bgrid[5,1,1,2]-Bgrid[5,1,1,0])/stepr

        #Assign matrix elements
        sin = np.sin(theta) #Calculate the sine
        cos = np.cos(theta) #Calculate the cosine
        omega = (- dBdr*sin + 1j*B*dphdr*sin - B*dthdr*cos)*np.exp(-1j*phi)/np.sqrt(2) #Calculates offdiagonals
        diff_hamx[0,0], diff_hamy[0,0], diff_hamz[0,0] = (dBdr*cos - B*dthdr*sin)
        diff_hamx[0,1], diff_hamy[0,1], diff_hamz[0,1] = omega
        diff_hamx[1,0], diff_hamy[1,0], diff_hamz[1,0] = np.conjugate(omega)
        diff_hamx[1,2], diff_hamy[1,2], diff_hamz[1,2] = omega
        diff_hamx[2,1], diff_hamy[2,1], diff_hamz[2,1] = np.conjugate(omega)
        diff_hamx[2,2], diff_hamy[2,2], diff_hamz[2,2] = (-dBdr*cos + B*dthdr*sin)

        #Return the matrices with correct prefactors
        diff_hamx = Gamma*hbar*diff_hamx
        diff_hamy = Gamma*hbar*diff_hamy
        diff_hamz = Gamma*hbar*diff_hamz

        #diffhamsave[(diffpar, pos)] = diff_hamx, diff_hamy, diff_hamz #Save the result

        return diff_hamx, diff_hamy, diff_hamz

    elif(diffpar=="th_r"):
        
        diff_ham  = np.zeros([3,3], dtype='complex_') #Initialize return matrix

        #Assign matrix elements
        sin = np.sin(2*pos[3]) #Calculate the sine of twice theta_r
        cos = np.cos(2*pos[3]) #Calculate the cosine of twice theta_r
        exp = np.exp(1j*pos[4]) #Calculate the complex exponential of phi_r
        sq = np.sqrt(2) #Calculate the square root of two used
        omega = sq*exp*cos # Calculate an often occuring element

        diff_ham[0,0] = -sin
        diff_ham[0,1] = -1*np.conjugate(omega)
        diff_ham[0,2] = sin/exp**2
        diff_ham[1,0] = -1*omega
        diff_ham[1,1] = 2*sin
        diff_ham[1,2] = np.conjugate(omega)
        diff_ham[2,0] = exp**2*sin
        diff_ham[2,1] = omega
        diff_ham[2,2] = -sin

        #Return the matrix with correct prefactors
        diff_ham = J*hbar*diff_ham

        #diffhamsave[(diffpar, pos)] = diff_ham #Save the result

        return diff_ham

    elif(diffpar=="ph_r"):
        
        diff_ham  = np.zeros([3,3], dtype='complex_') #Initialize return matrix

        #Assign matrix elements
        sin = np.sin(2*pos[3]) #Calculate the sine of twice theta_r
        sin2 = np.sin(pos[3])**2 #Calculate the square of sine of theta_r
        exp = np.exp(1j*pos[4]) #Calculate the complex exponential of phi_r
        sq = np.sqrt(2) #Calculate the square root of two used
        omega = 1j*exp*sin/sq # Calculate an often occuring element

        diff_ham[0,1] = -np.conjugate(omega)
        diff_ham[0,2] = -2j*sin2/exp**2
        diff_ham[1,0] = -omega
        diff_ham[1,2] = np.conjugate(omega)
        diff_ham[2,0] = 2j*sin2*exp**2
        diff_ham[2,1] = omega

        #Return the matrix with correct prefactors
        diff_ham = J*hbar*diff_ham

        #diffhamsave[(diffpar, pos)] = diff_ham #Save the result

        return diff_ham

    else:
        raise ValueError("Invalid string for differentiation coordinate passed to diffhamiltonian")


##Solves the eigenvalue problem of the fast Hamiltonian at position pos for field of par. field.
##The position is taken to be of shape (x, y, z, theta_r, phi_r).
##Returns the energies and eigenvectors in pairs with ascending energies. The vectors are
##normalized column vectors in the singlet-triplet basis.
def eigensolver(pos):

    warnings.filterwarnings("error")
    
    #First calculate the fast Hamiltonian at point
    ham = np.zeros([3,3], dtype='complex_') #Initialize empty matrix

    #Find the values of B, theta_B and phi_B at point
    Barray = mg.oppositecoilspos(field, pos)
    B = Barray[3]
    thetaB = Barray[4]
    phiB = Barray[5]
    xi = J/(Gamma*B)
    if xi == np.inf or np.isnan(xi):
        print(f'External field is zero at point {point}! Please correct the field or the streams')

    #Extract theta_r and phi_r at point
    thetar = pos[3]
    phir = pos[4]

    #Assign matrix elements
    cosr = np.cos(thetar) #Calculate the cosine of theta_r
    sin2r = np.sin(2*thetar) #Calculate the sine of twice theta_r
    cos2r = np.cos(2*thetar) #Calculate the cosine of twice theta_r
    sinrsq = np.sin(thetar)**2 #Calculate the square of sine of theta_r
    cosrsq = np.cos(thetar)**2 #Calculate the square of cosine of theta_r
    cosB = np.cos(thetaB) #Calculate the cosine of theta_B
    sinB = np.sin(thetaB) #Calculate the sine of theta_B
    expr = np.exp(1j*phir) #Calculate the complex exponential of phi_r
    expB = np.exp(1j*phiB) #Calculate the complex exponential of phi_B
    sq = np.sqrt(2) #Calculate the square root of two often used

    ham[0,0] = xi*cosrsq + cosB
    ham[0,1] = -sinB / (expB*sq) - xi*sin2r/(expr*sq)
    ham[0,2] = xi*sinrsq/expr**2
    ham[1,0] = -expB*sinB/sq - xi*expr*sin2r/sq
    ham[1,1] = -xi*cos2r
    ham[1,2] = -sinB/(expB*sq) + xi*sin2r/(expr*sq)
    ham[2,0] = xi*expr**2*sinrsq
    ham[2,1] = -expB*sinB/sq + xi*expr*sin2r/sq
    ham[2,2] = xi*cosrsq-cosB
    
    #Fix matrix prefactors
    ham = Gamma*B*hbar*ham

    #Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(ham)

    #Return the result
    return eigenvalues, eigenvectors

##Calculates the synthetic scalar field at position pos for field par. field for state number n.
##The position is taken to be of shape (x, y, z, theta_r, phi_r).
##Returns a tuple of the scalar field value followed by the fast energy.
#scalarsave = {}
def scalarcalc(pos, n):


    #First check if the scalar field has been calculated here before
    #global scalarsave
#    if (pos, n) in scalarsave:
#        #print('scalarsave used')
#        return scalarsave[(pos, n)]

    #First retrieve the energies and eigenstates
    energies, eigvec = eigensolver(pos)
    
    #Differentiate the Hamiltonian w.r.t. each coordinate
    dHam = [0,0,0,0,0]
    dHam[0], dHam[1], dHam[2] = diffhamiltonian('r', pos)
    dHam[3] = diffhamiltonian('th_r', pos)
    dHam[4] = diffhamiltonian('ph_r', pos)

    Phi = 0 #Initialize synthetic scalar

    for i in range(5):
        for l in range(3):
            if not l == n: #Remove diagonals
                braket = np.vdot(eigvec[n], np.dot(dHam[i], eigvec[l])) #Braket for formula
                Phi += (hbar**2 /(2*mass[i]) * #Add up contributions to the synthetic scalar
                braket*np.conjugate(braket) / (energies[n] - energies[l])**2).real #Note the discard of the imaginary part, numerical errors otherwise arise 

    returnlist = (Phi, energies[n])
    #scalarsave[(pos, n)] = returnlist

    return returnlist

##Calculates the acceleration due to the synthetic magnetic field and summarizes all
##acceleration contributions. This is done for the position pos, the velocity vel as a tuple
##with the field field for state number n. The position and velocity is taken to be of 
##shape (x, y, z, theta_r, phi_r).
##Returns the velocity in m/s (for integration purposes) followed by the acceleration of the 
##system in m/s^2.
##Note that the t argument is a dummy.
def acc(t, posvel, n, norot, nosyn):
    #Extract pos and vel:
    pos = posvel[0:5]
    vel = posvel[5:10]

    stepr = lablength/nr

    #Initialize forces
    Fa = np.zeros(5)
    dscalar = np.zeros(5)
    denergy = np.zeros(5)

    if nosyn == "False" or nosyn == "Noscalar":
        #Find energies and eigenstates at point
        energies, eigvec = eigensolver(pos)
    
        #Differentiate the Hamiltonian w.r.t. each coordinate
        dHam = [0,0,0,0,0]
        dHam[0], dHam[1], dHam[2] = diffhamiltonian('r', pos)
        dHam[3] = diffhamiltonian('th_r', pos)
        dHam[4] = diffhamiltonian('ph_r', pos)
    
        #Calculate the acceleration due to the syn. magnetic field
        for i in range(5):
            for j in range(5):
                if not j == i: #Remove diagonals
                    for l in range(3):
                        if not l == n: #Remove diagonals
                            if energies[n] == energies[l]:
                                print(f'Degenerate fast eigenvalues {energies[n]} and {energies[l]}!')
                            Fa[i] += (-2*hbar * vel[j] / (energies[n]-energies[l])**2 *
                                    np.imag(np.vdot(eigvec[n], np.dot(dHam[i], eigvec[l])) *
                                    np.vdot(eigvec[l], np.dot(dHam[j], eigvec[n]))))
    
        global Faavg
        Faavg = (Faavg + np.linalg.norm(Fa))/2

    if nosyn == "False" or nosyn == "Nomag":

    	#To get the derivatives of the scalar fields find coordinate values of all neighbouring sites
    
        ##meshgrid to generate neighbours
        neighgrid = np.mgrid[-1:2,-1:2,-1:2, -1:2, -1:2].astype('float')
        neighgrid[0:3,:,:,:,:,:] *= stepr #Fix step sizes
        neighgrid[3,:,:,:,:,:] *= steptheta 
        neighgrid[4,:,:,:,:,:] *= stepphi
        neighgrid = np.array(pos)[:,None,None,None, None, None] + neighgrid 
        #uses broadcasting to duplicate
    	#x,y,z into each point on the grid. the result has first dimension determining which 
    	#coordinate is given and the remaining specifying position related to the point
    
    	#Find neighbouring scalar field values and eigenstate energies
        scalar = np.zeros([3, 3, 3, 3, 3]) 
        energy = np.zeros([3, 3, 3, 3, 3]) 
        scalar[2,1,1,1,1], energy[2,1,1,1,1] = scalarcalc(neighgrid[:,2,1,1,1,1], n)
        scalar[0,1,1,1,1], energy[0,1,1,1,1] = scalarcalc(neighgrid[:,0,1,1,1,1], n)
        scalar[1,2,1,1,1], energy[1,2,1,1,1] = scalarcalc(neighgrid[:,1,2,1,1,1], n)
        scalar[1,0,1,1,1], energy[1,0,1,1,1] = scalarcalc(neighgrid[:,1,0,1,1,1], n)
        scalar[1,1,2,1,1], energy[1,1,2,1,1] = scalarcalc(neighgrid[:,1,1,2,1,1], n)
        scalar[1,1,0,1,1], energy[1,1,0,1,1] = scalarcalc(neighgrid[:,1,1,0,1,1], n)
        scalar[1,1,1,2,1], energy[1,1,1,2,1] = scalarcalc(neighgrid[:,1,1,1,2,1], n)
        scalar[1,1,1,0,1], energy[1,1,1,0,1] = scalarcalc(neighgrid[:,1,1,1,0,1], n)
        scalar[1,1,1,1,2], energy[1,1,1,1,2] = scalarcalc(neighgrid[:,1,1,1,1,2], n)
        scalar[1,1,1,1,0], energy[1,1,1,1,0] = scalarcalc(neighgrid[:,1,1,1,1,0], n)

        #Differentiate the scalar field
        dscalar[0] = (scalar[2,1,1,1,1] - scalar[0,1,1,1,1])/(2*stepr)
        dscalar[1] = (scalar[1,2,1,1,1] - scalar[1,0,1,1,1])/(2*stepr)
        dscalar[2] = (scalar[1,1,2,1,1] - scalar[1,1,0,1,1])/(2*stepr)
        dscalar[3] = (scalar[1,1,1,2,1] - scalar[1,1,1,0,1])/(2*steptheta)
        dscalar[4] = (scalar[1,1,1,1,2] - scalar[1,1,1,1,0])/(2*stepphi)

        global dscalaravg
        dscalaravg = (dscalaravg + np.linalg.norm(dscalar))/2

    elif nosyn == "True" or nosyn == "Noscalar":
    	#To get the derivatives of the energies find coordinate values of all neighbouring sites
    
        ##meshgrid to generate neighbours
        neighgrid = np.mgrid[-1:2,-1:2,-1:2, -1:2, -1:2].astype('float')
        neighgrid[0:3,:,:,:,:,:] *= stepr #Fix step sizes
        neighgrid[3,:,:,:,:,:] *= steptheta
        neighgrid[4,:,:,:,:,:] *= stepphi
        neighgrid = np.array(pos)[:, None, None, None, None, None] + neighgrid 
        #uses broadcasting to duplicate
    	#x,y,z into each point on the grid. the result has first dimension determining which 
    	#coordinate is given and the remaining specifying position related to the point
    
        #Find energies at neighbouring points
        energy = np.zeros([3, 3, 3, 3, 3])
        energy[2,1,1,1,1] = eigensolver(neighgrid[:,2,1,1,1,1])[0][n]
        energy[0,1,1,1,1] = eigensolver(neighgrid[:,0,1,1,1,1])[0][n]
        energy[1,2,1,1,1] = eigensolver(neighgrid[:,1,2,1,1,1])[0][n]
        energy[1,0,1,1,1] = eigensolver(neighgrid[:,1,0,1,1,1])[0][n]
        energy[1,1,2,1,1] = eigensolver(neighgrid[:,1,1,2,1,1])[0][n]
        energy[1,1,0,1,1] = eigensolver(neighgrid[:,1,1,0,1,1])[0][n]
        energy[1,1,1,2,1] = eigensolver(neighgrid[:,1,1,1,2,1])[0][n]
        energy[1,1,1,0,1] = eigensolver(neighgrid[:,1,1,1,0,1])[0][n]
        energy[1,1,1,1,2] = eigensolver(neighgrid[:,1,1,1,1,2])[0][n]
        energy[1,1,1,1,0] = eigensolver(neighgrid[:,1,1,1,1,0])[0][n]
        energy[1,1,1,1,1] = eigensolver(neighgrid[:,1,1,1,1,1])[0][n]

    else:
        print(f'Incorrect parameter "nosyn" = {nosyn}')

    global energyavg
    energyavg = (energyavg + energy[1,1,1,1,1])/2

    #Differentiate the energies
    denergy[0] = (energy[2,1,1,1,1] - energy[0,1,1,1,1])/(2*stepr)
    denergy[1] = (energy[1,2,1,1,1] - energy[1,0,1,1,1])/(2*stepr)
    denergy[2] = (energy[1,1,2,1,1] - energy[1,1,0,1,1])/(2*stepr)
    denergy[3] = (energy[1,1,1,2,1] - energy[1,1,1,0,1])/(2*steptheta)
    denergy[4] = (energy[1,1,1,1,2] - energy[1,1,1,1,0])/(2*stepphi)

    acc = Fa - dscalar - denergy #Summarize forces

    global denergyavg
    denergyavg = (denergyavg + np.linalg.norm(denergy))/2

    for i in range(5):
        acc[i] = acc[i]/mass[i] #Divide by mass to get acceleration

    if norot: #Freeze rotational axes if norot is turned on
        vel[3:5] = 0
        acc[3:5] = 0

    if acc[3] > 1e10:
        print(f'The rotation is out of control, acc = {acc}')

    return np.concatenate((vel, acc))

##Solves the ODE and returns the solution as per sscipy.integrate.solve_ivp.
##The external magnetic field is given as field. The
##dumbbell is placed initially at position pos and with velocity vel of the shape (x, y,
##z, theta_r, phi_r). Note that cartesian position here is in m.
##The spin subsystem is assumed to remain in the fast
##eigenstate labeled n. Runs until the time reaches tmax.
##The given initial conditions must be tuples.
def solvedyn(pos, vel, n, norot=False, nosyn='False'):
    posvel = pos + vel

    #Reset average force counters
    global Faavg
    global dscalaravg
    global denergyavg
    global energyavg
    Faavg = 0
    dscalaravg = 0
    denergyavg = 0
    energyavg = 0

    #Find times to require ODE evaluation
    t_eval = np.linspace(0, tmax, 100000)

    #Set error tolerances
    edgedistance.terminal = True
    sol = solve_ivp(acc, (0,tmax), posvel, events=edgedistance, args=(n, norot, nosyn), t_eval=t_eval)

    sol.Faavg = Faavg
    sol.denergyavg = denergyavg
    sol.dscalaravg = dscalaravg
    sol.energyavg = energyavg

    return sol

##Event to terminate integration, returns distance to the closest edge minus a small
##correction to avoid hitting the edge
def edgedistance(t, posvel, n, norot, nosyn):
    pos= posvel[0:3]
    mindist = np.amin(pos)
    maxdist = np.amax(pos)
    distancetoedge = min(mindist+lablength/1000, lablength - maxdist+lablength/1000)

    return distancetoedge
    

##Plotting function, takes a list of solutions sol from solve_ivp, a field field and displays an 
##interactive 3D swarm plot. Uses matplotlib.
def lineplot(sol, initvel, swarmnum, n, norot, nosyn, alternatestreams):

    #Print average acceleration components
    for stream in sol:
        print(f'Statistics for stream starting at {stream.y[0:3,0]}')
        print(f'Faavg = {stream.Faavg}')
        print(f'dscalaravg = {stream.dscalaravg}')
        print(f'denergyavg = {stream.denergyavg}')
        print(f'energyavg = {stream.energyavg}')

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.set_xlim((0, lablength*1000))
    ax.set_ylim((0, lablength*1000))
    ax.set_zlim((0, lablength*1000))
    ax.set_xlabel('x (mm)', fontsize=10, color='blue')
    ax.set_ylabel('y (mm)', fontsize=10, color='blue')
    ax.set_zlabel('z (mm)', fontsize=10, color='blue')
    
    #Create grids for the quiver:
#    xx, yy, zz = stepr*np.mgrid[0:nr, 0:nr, 0:nr]
#    xx = xx[0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]
#    yy = yy[0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]
#    zz = zz[0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]
#    Bx = field[0,:,:,:][0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]
#    By = field[1,:,:,:][0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]
#    Bz = field[2,:,:,:][0::int(nr/5), 0::int(nr/5), 0::int(nr/5)]


    for stream in sol:
    #Extract pos
        #Set stream colour
        try:
            color = stream.color
        except AttributeError:
            color = 'red'
        pos = stream.y[0:3,:]
        pos = pos*1000 #Plot in mm
        ax.plot3D(pos[0,:], pos[1,:], pos[2,:], color=color) #Plot the integrated path
    #Plot magnetic field for testing purposes
    #ax.quiver(xx, yy, zz, Bx, By, Bz, length=0.0001, normalize=True)

    plt.savefig(f'saves/graphs/field{field}nr{nr}lablength{lablength}tmax{tmax}J{J}Gamma{Gamma}mass{mass0}len{len0}n{n}vel{initvel}swarmnum{swarmnum}norot{norot}nosyn{nosyn}altstream{alternatestreams}.png',
                bbox_inches='tight')
    plt.show()
