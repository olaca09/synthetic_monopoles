import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as mu
from scipy.constants import pi as pi

def simplewire(nr, lablength, I, overwrite=False):
    ##Returns a placeholder field corresponding to a current through a wire along the
    ##x-axis. Saves the field and won't generate a preexisting field unless
    ##'overwrite=True' is called. Takes nr as the number of points along each axis.

    #Initiate variables:
    stepr = lablength/(nr-1) #Length of each lattice site
    generate = False #Whether the field needs to be generated
    wire = np.array([0,0]) #Position in x and y of current

    try: #Try to load pregenerated field
        field = np.load(f'saves/simplewire{I}field{nr},{lablength}.npy')
    except FileNotFoundError:
        generate = True

    #Returned saved field unless not found or overwrite turned on
    if not (generate or overwrite):
        #Return pregenerated field
        print("Loading saved magnetic field")
        return field
    else:
        #Generate field
        print("Generating magnetic field")

        xx, yy, zz = np.mgrid[0:nr, 0:nr, 0:nr]
        distance = [(xx - xx)*stepr, (yy - wire[0])*stepr, (zz - wire[1])*stepr]
        #Use Biot-Savart formula to calculate the magnetic field
        Bx, By, Bz = np.zeros([nr, nr, nr]), np.zeros([nr, nr, nr]), np.zeros([nr, nr, nr])
        Bx, By, Bz = mu/(4*pi)*2*I*np.cross([1, 0, 0], distance,
                axisb=0, axisc=0)/griddot(distance, distance)
        Br, Bth, Bph = cart_to_sph(np.array([Bx,By,Bz])) #Express as spherical coordinates
        #field = xr.DataArray([Bx, By, Bz, Br, Bth, Bph], dims=("direc", "x", "y", "z"), coords={"direc":
        field = np.array((Bx, By, Bz, Br, Bth, Bph))

        np.save(f'saves/simplewire{I}field{nr},{lablength}.npy', field)
        #field.to_netcdf("saves/simplewirefield.nc")
        return field

def oppositecoils(nr, lablength, I, overwrite=False):
    ##Returns a field corresponding to two currents of opposing directions through square coils 
    ##placed orthogonally to the z-axis centred 1/4th from the edges. Saves the field and won't generate a preexisting field unless
    ##'overwrite=True' is called. Takes nr as the number of points along each axis.

    #Initiate variables:
    stepr = lablength/(nr-1) #Length of each lattice site
    generate = False #Whether the field needs to be generated

    #Positions of all flowing currents below:
    #qindex = int(np.floor(nr/4))
    qindex = int(nr/3)
    wire1 = np.array([-qindex,-qindex]) #Current in positive x close to z=0 and y=0 (pos in y,z)
    wire2 = np.array([-qindex,nr+qindex]) #Current in positive y close to z=0 and far from x=0 (pos in z,x)
    wire3 = np.array([nr+qindex,-qindex]) #Current in negative x close to z=0 and far from y=0 (pos in y,z)
    wire4 = np.array([-qindex,-qindex]) #Current in negative y close to z=0 and close to x=0 (pos in z,x)
    wire5 = np.array([-qindex,nr+qindex]) #Current in negative x far from z=0 and close to y=0 (pos in y,z)
    wire6 = np.array([nr+qindex,nr+qindex]) #Current in negative y far from z=0 and x=0 (pos in z,x)
    wire7 = np.array([nr+qindex,nr+qindex]) #Current in positive x far from z=0 and y=0 (pos in y,z)
    wire8 = np.array([nr+qindex,-qindex]) #Current in positive y far from z=0 and close to x=0 (pos in z,x)

    try: #Try to load pregenerated field
        field = np.load(f'saves/fields/oppositecoils{I}field{nr},{lablength}.npy')
    except FileNotFoundError:
        generate = True

    #Returned saved field unless not found or overwrite turned on
    if not (generate or overwrite):
        #Return pregenerated field
        print("Loading saved magnetic field")
        return field
    else:
        #Generate field
        print("Generating magnetic field")

        xx, yy, zz = np.mgrid[0:nr, 0:nr, 0:nr]
        #Integrate the field per Biot-Savart along all currents
        Bx, By, Bz = np.zeros([nr, nr, nr]), np.zeros([nr, nr, nr]), np.zeros([nr, nr, nr])

        #Currents along x:
        for px in range(-qindex, nr+qindex): #Maybe adjust with a +1
            dx = [stepr, 0, 0]
            #wire1
            distance = [(xx-px)*stepr, (yy-wire1[0])*stepr, (zz-wire1[1])*stepr]
            dBx1, dBy1, dBz1 = (mu*I/(4*pi) * np.cross(dx, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx1
            By += dBy1
            Bz += dBz1
        
            #wire3
            distance = [(xx-px)*stepr, (yy-wire3[0])*stepr, (zz-wire3[1])*stepr]
            dBx3, dBy3, dBz3 = -(mu*I/(4*pi) * np.cross(dx, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx3
            By += dBy3
            Bz += dBz3

            #wire5
            distance = [(xx-px)*stepr, (yy-wire5[0])*stepr, (zz-wire5[1])*stepr]
            dBx5, dBy5, dBz5 = -(mu*I/(4*pi) * np.cross(dx, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx5
            By += dBy5
            Bz += dBz5

            #wire7
            distance = [(xx-px)*stepr, (yy-wire7[0])*stepr, (zz-wire7[1])*stepr]
            dBx7, dBy7, dBz7 = (mu*I/(4*pi) * np.cross(dx, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx7
            By += dBy7
            Bz += dBz7

        #Currents along y:
        for py in range(-qindex, nr+qindex): #Maybe adjust with a +1
            dy = [0, stepr, 0]
            #wire2
            distance = [(xx-wire2[1])*stepr, (yy-py)*stepr, (zz-wire2[0])*stepr]
            dBx2, dBy2, dBz2 = (mu*I/(4*pi) * np.cross(dy, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx2
            By += dBy2
            Bz += dBz2
        
            #wire4
            distance = [(xx-wire4[1])*stepr, (yy-py)*stepr, (zz-wire4[0])*stepr]
            dBx4, dBy4, dBz4 = -(mu*I/(4*pi) * np.cross(dy, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx4
            By += dBy4
            Bz += dBz4
        
            #wire6
            distance = [(xx-wire6[1])*stepr, (yy-py)*stepr, (zz-wire6[0])*stepr]
            dBx6, dBy6, dBz6 = -(mu*I/(4*pi) * np.cross(dy, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx6
            By += dBy6
            Bz += dBz6
        
            #wire8
            distance = [(xx-wire8[1])*stepr, (yy-py)*stepr, (zz-wire8[0])*stepr]
            dBx8, dBy8, dBz8 = (mu*I/(4*pi) * np.cross(dy, distance, axisb=0,
                                axisc=0)/(griddot(distance, distance)**(3/2)))
            Bx += dBx8
            By += dBy8
            Bz += dBz8
        
        Br, Bth, Bph = cart_to_sph(np.array([Bx,By,Bz])) #Express as spherical coordinates
        field = np.array((Bx, By, Bz, Br, Bth, Bph))

        np.save(f'saves/fields/oppositecoils{I}field{nr},{lablength}.npy', field)
        return field

def griddot(a, b):
    ##Returns the dot product for each point in the supplied grids a, b. Contracts the
    ##first dimension.
    result = np.zeros(a[0].shape) + 1e-20 #Super ugly bodge to fix NaN
    for i in range(len(a)):
        result += a[i]*b[i]

    result = np.where(result == 0, 1e-20, result)
    return result

def cart_to_sph(cart):
    ##Returns spherical coordinates of the form(r, polar, azimuthal) for the given cartesian 
    ##coordinates of the form (x,y,z) takes an array with coords as the first dimension.
    sph = np.zeros(cart.shape) #Initialize array
    xsqysq = cart[0]**2 + cart[1]**2 #Value of x^2 + y^2
    sph[0] = np.sqrt(xsqysq + cart[2]**2) #Radius r
    sph[1] = np.arctan2(np.sqrt(xsqysq), cart[2]) #Polar angle theta
    sph[2] = np.arctan2(cart[1],cart[0]) #Azimuthal angle phi
    return sph

        
def sliceplot(field):
    ##Plots the given field in the y-z-plane at x=0

    #Load the slice of B-values at x=0, indexed after y, z.
    By = field.sel(direc="By")[0,:,:]
    Bz = field.sel(direc="Bz")[0,:,:]
    #By = By.drop(dim="x")
    #Bz = Bz.drop(dim="x")
    y = np.arange(field.sizes["y"])
    z = np.arange(field.sizes["z"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #Plot neat field lines
    color = np.array(2 * np.log(np.hypot(By, Bz) + 1e-10))
    #Note that streamplot unfortunately indexes the velocities as [vertical, horizontal],
    #such that this yields z on the horizontal and y on the vertical.
    ax.streamplot(z, y, Bz, By, linewidth=1,
            cmap=plt.cm.inferno, color=color, density=2, arrowstyle='->', arrowsize=1.5)

    #Touch up appearance and show plot
    plt.xlabel('Position along z-axis')
    plt.ylabel('Position along y-axis')
    plt.show()


