from numba import jit
import numpy as np
import time
import vtk


# @jit(nopython=True, cache=True)
def getBoxRSA2D(seed, a_LowerBounds, a_UpperBounds, a_N, a_R0, max_time_sec, a_R1=None):
    '''Generates an ensemble or packing of circular disks in 2-dimensions within a rectangular box domain.

    Note:
    - To generate a monodisperse set of disks - that is, all disks of equal
    radius - set a_R0, but leave a_R1 as None.
    - To generale a polydisperse set of disks - that is, all disks with sizes
    generated from a uniform random distribution between [a,b] set a_R0 to a
    and a_R1 to b.

    '''

    np.random.seed(seed)

    countParticle   = 1
    posRSA          = np.zeros((a_N, 3), dtype=np.float32)
    countIterations = 0

    # Record start time
    start_time = time.time()

    while countParticle <= a_N:

        # Check if time limit has been exceeded
        if time.time() - start_time > max_time_sec:
            print(f'Time limit of {max_time_sec} sec. exceeded')
            return posRSA, False

        countIterations +=1

        xP  = np.random.uniform(a_LowerBounds[0],a_UpperBounds[0])
        yP  = np.random.uniform(a_LowerBounds[1],a_UpperBounds[1])
        rP  = np.random.uniform(a_R0, a_R1) if a_R1 is not None else a_R0

        if countParticle == 1:

            posRSA[0,0] = xP
            posRSA[0,1] = yP
            posRSA[0,2] = rP

            countParticle   += 1

        elif countParticle >= 1:

            countViolate = 0

            isInside = (xP - a_LowerBounds[0] > rP) and (a_UpperBounds[0] - xP > rP)\
                    and (yP - a_LowerBounds[1] > rP) and (a_UpperBounds[1] - yP > rP)

            for kCheck in np.arange(1,countParticle):

                isContact   = np.sqrt( (xP - posRSA[kCheck-1,0])**2 + (yP - posRSA[kCheck-1,1])**2 ) <= (rP + posRSA[kCheck-1,2])

                if isContact == True:
                    countViolate += 1

            if countViolate == 0 and isInside:

                posRSA[countParticle-1,0] = xP
                posRSA[countParticle-1,1] = yP
                posRSA[countParticle-1,2] = rP
                countParticle           += 1

    return posRSA, True

# @jit(nopython=True, cache=True)
def getBoxRSA3D(seed, a_LowerBounds, a_UpperBounds, a_N, a_R0, max_time_sec, a_R1=None):
    """
    Generates an ensemble or packing of spheres in 3-dimensions within a rectangular box domain.

    Parameters:
      seed          : Random seed.
      a_LowerBounds : numpy array of lower bounds [x_lower, y_lower, z_lower].
      a_UpperBounds : numpy array of upper bounds [x_upper, y_upper, z_upper].
      a_N           : The desired number of spheres.
      a_R0          : A constant radius (if a_R1 is None) or the lower bound for the radius.
      max_time_sec  : Maximum time allowed for the packing generation.
      a_R1          : Upper bound for the radius (optional for a polydisperse packing).

    Returns:
      posRSA : a_Nx4 numpy array whose rows are [x, y, z, r] for each accepted sphere.
      success: Boolean that is True if packing terminated normally, False if the time limit was exceeded.
    """

    np.random.seed(seed)
    countParticle = 0
    posRSA = np.zeros((a_N, 4), dtype=np.float32)
    start_time = time.time()

    while countParticle < a_N:
        # Check for time limit
        if time.time() - start_time > max_time_sec:
            print(f"Time limit of {max_time_sec} sec. exceeded")
            return posRSA, False

        # Generate a candidate sphere
        xP = np.random.uniform(a_LowerBounds[0], a_UpperBounds[0])
        yP = np.random.uniform(a_LowerBounds[1], a_UpperBounds[1])
        zP = np.random.uniform(a_LowerBounds[2], a_UpperBounds[2])
        rP = np.random.uniform(a_R0, a_R1) if a_R1 is not None else a_R0

        # Boundary check for the candidate using correct z-coordinate
        isInside = ((xP - a_LowerBounds[0] > rP) and (a_UpperBounds[0] - xP > rP) and
                    (yP - a_LowerBounds[1] > rP) and (a_UpperBounds[1] - yP > rP) and
                    (zP - a_LowerBounds[2] > rP) and (a_UpperBounds[2] - zP > rP))
        if not isInside:
            continue

        # Check for overlap with previously accepted spheres
        collision_found = False
        for i in range(countParticle):
            dx = xP - posRSA[i, 0]
            dy = yP - posRSA[i, 1]
            dz = zP - posRSA[i, 2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            if distance <= (rP + posRSA[i, 3]):
                collision_found = True
                break

        if collision_found:
            continue

        # Accept the candidate sphere.
        posRSA[countParticle, :] = [xP, yP, zP, rP]
        countParticle += 1

    return posRSA, True


# @jit(nopython=True, cache=True)
def getPorosity2D(a_LowerBounds, a_UpperBounds, a_PosRSA):
    '''Computes porosity of the 2-dimensional packing of circular disks

    '''

    domainArea  = (a_UpperBounds[0] - a_LowerBounds[0]) * \
                (a_UpperBounds[1] - a_LowerBounds[1])

    area        = 0.0

    for i in range(a_PosRSA.shape[0]):
        area += np.pi*a_PosRSA[i,2]**2

    porosity = area/domainArea

    return porosity

def getPorosity3D(a_LowerBounds, a_UpperBounds, a_PosRSA):
    '''Computes porosity of the 3-dimensional packing of spheres

    '''

    domainVolume    = (a_UpperBounds[0] - a_LowerBounds[0]) * \
                    (a_UpperBounds[1] - a_LowerBounds[1]) * \
                    (a_UpperBounds[2] - a_LowerBounds[2])

    volume          = 0.0

    for i in range(a_PosRSA.shape[0]):
        volume += (4.0/3.0)*np.pi*a_PosRSA[i,3]**3

    porosity    = volume/domainVolume

    return porosity
