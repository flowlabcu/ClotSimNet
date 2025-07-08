from numba import jit
import numpy as np
import time
import vtk

# @jit(nopython=True, cache=True)
def getBoxRSA2D(seed, a_LowerBounds, a_UpperBounds, a_N, a_R0, max_time_sec, a_R1=None):
    '''Generates an ensemble or packing of circular disks in 2-dimensions
    within a rectangular box domain.

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

@jit(nopython=True, cache=True)
def getBoxRSA2D_phi_N(seed, phi, a_N, a_LowerBounds, a_UpperBounds, max_attempts=1000, a_R1=None):
    '''Generates an ensemble or packing of circular disks in 2-dimensions
    within a rectangular box domain.

    Note:
    - To generate a monodisperse set of disks - that is, all disks of equal
    radius - set a_R0, but leave a_R1 as None.
    - To generale a polydisperse set of disks - that is, all disks with sizes
    generated from a uniform random distribution between [a,b] set a_R0 to a
    and a_R1 to b.

    '''

    np.random.seed(seed)

    # Calculate area of box to calculate a_R0 for use later

    delta_x = a_UpperBounds[0] - a_LowerBounds[0]
    delta_y = a_UpperBounds[1] - a_LowerBounds[1]

    area = delta_x * delta_y

    numerator = area - area*phi
    a_R0 = np.sqrt(numerator / (a_N*np.pi))

    countParticle   = 1
    posRSA          = np.zeros((a_N, 3), dtype=np.float32)
    countIterations = 0
    attempts = 0

    max_density = 0.547
    max_porosity = 1 - max_density

    if phi >= max_porosity:
        print(f'Porosity {phi} greater than {max_porosity}, exiting')
        return posRSA

    while countParticle <= a_N:

        attempts += 1

        # Check if maximum attempts have been reached
        if attempts > max_attempts:
            print(f'Could not find particle configuration after {attempts} attempts')
            return posRSA

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

    return posRSA

@jit(nopython=True, cache=True)
def getBoxRSA2D_phi_R(seed, phi, a_R0, a_LowerBounds, a_UpperBounds, max_attempts=1000, a_R1=None):
    '''Generates an ensemble or packing of circular disks in 2-dimensions
    within a rectangular box domain.

    Note:
    - To generate a monodisperse set of disks - that is, all disks of equal
    radius - set a_R0, but leave a_R1 as None.
    - To generale a polydisperse set of disks - that is, all disks with sizes
    generated from a uniform random distribution between [a,b] set a_R0 to a
    and a_R1 to b.

    '''

    np.random.seed(seed)

    # Calculate area of box to calculate a_R0 for use later

    delta_x = a_UpperBounds[0] - a_LowerBounds[0]
    delta_y = a_UpperBounds[1] - a_LowerBounds[1]

    area = delta_x * delta_y

    a_N = int(area*(1-phi) / (np.pi*a_R0**2))

    countParticle   = 1
    posRSA          = np.zeros((a_N, 3), dtype=np.float32)
    countIterations = 0
    attempts = 0

    max_density = 0.547
    max_porosity = 1 - max_density

    if phi >= max_porosity:
        print(f'Porosity {phi} greater than {max_porosity}, exiting')
        return posRSA

    while countParticle <= a_N:
        print(countParticle)
        attempts += 1

        # Check if maximum attempts have been reached
        if attempts > max_attempts:
            print(f'Could not find particle configuration after {attempts} attempts')
            return posRSA

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

    return posRSA

@jit(nopython=True, cache=True)
def getBoxRSA3D(a_LowerBounds, a_UpperBounds, a_N, a_R0, a_R1=None):

    countParticle   = 1
    posRSA          = np.zeros((a_N, 4), dtype=np.float32)
    countIterations = 0

    while countParticle <= a_N:

        xP  = np.random.uniform(a_LowerBounds[0],a_UpperBounds[0])
        yP  = np.random.uniform(a_LowerBounds[1],a_UpperBounds[1])
        zP  = np.random.uniform(a_LowerBounds[2],a_UpperBounds[2])
        rP  = np.random.uniform(a_R0, a_R1) if a_R1 is not None else a_R0

        if countParticle == 1:

            posRSA[0,0]     = xP
            posRSA[0,1]     = yP
            posRSA[0,2]     = zP
            posRSA[0,3]     = rP
            countParticle   += 1

        elif countParticle >= 1:

            countViolate = 0

            isInside = (xP - a_LowerBounds[0] > rP) and (a_UpperBounds[0] - xP > rP)\
                    and (yP - a_LowerBounds[1] > rP) and (a_UpperBounds[1] - yP > rP)\
                    and (zP - a_LowerBounds[2] > rP) and (a_UpperBounds[2] - yP > rP)

            for kCheck in np.arange(1,countParticle):

                isContact   = np.sqrt( (xP - posRSA[kCheck-1,0])**2 + (yP - posRSA[kCheck-1,1])**2 + (zP - posRSA[kCheck-1,2])**2 ) <= (rP + posRSA[kCheck-1,2])

                if isContact == True:
                    countViolate += 1

            if countViolate == 0:

                posRSA[countParticle-1,0]   = xP
                posRSA[countParticle-1,1]   = yP
                posRSA[countParticle-1,2]   = zP
                posRSA[countParticle-1,3]   = rP
                countParticle               += 1

    return posRSA

@jit(nopython=True, cache=True)
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
