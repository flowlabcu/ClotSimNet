import numpy as np
import sys

def writeGeoFromXYZConstantSizing(a_XYZFile, a_GeoFile, a_Sizing):
    '''Function to compute a Gmsh compatible `*.geo` file to include all the points from a specified file with formatted point coordinates

    Args:
        a_XYZFile (string): file with formatted point coordinates (space separated)
        a_GeoFile (string): output geo file where all points are to be added
        a_Sizing (float): the point-wise sizing parameter (constant for all points)

    Returns:
        none

    '''

    geoFile = open(a_GeoFile,"w+")

    with open(a_XYZFile) as xyzFile:

        next(xyzFile)

        for line in xyzFile:

            lineDict = line.split()

            x   = lineDict[0]
            y   = lineDict[1]
            z   = lineDict[2]

            geoLine = "Point(newp) = {"+x+","+y+","+z+","+str(a_Sizing)+"};\n"
            geoFile.write(geoLine)

    geoFile.close()

def writeGeoFromVTKConstantSizing(a_VTPFile, a_GeoFile, a_Sizing):

    try:
        import vtk
    except ImportError:
        sys.exit("Could Not Import Module VTK")

    geoFile = open(a_GeoFile, "w+")
    geoLine = "hp = "+str(a_Sizing)+";"+"\n"
    geoFile.write(geoLine)

    if a_VTPFile.endswith('vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif a_VTPFile.endswith('vtk'):
        reader = vtk.vtkPolyDataReader()

    reader.SetFileName(a_VTPFile)
    reader.Update()

    numPoints = reader.GetOutput().GetNumberOfPoints()

    for p in range(numPoints):

        xyz     = reader.GetOutput().GetPoint(p)
        geoLine = "Point(newp) = {"+str(xyz[0])+","+str(xyz[1])+","+str(xyz[2])+",hp};\n"
        geoFile.write(geoLine)

    geoFile.close()

#------------------------------------------------------------------------------
# Function to compose a Gmsh compatible '*.geo' file to include
# all the points from a specified file with formatted point coordinates
#
# Parameters:
# a_XYZFile         file with formatted point coordinates (space separated)
# a_GeoFile         output geo file where all points are to be loaded
# a_ParametricSize  set this to True if you want to leave size as a parameter h
#------------------------------------------------------------------------------
def writeGeoFromXYZVarySizing(a_XYZFile, a_GeoFile, a_ParametricSize=True):

    geoFile = open(a_GeoFile,"w+")

    with open(a_XYZFile) as xyzFile:

        next(xyzFile)

        for line in xyzFile:

            lineDict = line.split()

            x   = lineDict[0]
            y   = lineDict[1]
            z   = lineDict[2]

            if a_ParametricSize == True:
                geoLine = "Point(newp) = {"+x+","+y+","+z+",h};\n"
            else:
                h       = lineDict[3]
                geoLine = "Point(newp) = {"+x+","+y+","+z+","+h+"};\n"

            geoFile.write(geoLine)

    geoFile.close()

#----------------------------------------------------------------------------------
# Function to compose a Gmsh compatible '*.geo' file to include all points
# from a specified file with particle packing data, with a constant mesh
# sizing specified for all points/regions in the mesh
#
# Parameters:
# a_Box         numpy array containing bottom-left and top-right box coordinates
# a_XYZFile     file with point coordinates and radius (space/tab separated)
# a_GeoFile     output geo file where all meshing information is to be loaded
# a_Sizing      fixed mesh sizing associated with all points in the domain
#----------------------------------------------------------------------------------
def xyBoxPackingGeoWriterFixed(a_Box, a_XYZFile, a_GeoFile, a_Sizing):

    geoFile = open(a_GeoFile, "w+")

    with open(a_XYZFile) as xyzFile:

        p       = 1
        c       = 1
        holeID  = []
        holeStr = ","

        for line in xyzFile:

            lineDict = line.split()

            xP  = float(lineDict[0])
            yP  = float(lineDict[1])
            zP  = float(lineDict[2])
            rP  = float(lineDict[3])

            geoLine = "p1 = "+str(p)+"; Point(p1) = {"+str(xP-rP)+","+str(yP)+","+str(zP)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            geoLine = "p2 = "+str(p)+"; Point(p2) = {"+str(xP)+","+str(yP)+","+str(zP)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            geoLine = "p3 = "+str(p)+"; Point(p3) = {"+str(xP+rP)+","+str(yP)+","+str(zP)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            geoLine = "c1 = "+str(c)+"; Circle(c1) = {p1, p2, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = "c2 = "+str(c)+"; Circle(c2) = {p3, p2, p1};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = "c3 = "+str(c)+"; Line Loop(c3) = {"+str(c-1)+","+str(c-2)+"};"+"\n"
            holeID.append(c)
            holeStr = holeStr + str(c) + ","
            c = c + 1
            geoFile.write(geoLine)

            geoLine = "c4 = "+str(c)+"; Physical Line(c4) = {"+str(c-2)+","+str(c-3)+"};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = 'Printf("Added Physical Boundary %g",'+str(c-1)+');'+"\n"
            geoFile.write(geoLine)

    geoLine = "b1 = newp; Point(b1) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+", 0.0, "+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "b2 = newp; Point(b2) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+", 0.0, "+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "b3 = newp; Point(b3) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+", 0.0, "+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "b4 = newp; Point(b4) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+", 0.0, "+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l1 = newreg; Line(l1) = {b4, b3};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l2 = newreg; Line(l2) = {b3, b2};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l3 = newreg; Line(l3) = {b2, b1};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l4 = newreg; Line(l4) = {b1, b4};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l5 = 1000000; Physical Line(l5) = {l1};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l6 = 2000000; Physical Line(l6) = {l2};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l7 = 3000000; Physical Line(l7) = {l3};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l8 = 4000000; Physical Line(l8) = {l4};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l9 = newreg; Line Loop(l9) = {l4, l3, l2, l1};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l10 = newreg; Plane Surface(l10) = {l9"+holeStr[:-1]+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l11 = newreg; Physical Surface(l11) = {l10};"+"\n"
    geoFile.write(geoLine)

    geoFile.close()

#----------------------------------------------------------------------------------
# Function to compose a Gmsh compatible '*.geo' file to include all points
# from a specified file with particle packing data, with a constant mesh
# sizing specified for all points/regions in the mesh
#
# Parameters:
# a_Box         numpy array containing bottom-left and top-right box coordinates
# a_XYZFile     file with point coordinates and radius (space/tab separated)
# a_GeoFile     output geo file where all meshing information is to be loaded
# a_Sizing      fixed mesh sizing associated with all points in the domain
# a_ReadSize
# a_SizeArray
# a_SizeBox
#----------------------------------------------------------------------------------
def xyBoxPackingGeoWriterVarying(a_Box, a_XYZFile, a_GeoFile, a_ReadSize=True, a_SizeArray=None, a_SizeBox=None):

    geoFile = open(a_GeoFile, "w+")

    if a_ReadSize == True and a_SizeArray is not None:
        sys.exit()
    elif a_ReadSize == False and a_SizeArray is None:
        sys.exit()

    hB = a_SizeBox if a_SizeBox is not None else (a_Box[1,0]-a_Box[0,0])/20

    with open(a_XYZFile) as xyzFile:

        p       = 1
        c       = 1
        holeID  = []
        holeStr = ","

        for line in xyzFile:

            lineDict = line.split()

            xP  = float(lineDict[0])
            yP  = float(lineDict[1])
            zP  = float(lineDict[2])
            rP  = float(lineDict[3])

            if (a_ReadSize == True) and (a_SizeArray is None):
                hP  = float(lineDict[4])
            else:
                hP  = a_SizeArray[p-1]

            geoLine = "p1 = "+str(p)+"; Point(p1) = {"+str(xP-rP)+","+str(yP)+","+str(zP)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p2 = "+str(p)+"; Point(p2) = {"+str(xP)+","+str(yP)+","+str(zP)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p3 = "+str(p)+"; Point(p3) = {"+str(xP+rP)+","+str(yP)+","+str(zP)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            geoLine = "c1 = "+str(c)+"; Circle(c1) = {p1, p2, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c2 = "+str(c)+"; Circle(c2) = {p3, p2, p1};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = "c3 = "+str(c)+"; Line Loop(c3) = {"+str(c-1)+","+str(c-2)+"};"+"\n"
            holeID.append(c)
            holeStr = holeStr + str(c) + ","
            c = c + 1
            geoFile.write(geoLine)

            geoLine = "c4 = "+str(c)+"; Physical Line(c4) = {"+str(c-2)+","+str(c-3)+"};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            #geoLine = 'Printf("Added Physical Boundary %g",'+str(c-1)+');'+"\n"
            #geoFile.write(geoLine)

    geoLine = "b1 = newp; Point(b1) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+", 0.0, "+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b2 = newp; Point(b2) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+", 0.0, "+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b3 = newp; Point(b3) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+", 0.0, "+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b4 = newp; Point(b4) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+", 0.0, "+str(hB)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l1 = newreg; Line(l1) = {b4, b3};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l2 = newreg; Line(l2) = {b3, b2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l3 = newreg; Line(l3) = {b2, b1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l4 = newreg; Line(l4) = {b1, b4};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l5 = newreg; Physical Line(l5) = {l1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l6 = newreg; Physical Line(l6) = {l2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l7 = newreg; Physical Line(l7) = {l3};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l8 = newreg; Physical Line(l8) = {l4};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l9 = newreg; Line Loop(l9) = {l4, l3, l2, l1};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l10 = newreg; Plane Surface(l10) = {l9"+holeStr[:-1]+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l11 = newreg; Physical Surface(l11) = {l10};"+"\n"
    geoFile.write(geoLine)

    geoFile.close()


#
#
#
def xyzBoxPackingGeoWriterFixed(a_Box, a_XYZFile, a_GeoFile, a_Sizing):

    geoFile = open(a_GeoFile, "w+")

    with open(a_XYZFile) as xyzFile:

        p       = 1
        c       = 1
        holeID  = []
        holeStr = ","

        for line in xyzFile:

            lineDict = line.split()

            xC  = float(lineDict[0])
            yC  = float(lineDict[1])
            zC  = float(lineDict[2])
            rP  = float(lineDict[3])

            #
            # define the set of 6 north-south-east-west points and the center coordinate with associated sizing
            #
            geoLine = "p1 = "+str(p)+"; Point(p1) = {"+str(xC)+","+str(yC)+","+str(zC)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p2 = "+str(p)+"; Point(p2) = {"+str(xC+rP)+","+str(yC)+","+str(zC)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p3 = "+str(p)+"; Point(p3) = {"+str(xC-rP)+","+str(yC)+","+str(zC)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p4 = "+str(p)+"; Point(p4) = {"+str(xC)+","+str(yC+rP)+","+str(zC)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p5 = "+str(p)+"; Point(p5) = {"+str(xC)+","+str(yC-rP)+","+str(zC)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p6 = "+str(p)+"; Point(p6) = {"+str(xC)+","+str(yC)+","+str(zC+rP)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p7 = "+str(p)+"; Point(p7) = {"+str(xC)+","+str(yC)+","+str(zC-rP)+","+str(a_Sizing)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            #
            # define circle arcs defining the 8 longitude-latitude patches
            #
            geoLine = "c1 = "+str(c)+"; Circle(c1) = {p6, p1, p4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c2 = "+str(c)+"; Circle(c2) = {p4, p1, p7};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c3 = "+str(c)+"; Circle(c3) = {p7, p1, p5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c4 = "+str(c)+"; Circle(c4) = {p5, p1, p6};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c5 = "+str(c)+"; Circle(c5) = {p6, p1, p2};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c6 = "+str(c)+"; Circle(c6) = {p2, p1, p7};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c7 = "+str(c)+"; Circle(c7) = {p7, p1, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c8 = "+str(c)+"; Circle(c8) = {p3, p1, p6};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c9 = "+str(c)+"; Circle(c9) = {p4, p1, p2};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c10 = "+str(c)+"; Circle(c10) = {p2, p1, p5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c11 = "+str(c)+"; Circle(c11) = {p5, p1, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c12 = "+str(c)+"; Circle(c12) = {p3, p1, p4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            #
            # define the patches as ruled surfaces based on collection of three arcs each
            #
            geoLine = "c13 = "+str(c)+"; Line Loop(c13) = {c2, c7, c12};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c14 = "+str(c)+"; Ruled Surface(c14) = {c13};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c15 = "+str(c)+"; Line Loop(c15) = {c12, -c1, -c8};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c16 = "+str(c)+"; Ruled Surface(c16) = {c15};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c17 = "+str(c)+"; Line Loop(c17) = {c1, c9, -c5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c18 = "+str(c)+"; Ruled Surface(c18) = {c17};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c19 = "+str(c)+"; Line Loop(c19) = {c2, -c6, -c9};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c20 = "+str(c)+"; Ruled Surface(c20) = {c19};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c21 = "+str(c)+"; Line Loop(c21) = {c6, c3, -c10};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c22 = "+str(c)+"; Ruled Surface(c22) = {c21};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c23 = "+str(c)+"; Line Loop(c23) = {c7, -c11, -c3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c24 = "+str(c)+"; Ruled Surface(c24) = {c23};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c25 = "+str(c)+"; Line Loop(c25) = {c11, c8, -c4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c26 = "+str(c)+"; Ruled Surface(c26) = {c25};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c27 = "+str(c)+"; Line Loop(c27) = {c4, c5, c10};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c28 = "+str(c)+"; Ruled Surface(c28) = {c27};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            #
            # define the overall sphere surface
            #
            geoLine = "c29 = "+str(c)+"; Surface Loop(c29) = {c14, c20, c22, c24, c26, c16, c18, c28};"+"\n"
            holeID.append(c)
            holeStr = holeStr + str(c) + ","
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = "c30 = "+str(c)+"; Physical Surface(c30) = {c14, c20, c22, c24, c26, c16, c18, c28};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

    geoLine = "b1 = newp; Point(b1) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+","+str(a_Box[0,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b2 = newp; Point(b2) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+","+str(a_Box[0,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b3 = newp; Point(b3) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+","+str(a_Box[0,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b4 = newp; Point(b4) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+","+str(a_Box[0,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b5 = newp; Point(b5) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+","+str(a_Box[1,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b6 = newp; Point(b6) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+","+str(a_Box[1,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b7 = newp; Point(b7) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+","+str(a_Box[1,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b8 = newp; Point(b8) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+","+str(a_Box[1,2])+","+str(a_Sizing)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l1 = newreg; Line(l1) = {b2, b6};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l2 = newreg; Line(l2) = {b6, b7};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l3 = newreg; Line(l3) = {b7, b3};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l4 = newreg; Line(l4) = {b3, b2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l5 = newreg; Line(l5) = {b2, b1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l6 = newreg; Line(l6) = {b1, b5};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l7 = newreg; Line(l7) = {b5, b6};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l8 = newreg; Line(l8) = {b5, b8};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l9 = newreg; Line(l9) = {b8, b4};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l10 = newreg; Line(l10) = {b4, b1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l11 = newreg; Line(l11) = {b3, b4};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l12 = newreg; Line(l12) = {b7, b8};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l13 = newreg; Line Loop(l13) = {l2, l12, -l8, l7};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l14 = newreg; Ruled Surface(l14) = {l13};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l15 = 1000000; Physical Surface(l15) = {l14};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l16 = newreg; Line Loop(l16) = {l3, l11, -l9, -l12};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l17 = newreg; Ruled Surface(l17) = {l16};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l18 = 2000000; Physical Surface(l18) = {l17};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l19 = newreg; Line Loop(l19) = {l3, l4, l1, l2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l20 = newreg; Ruled Surface(l20) = {l19};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l21 = 3000000; Physical Surface(l21) = {l20};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l22 = newreg; Line Loop(l22) = {l5, l6, l7, -l1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l23 = newreg; Ruled Surface(l23) = {l22};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l24 = 4000000; Physical Surface(l24) = {l23};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l25 = newreg; Line Loop(l25) = {l10, l6, l8, l9};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l26 = newreg; Ruled Surface(l26) = {l25};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l27 = 5000000; Physical Surface(l27) = {l26};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l28 = newreg; Line Loop(l28) = {l4, l5, -l10, -l11};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l29 = newreg; Ruled Surface(l29) = {l28};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l30 = 6000000; Physical Surface(l30) = {l29};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l31 = newreg; Surface Loop(l31) = {l20, l17, l29, l23, l26, l14};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l32 = newreg; Volume(l32) = {l31"+holeStr[:-1]+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l33 = newreg; Physical Volume(l33) = {l32};"+"\n"
    geoFile.write(geoLine)
    
    print('GMSH UTILITIES BULLSHIT')
    print(a_Box[0,0])
    print(a_Box[0,1])
    print(a_Box[0,2])


#
#
#
def xyzBoxPackingGeoWriterVarying(a_Box, a_XYZFile, a_GeoFile, a_ReadSize=True, a_SizeArray=None, a_SizeBox=None):
    


    geoFile = open(a_GeoFile, "w+")

    if a_ReadSize == True and a_SizeArray is not None:
        sys.exit()
    elif a_ReadSize == False and a_SizeArray is None:
        sys.exit()

    hB = a_SizeBox if a_SizeBox is not None else (a_Box[1,0]-a_Box[0,0])/20

    with open(a_XYZFile) as xyzFile:

        p       = 1
        c       = 1
        holeID  = []
        holeStr = ","

        for line in xyzFile:

            lineDict = line.split()

            xC  = float(lineDict[0])
            yC  = float(lineDict[1])
            zC  = float(lineDict[2])
            rP  = float(lineDict[3])

            if (a_ReadSize == True) and (a_SizeArray is None):
                hP  = float(lineDict[4])
            else:
                hP  = a_SizeArray[p-1]

            #
            # define the set of 6 north-south-east-west points and the center coordinate with associated sizing
            #
            geoLine = "p1 = "+str(p)+"; Point(p1) = {"+str(xC)+","+str(yC)+","+str(zC)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p2 = "+str(p)+"; Point(p2) = {"+str(xC+rP)+","+str(yC)+","+str(zC)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p3 = "+str(p)+"; Point(p3) = {"+str(xC-rP)+","+str(yC)+","+str(zC)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p4 = "+str(p)+"; Point(p4) = {"+str(xC)+","+str(yC+rP)+","+str(zC)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p5 = "+str(p)+"; Point(p5) = {"+str(xC)+","+str(yC-rP)+","+str(zC)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p6 = "+str(p)+"; Point(p6) = {"+str(xC)+","+str(yC)+","+str(zC+rP)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)
            geoLine = "p7 = "+str(p)+"; Point(p7) = {"+str(xC)+","+str(yC)+","+str(zC-rP)+","+str(hP)+"};"+"\n"
            p       = p + 1
            geoFile.write(geoLine)

            #
            # define circle arcs defining the 8 longitude-latitude patches
            #
            geoLine = "c1 = "+str(c)+"; Circle(c1) = {p6, p1, p4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c2 = "+str(c)+"; Circle(c2) = {p4, p1, p7};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c3 = "+str(c)+"; Circle(c3) = {p7, p1, p5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c4 = "+str(c)+"; Circle(c4) = {p5, p1, p6};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c5 = "+str(c)+"; Circle(c5) = {p6, p1, p2};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c6 = "+str(c)+"; Circle(c6) = {p2, p1, p7};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c7 = "+str(c)+"; Circle(c7) = {p7, p1, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c8 = "+str(c)+"; Circle(c8) = {p3, p1, p6};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c9 = "+str(c)+"; Circle(c9) = {p4, p1, p2};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c10 = "+str(c)+"; Circle(c10) = {p2, p1, p5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c11 = "+str(c)+"; Circle(c11) = {p5, p1, p3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c12 = "+str(c)+"; Circle(c12) = {p3, p1, p4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            #
            # define the patches as ruled surfaces based on collection of three arcs each
            #
            geoLine = "c13 = "+str(c)+"; Line Loop(c13) = {c2, c7, c12};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c14 = "+str(c)+"; Ruled Surface(c14) = {c13};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c15 = "+str(c)+"; Line Loop(c15) = {c12, -c1, -c8};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c16 = "+str(c)+"; Ruled Surface(c16) = {c15};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c17 = "+str(c)+"; Line Loop(c17) = {c1, c9, -c5};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c18 = "+str(c)+"; Ruled Surface(c18) = {c17};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c19 = "+str(c)+"; Line Loop(c19) = {c2, -c6, -c9};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c20 = "+str(c)+"; Ruled Surface(c20) = {c19};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c21 = "+str(c)+"; Line Loop(c21) = {c6, c3, -c10};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c22 = "+str(c)+"; Ruled Surface(c22) = {c21};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c23 = "+str(c)+"; Line Loop(c23) = {c7, -c11, -c3};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c24 = "+str(c)+"; Ruled Surface(c24) = {c23};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c25 = "+str(c)+"; Line Loop(c25) = {c11, c8, -c4};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c26 = "+str(c)+"; Ruled Surface(c26) = {c25};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c27 = "+str(c)+"; Line Loop(c27) = {c4, c5, c10};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)
            geoLine = "c28 = "+str(c)+"; Ruled Surface(c28) = {c27};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

            #
            # define the overall sphere surface
            #
            geoLine = "c29 = "+str(c)+"; Surface Loop(c29) = {c14, c20, c22, c24, c26, c16, c18, c28};"+"\n"
            holeID.append(c)
            holeStr = holeStr + str(c) + ","
            c       = c + 1
            geoFile.write(geoLine)

            geoLine = "c30 = "+str(c)+"; Physical Surface(c30) = {c14, c20, c22, c24, c26, c16, c18, c28};"+"\n"
            c       = c + 1
            geoFile.write(geoLine)

    geoLine = "b1 = newp; Point(b1) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+","+str(a_Box[0,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b2 = newp; Point(b2) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+","+str(a_Box[0,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b3 = newp; Point(b3) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+","+str(a_Box[0,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b4 = newp; Point(b4) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+","+str(a_Box[0,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b5 = newp; Point(b5) = {"+str(a_Box[0,0])+","+str(a_Box[0,1])+","+str(a_Box[1,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b6 = newp; Point(b6) = {"+str(a_Box[0,0])+","+str(a_Box[1,1])+","+str(a_Box[1,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b7 = newp; Point(b7) = {"+str(a_Box[1,0])+","+str(a_Box[1,1])+","+str(a_Box[1,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)
    geoLine = "b8 = newp; Point(b8) = {"+str(a_Box[1,0])+","+str(a_Box[0,1])+","+str(a_Box[1,2])+","+str(hB)+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l1 = newreg; Line(l1) = {b2, b6};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l2 = newreg; Line(l2) = {b6, b7};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l3 = newreg; Line(l3) = {b7, b3};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l4 = newreg; Line(l4) = {b3, b2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l5 = newreg; Line(l5) = {b2, b1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l6 = newreg; Line(l6) = {b1, b5};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l7 = newreg; Line(l7) = {b5, b6};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l8 = newreg; Line(l8) = {b5, b8};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l9 = newreg; Line(l9) = {b8, b4};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l10 = newreg; Line(l10) = {b4, b1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l11 = newreg; Line(l11) = {b3, b4};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l12 = newreg; Line(l12) = {b7, b8};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l13 = newreg; Line Loop(l13) = {l2, l12, -l8, l7};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l14 = newreg; Ruled Surface(l14) = {l13};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l15 = newreg; Physical Surface(l15) = {l14};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l16 = newreg; Line Loop(l16) = {l3, l11, -l9, -l12};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l17 = newreg; Ruled Surface(l17) = {l16};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l18 = newreg; Physical Surface(l18) = {l17};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l19 = newreg; Line Loop(l19) = {l3, l4, l1, l2};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l20 = newreg; Ruled Surface(l20) = {l19};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l21 = newreg; Physical Surface(l21) = {l20};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l22 = newreg; Line Loop(l22) = {l5, l6, l7, -l1};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l23 = newreg; Ruled Surface(l23) = {l22};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l24 = newreg; Physical Surface(l24) = {l23};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l25 = newreg; Line Loop(l25) = {l10, l6, l8, l9};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l26 = newreg; Ruled Surface(l26) = {l25};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l27 = newreg; Physical Surface(l27) = {l26};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l28 = newreg; Line Loop(l28) = {l4, l5, -l10, -l11};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l29 = newreg; Ruled Surface(l29) = {l28};"+"\n"
    geoFile.write(geoLine)
    geoLine = "l30 = newreg; Physical Surface(l30) = {l29};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l31 = newreg; Surface Loop(l31) = {l20, l17, l29, l23, l26, l14};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l32 = newreg; Volume(l32) = {l31"+holeStr[:-1]+"};"+"\n"
    geoFile.write(geoLine)

    geoLine = "l33 = newreg; Physical Volume(l33) = {l32};"+"\n"
    geoFile.write(geoLine)

