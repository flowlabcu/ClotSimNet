# trace generated using paraview version 5.11.1
import os
import argparse

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', required=True, help='Path to case directory')
    return parser.parse_args()

def paraview_convert(case_dir):
    # Setup paths
    pvd_path = os.path.join(case_dir, 'output', 'c.pvd')
    sim_id = os.path.basename(case_dir)
    print(f'Converting {pvd_path}')

    # create a new 'PVD Reader'
    cpvd = PVDReader(registrationName='c.pvd', FileName=pvd_path)
    cpvd.CellArrays = ['connectivity', 'offsets', 'types']
    cpvd.PointArrays = ['c']

    # Properties modified on cpvd
    cpvd.CellArrays = []

    # # get active view
    # renderView1 = GetActiveViewOrCreate('RenderView')

    # # show data in view
    # cpvdDisplay = Show(cpvd, renderView1, 'UnstructuredGridRepresentation')

    # # get color transfer function/color map for 'c'
    # cLUT = GetColorTransferFunction('c')

    # # get opacity transfer function/opacity map for 'c'
    # cPWF = GetOpacityTransferFunction('c')

    # # trace defaults for the display properties.
    # cpvdDisplay.Representation = 'Surface'
    # cpvdDisplay.ColorArrayName = ['POINTS', 'c']
    # cpvdDisplay.LookupTable = cLUT
    # cpvdDisplay.SelectTCoordArray = 'None'
    # cpvdDisplay.SelectNormalArray = 'None'
    # cpvdDisplay.SelectTangentArray = 'None'
    # cpvdDisplay.OSPRayScaleArray = 'c'
    # cpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    # cpvdDisplay.SelectOrientationVectors = 'None'
    # cpvdDisplay.ScaleFactor = 0.1
    # cpvdDisplay.SelectScaleArray = 'c'
    # cpvdDisplay.GlyphType = 'Arrow'
    # cpvdDisplay.GlyphTableIndexArray = 'c'
    # cpvdDisplay.GaussianRadius = 0.005
    # cpvdDisplay.SetScaleArray = ['POINTS', 'c']
    # cpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    # cpvdDisplay.OpacityArray = ['POINTS', 'c']
    # cpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    # cpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    # cpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    # cpvdDisplay.ScalarOpacityFunction = cPWF
    # cpvdDisplay.ScalarOpacityUnitDistance = 0.011304906915519712
    # cpvdDisplay.OpacityArrayName = ['POINTS', 'c']
    # cpvdDisplay.SelectInputVectors = [None, '']
    # cpvdDisplay.WriteLog = ''

    # # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    # cpvdDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 9.845904517234901, 1.0, 0.5, 0.0]

    # # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    # cpvdDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 9.845904517234901, 1.0, 0.5, 0.0]

    # # reset view to fit data
    # renderView1.ResetCamera(False)

    # # get the material library
    # materialLibrary1 = GetMaterialLibrary()

    # # show color bar/color legend
    # cpvdDisplay.SetScalarBarVisibility(renderView1, True)

    # # update the view to ensure updated data information
    # renderView1.Update()

    # # get 2D transfer function for 'c'
    # cTF2D = GetTransferFunction2D('c')

    # create a new 'Python Calculator'
    pythonCalculator1 = PythonCalculator(registrationName='PythonCalculator1', Input=cpvd)
    pythonCalculator1.Expression = ''

    # Properties modified on pythonCalculator1
    pythonCalculator1.Expression = '(c-min(c))/(max(c)-min(c))'

    # # show data in view
    # pythonCalculator1Display = Show(pythonCalculator1, renderView1, 'UnstructuredGridRepresentation')

    # # trace defaults for the display properties.
    # pythonCalculator1Display.Representation = 'Surface'
    # pythonCalculator1Display.ColorArrayName = ['POINTS', 'c']
    # pythonCalculator1Display.LookupTable = cLUT
    # pythonCalculator1Display.SelectTCoordArray = 'None'
    # pythonCalculator1Display.SelectNormalArray = 'None'
    # pythonCalculator1Display.SelectTangentArray = 'None'
    # pythonCalculator1Display.OSPRayScaleArray = 'c'
    # pythonCalculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    # pythonCalculator1Display.SelectOrientationVectors = 'None'
    # pythonCalculator1Display.ScaleFactor = 0.1
    # pythonCalculator1Display.SelectScaleArray = 'c'
    # pythonCalculator1Display.GlyphType = 'Arrow'
    # pythonCalculator1Display.GlyphTableIndexArray = 'c'
    # pythonCalculator1Display.GaussianRadius = 0.005
    # pythonCalculator1Display.SetScaleArray = ['POINTS', 'c']
    # pythonCalculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
    # pythonCalculator1Display.OpacityArray = ['POINTS', 'c']
    # pythonCalculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
    # pythonCalculator1Display.DataAxesGrid = 'GridAxesRepresentation'
    # pythonCalculator1Display.PolarAxes = 'PolarAxesRepresentation'
    # pythonCalculator1Display.ScalarOpacityFunction = cPWF
    # pythonCalculator1Display.ScalarOpacityUnitDistance = 0.011304906915519712
    # pythonCalculator1Display.OpacityArrayName = ['POINTS', 'c']
    # pythonCalculator1Display.SelectInputVectors = [None, '']
    # pythonCalculator1Display.WriteLog = ''

    # # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    # pythonCalculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 9.845904517234901, 1.0, 0.5, 0.0]

    # # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    # pythonCalculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 9.845904517234901, 1.0, 0.5, 0.0]

    # # hide data in view
    # Hide(cpvd, renderView1)

    # # show color bar/color legend
    # pythonCalculator1Display.SetScalarBarVisibility(renderView1, True)

    # # update the view to ensure updated data information
    # renderView1.Update()

    # # set scalar coloring
    # ColorBy(pythonCalculator1Display, ('POINTS', 'result'))

    # # Hide the scalar bar for this color map if no visible data is colored by it.
    # HideScalarBarIfNotNeeded(cLUT, renderView1)

    # # rescale color and/or opacity maps used to include current data range
    # pythonCalculator1Display.RescaleTransferFunctionToDataRange(True, False)

    # # show color bar/color legend
    # pythonCalculator1Display.SetScalarBarVisibility(renderView1, True)

    # # get color transfer function/color map for 'result'
    # resultLUT = GetColorTransferFunction('result')

    # # get opacity transfer function/opacity map for 'result'
    # resultPWF = GetOpacityTransferFunction('result')

    # # get 2D transfer function for 'result'
    # resultTF2D = GetTransferFunction2D('result')

    # create a new 'Resample To Image'
    resampler = ResampleToImage(registrationName='ResampleToImage1', Input=pythonCalculator1)

    # # show data in view
    # resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

    # # trace defaults for the display properties.
    # resampleToImage1Display.Representation = 'Outline'
    # resampleToImage1Display.ColorArrayName = ['POINTS', 'result']
    # resampleToImage1Display.LookupTable = resultLUT
    # resampleToImage1Display.SelectTCoordArray = 'None'
    # resampleToImage1Display.SelectNormalArray = 'None'
    # resampleToImage1Display.SelectTangentArray = 'None'
    # resampleToImage1Display.OSPRayScaleArray = 'c'
    # resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    # resampleToImage1Display.SelectOrientationVectors = 'None'
    # resampleToImage1Display.ScaleFactor = 0.09999989999999999
    # resampleToImage1Display.SelectScaleArray = 'c'
    # resampleToImage1Display.GlyphType = 'Arrow'
    # resampleToImage1Display.GlyphTableIndexArray = 'c'
    # resampleToImage1Display.GaussianRadius = 0.004999994999999999
    # resampleToImage1Display.SetScaleArray = ['POINTS', 'c']
    # resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
    # resampleToImage1Display.OpacityArray = ['POINTS', 'c']
    # resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
    # resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
    # resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
    # resampleToImage1Display.ScalarOpacityUnitDistance = 0.017495445207253223
    # resampleToImage1Display.ScalarOpacityFunction = resultPWF
    # resampleToImage1Display.TransferFunction2D = resultTF2D
    # resampleToImage1Display.OpacityArrayName = ['POINTS', 'c']
    # resampleToImage1Display.ColorArray2Name = ['POINTS', 'c']
    # resampleToImage1Display.IsosurfaceValues = [4.8760623415360955]
    # resampleToImage1Display.SliceFunction = 'Plane'
    # resampleToImage1Display.Slice = 49
    # resampleToImage1Display.SelectInputVectors = [None, '']
    # resampleToImage1Display.WriteLog = ''

    # # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    # resampleToImage1Display.ScaleTransferFunction.Points = [-0.0054704500995863365, 0.0, 0.5, 0.0, 9.757595133171778, 1.0, 0.5, 0.0]

    # # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    # resampleToImage1Display.OpacityTransferFunction.Points = [-0.0054704500995863365, 0.0, 0.5, 0.0, 9.757595133171778, 1.0, 0.5, 0.0]

    # # init the 'Plane' selected for 'SliceFunction'
    # resampleToImage1Display.SliceFunction.Origin = [0.5, 0.5, 0.5]

    # # hide data in view
    # Hide(pythonCalculator1, renderView1)

    # # show color bar/color legend
    # resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

    # # update the view to ensure updated data information
    # renderView1.Update()

    # Save the resampled data
    vti_path = os.path.join(case_dir, 'output', f'{sim_id}.vti')
    SaveData(vti_path, proxy=resampler, 
            ChooseArraysToWrite=1,
            PointDataArrays=['result'])
    print(f'Saved to {vti_path}')

if __name__ == '__main__':
    args = parse_args()
    paraview_convert(args.case_dir)