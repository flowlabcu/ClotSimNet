# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

def paraview_convert(output_dir):

    vtu_path = os.path.join(output_dir, 'output', 'c000000.vtu')
    sim_id = os.path.basename(output_dir) # Get the simulation id to save jpeg later with unique tag

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # Hide orientation axes
    renderView1.OrientationAxesVisibility = 0

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # create a new 'XML Unstructured Grid Reader'
    c000000vtu = XMLUnstructuredGridReader(registrationName='c000000.vtu', FileName=[vtu_path])
    c000000vtu.PointArrayStatus = ['c']

    # Properties modified on c000000vtu
    c000000vtu.TimeArray = 'None'

    # show data in view
    c000000vtuDisplay = Show(c000000vtu, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'c'
    cLUT = GetColorTransferFunction('c')

    # get opacity transfer function/opacity map for 'c'
    cPWF = GetOpacityTransferFunction('c')

    # trace defaults for the display properties.
    c000000vtuDisplay.Representation = 'Surface'
    c000000vtuDisplay.ColorArrayName = ['POINTS', 'c']
    c000000vtuDisplay.LookupTable = cLUT
    c000000vtuDisplay.SelectTCoordArray = 'None'
    c000000vtuDisplay.SelectNormalArray = 'None'
    c000000vtuDisplay.SelectTangentArray = 'None'
    c000000vtuDisplay.OSPRayScaleArray = 'c'
    c000000vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    c000000vtuDisplay.SelectOrientationVectors = 'None'
    c000000vtuDisplay.ScaleFactor = 0.2
    c000000vtuDisplay.SelectScaleArray = 'c'
    c000000vtuDisplay.GlyphType = 'Arrow'
    c000000vtuDisplay.GlyphTableIndexArray = 'c'
    c000000vtuDisplay.GaussianRadius = 0.01
    c000000vtuDisplay.SetScaleArray = ['POINTS', 'c']
    c000000vtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    c000000vtuDisplay.OpacityArray = ['POINTS', 'c']
    c000000vtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    c000000vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
    c000000vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
    c000000vtuDisplay.ScalarOpacityFunction = cPWF
    c000000vtuDisplay.ScalarOpacityUnitDistance = 0.04083102294970884
    c000000vtuDisplay.OpacityArrayName = ['POINTS', 'c']
    c000000vtuDisplay.SelectInputVectors = [None, '']
    c000000vtuDisplay.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    c000000vtuDisplay.ScaleTransferFunction.Points = [-0.00036176523167882386, 0.0, 0.5, 0.0, 0.013049998105502936, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    c000000vtuDisplay.OpacityTransferFunction.Points = [-0.00036176523167882386, 0.0, 0.5, 0.0, 0.013049998105502936, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera(False)

    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [1.0, 0.5, 6.7]
    renderView1.CameraFocalPoint = [1.0, 0.5, 0.0]

    # show color bar/color legend
    c000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # get 2D transfer function for 'c'
    cTF2D = GetTransferFunction2D('c')

    # create a new 'Python Calculator'
    pythonCalculator1 = PythonCalculator(registrationName='PythonCalculator1', Input=c000000vtu)
    pythonCalculator1.Expression = ''

    # Properties modified on pythonCalculator1
    pythonCalculator1.Expression = '(c-min(c))/(max(c)-min(c))'

    # show data in view
    pythonCalculator1Display = Show(pythonCalculator1, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    pythonCalculator1Display.Representation = 'Surface'
    pythonCalculator1Display.ColorArrayName = ['POINTS', 'c']
    pythonCalculator1Display.LookupTable = cLUT
    pythonCalculator1Display.SelectTCoordArray = 'None'
    pythonCalculator1Display.SelectNormalArray = 'None'
    pythonCalculator1Display.SelectTangentArray = 'None'
    pythonCalculator1Display.OSPRayScaleArray = 'c'
    pythonCalculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    pythonCalculator1Display.SelectOrientationVectors = 'None'
    pythonCalculator1Display.ScaleFactor = 0.2
    pythonCalculator1Display.SelectScaleArray = 'c'
    pythonCalculator1Display.GlyphType = 'Arrow'
    pythonCalculator1Display.GlyphTableIndexArray = 'c'
    pythonCalculator1Display.GaussianRadius = 0.01
    pythonCalculator1Display.SetScaleArray = ['POINTS', 'c']
    pythonCalculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
    pythonCalculator1Display.OpacityArray = ['POINTS', 'c']
    pythonCalculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
    pythonCalculator1Display.DataAxesGrid = 'GridAxesRepresentation'
    pythonCalculator1Display.PolarAxes = 'PolarAxesRepresentation'
    pythonCalculator1Display.ScalarOpacityFunction = cPWF
    pythonCalculator1Display.ScalarOpacityUnitDistance = 0.04083102294970884
    pythonCalculator1Display.OpacityArrayName = ['POINTS', 'c']
    pythonCalculator1Display.SelectInputVectors = [None, '']
    pythonCalculator1Display.WriteLog = ''

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    pythonCalculator1Display.ScaleTransferFunction.Points = [-0.00036176523167882386, 0.0, 0.5, 0.0, 0.013049998105502936, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    pythonCalculator1Display.OpacityTransferFunction.Points = [-0.00036176523167882386, 0.0, 0.5, 0.0, 0.013049998105502936, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(c000000vtu, renderView1)

    # show color bar/color legend
    pythonCalculator1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(pythonCalculator1Display, ('POINTS', 'result'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(cLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    pythonCalculator1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    pythonCalculator1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'result'
    resultLUT = GetColorTransferFunction('result')

    # get opacity transfer function/opacity map for 'result'
    resultPWF = GetOpacityTransferFunction('result')

    # get 2D transfer function for 'result'
    resultTF2D = GetTransferFunction2D('result')

    # rescale color and/or opacity maps used to exactly fit the current data range
    pythonCalculator1Display.RescaleTransferFunctionToDataRange(False, True)

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    resultLUT.ApplyPreset('X Ray', True)

    # reset view to fit data bounds
    renderView1.ResetCamera(0.0, 2.0, 0.0, 1.0, 0.0, 0.0, True)

    # hide color bar/color legend
    pythonCalculator1Display.SetScalarBarVisibility(renderView1, False)

    # get layout
    layout1 = GetLayout()

    # layout/tab size in pixels
    layout1.SetSize(1077, 648)

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [1.0, 0.5, 4.319751617610021]
    renderView1.CameraFocalPoint = [1.0, 0.5, 0.0]
    renderView1.CameraParallelScale = 0.6678445058147005

    # save screenshot
    image_path = os.path.join(output_dir, 'output', f'{sim_id}.jpeg')
    SaveScreenshot(image_path, renderView1, ImageResolution=[1280, 720],
        OverrideColorPalette='WhiteBackground', 
        # JPEG options
        Quality=100)

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(1077, 648)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [1.0, 0.5, 4.319751617610021]
    renderView1.CameraFocalPoint = [1.0, 0.5, 0.0]
    renderView1.CameraParallelScale = 0.6678445058147005

    #--------------------------------------------
    # uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    
    return image_path