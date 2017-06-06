
from manta import *

dim = 2
# resolution
resC = 1
resF = 24
# grid size
gsC = vec3(resC, resC, resC)
gsF = vec3(resF+2, resF+2, resF+2)

# buoyancy parameters
smokeDensity = -0.001 # alpha
smokeTempDiff = 0.1   # beta

if dim==2:
	gsC.z = 1
	gsF.z = 1

ms = MultiGridSolver(name='MultiGridSolver', coarseGridSize=gsC, fineGridSize=gsF, dim=dim)
csolver = Solver(name='CoarseSolver', gridSize=gsC, dim=dim)
fsolver = Solver(name='FineSolver', gridSize=gsF, dim=dim)
ms.setMultiGridSolver(csolver, fsolver);
ms.initMultiGrid(dim=dim, bWidth=0)

doOpen = True

setOpenBoundMultiGrid(ms)

ms.frameLength = 1.2
ms.timestepMin = 0.2
ms.timestepMax = 2.0
ms.cfl         = 3.0
ms.timestep    = (ms.timestepMax+ms.timestepMin)*0.5
timings = Timings()

frames = 250

# noise
noise = fsolver.create(NoiseField, loadFromFile=True)
noise.posScale = vec3(45)
noise.clamp = True
noise.clampNeg = 0
noise.clampPos = 1
noise.valScale = 1
noise.valOffset = 0.75
noise.timeAnim = 0.2

# source: cube in center of domain (x, y), standing on bottom of the domain
boxSize = vec3(resF, 0.05*resF, resF/8)
boxCenter = gsF*vec3(0.5, 0.15, 0.5)
sourceBox = ms.create( Box, center=boxCenter, size=boxSize )


# needs positive gravity because of addHeatBuoyancy2()
gravity = vec3(0,-0.0981,0)

vel = ms.getVelObj()

if (GUI):
	gui = Gui()
	gui.show(True)

while ms.frame < frames:
	maxvel = vel.getMaxValue()
	ms.adaptTimestep( maxvel )
	mantaMsg('\nFrame %i, time-step size %f' % (ms.frame, ms.timestep))
	
	# coarse grids:
	ms.calculateCoarseGrid()

	processBurnCoarseGrid(ms)

	advectCoarseGridSL(ms)

	addBuoyancyCoarseDensityGrid(ms, gravity*smokeDensity)
	addBuoyancyCoarseHeatGrid(ms, gravity*smokeTempDiff)

	solvePressureCoarseGrid(ms)

	updateFlameCoarseGrid(ms)

	ms.calculateFineGrid()

	# fine grids:
	if ms.timeTotal<250:
		densityInflowMultiGrid(ms, 0, 0, int(resC/2), noise, sourceBox)

	processBurnFineGrid(ms)

	advectFineGridSL(ms)

	if doOpen:
		resetOutflowFineGrid(ms)

	addBuoyancyFineDensityGrid(ms, gravity*smokeDensity)
	addBuoyancyFineHeatGrid(ms, gravity*smokeTempDiff)

	solvePressureFineGrid(ms)

	updateFlameFineGrid(ms)

	ms.gatherGlobalData()
	# updateFlame( react=react, flame=flame )

	timings.display()
	ms.step()
