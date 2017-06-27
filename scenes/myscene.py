
from manta import *

dim = 2
# resolution
resC = 6
resF = 6
# grid size
gsC = vec3(resC+2, resC+2, resC+2)
gsF = vec3(resF+2, resF+2, resF+2)
gsG = vec3(resC*resF+2, resC*resF+2, resC*resF+2)
# buoyancy parameters
smokeDensity = -0.001 # alpha
smokeTempDiff = 0.1   # beta
bWidth = 0
if dim==2:
	gsC.z = 1
	gsF.z = 1
	gsG.z = 1

ms = MultiGridSolver(name='MultiGridSolver',
	coarseGridSize=gsC, fineGridSize=gsF, globalGridSize=gsG, dim=dim)
csolver = Solver(name='CoarseSolver', gridSize=gsC, dim=dim)
fsolver = Solver(name='FineSolver', gridSize=gsF, dim=dim)
ms.setMultiGridSolver(csolver, fsolver);
ms.initMultiGrid(dim=dim, bWidth=bWidth)

doOpen = True

setOpenBoundMultiGrid(ms)

ms.frameLength = 1.2
ms.cfl         = 3.0
ms.timestep    = 0.2 #(ms.timestepMax+ms.timestepMin)*0.5
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
boxSize = gsG * vec3(1/8, 0.05, 1/8)
boxCenter = gsG*vec3(0.5, 0.15, 0.5)
sourceBox = ms.create( Box, center=boxCenter, size=boxSize )


# needs positive gravity because of addHeatBuoyancy2()
gravity = vec3(0,-0.0981,0)

vel     = ms.getVelObj()
flags   = ms.getFlagsObj()
density = ms.getDensityObj()
heat    = ms.getHeatObj()
fuel    = ms.getFuelObj()
react   = ms.getReactObj()
flame   = ms.getFlameObj()


if (GUI):
	gui = Gui()
	gui.show(True)

while ms.frame < frames:
	maxvel = vel.getMaxValue()
	ms.adaptTimestep( maxvel )
	mantaMsg('\nFrame %i, time-step size %f' % (ms.frame, ms.timestep))
	
	# global inflow
	if ms.timeTotal < 250:
		densityInflow(flags=flags, density=density, noise=noise, shape=sourceBox,
			scale=1, sigma=0.5)
		densityInflow(flags=flags, density=heat, noise=noise, shape=sourceBox,
			scale=1, sigma=0.5)
		densityInflow(flags=flags, density=fuel, noise=noise, shape=sourceBox,
			scale=1, sigma=0.5)
		densityInflow(flags=flags, density=react, noise=noise, shape=sourceBox,
			scale=1, sigma=0.5)
	mantaMsg('1')

	# global process burn
	processBurn(fuel=fuel, density=density, react=react, heat=heat)

	# global advectSL
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)
	advectSemiLagrange(flags=flags, vel=vel, grid=heat,    order=2)
	advectSemiLagrange(flags=flags, vel=vel, grid=fuel,    order=2)
	advectSemiLagrange(flags=flags, vel=vel, grid=react,   order=2)
	advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
		openBounds=doOpen, boundaryWidth=bWidth)

	# global reset outflow
	if doOpen:
		resetOutflow(flags=flags, real=density)

	# global add buoyancy
	addBuoyancy(flags=flags, density=density, vel=vel, gravity=(gravity*smokeDensity))
	addBuoyancy(flags=flags, density=heat,    vel=vel, gravity=(gravity*smokeTempDiff))

	# global copy to fine
	ms.mapDataToFineGrid()

	# solve pressure fine
	solvePressureFineGrid(ms)

	# global copy to coarse
	ms.mapDataToCoarseGrid()

	# solve pressure coarse
	solvePressureCoarseGrid(ms)

	# copy to global
	ms.mapCoarseDataToFineGrid()
	ms.gatherGlobalData()

	# global update flame
	updateFlame( react=react, flame=flame )

	# # coarse grids:
	# ms.calculateCoarseGrid()

	# processBurnCoarseGrid(ms)
	# advectCoarseGridSL(ms)

	# resetOutflowCoarseGrid(ms)


	# addBuoyancyCoarseDensityGrid(ms, gravity*smokeDensity/resF)
	# addBuoyancyCoarseHeatGrid(ms, gravity*smokeTempDiff/resF)

	# solvePressureCoarseGrid(ms)

	# updateFlameCoarseGrid(ms)

	# ms.calculateFineGrid()	

	# # fine grids:
	# if ms.timeTotal<250:
	# 	densityInflowMultiGrid(ms, 1, 1, int(gsC.z/2), noise, sourceBox)

	# processBurnFineGrid(ms)

	# advectFineGridSL(ms)

	# if doOpen:
	# 	resetOutflowFineGrid(ms)

	# addBuoyancyFineDensityGrid(ms, gravity*smokeDensity)
	# addBuoyancyFineHeatGrid(ms, gravity*smokeTempDiff)

	# solvePressureFineGrid(ms)

	# updateFlameFineGrid(ms)

	# ms.gatherGlobalData()
	# # updateFlame( react=react, flame=flame )

	timings.display()
	ms.step()
