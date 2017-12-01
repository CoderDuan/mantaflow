
from manta import *

dim = 2
# resolution
resC = 8
resF = 8
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
ms.initMultiGrid(bWidth=bWidth)

doOpen = True

setOpenBoundMultiGrid(ms)

ms.frameLength = 1.2
ms.cfl         = 3.0
ms.timestep    = 0.2 #(ms.timestepMax+ms.timestepMin)*0.5
timings = Timings()

frames = 20

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


# needs positive gravity because of addHeatBuoyancy2()
gravity = vec3(0,-0.0981,0)

vel     = ms.getVelObj()
flags   = ms.getFlagsObj()
density = ms.getDensityObj()
heat    = ms.getHeatObj()
fuel    = ms.getFuelObj()
react   = ms.getReactObj()
flame   = ms.getFlameObj()
pressure = ms.getPressureObj()

if doOpen:
	setOpenBound( flags, bWidth, 'xXyYzZ', FlagOutflow | FlagEmpty )

if (GUI):
	gui = Gui()
	gui.show(True)

first_frame = True
cnt = 0

MAX_SIM = 8
for boxSizeX in range(1, MAX_SIM):
	for boxSizeY in range(1, MAX_SIM):
		for boxCenterX in range(0, MAX_SIM):
			for boxCenterY in range(1, MAX_SIM-1):
				first_frame = True
				ms.resetGrid();
				setOpenBound( flags, bWidth, 'xXyYzZ', FlagOutflow | FlagEmpty )

				boxSize = gsG * vec3(float((MAX_SIM-boxSizeX)/MAX_SIM/2.0),
									 float(boxSizeY/MAX_SIM/2.2),
									 0.03)
				boxCenter = gsG*vec3(float(boxCenterX/MAX_SIM),
									 float(boxCenterY/MAX_SIM),
									 0.5)
				sourceBox = ms.create( Box, center=boxCenter, size=boxSize )

				total = cnt + 100
				print ('boxSize:(%.3f, %.3f, %.3f), boxCenter:(%.3f, %.3f,%.3f), cnt:%d'
					% (boxSize.x, boxSize.y, boxSize.z, 
					   boxCenter.x, boxCenter.y, boxCenter.z,
					   cnt))
				while cnt < total:
					maxvel = vel.getMax()
					# ms.adaptTimestep( maxvel * 10.0 )
					# mantaMsg('\nFrame %i, time-step size %f, maxVel: %f' % (ms.frame, ms.timestep, maxvel))
					# global inflow
					# if ms.timeTotal < 250:
					densityInflow(flags=flags, density=density, noise=noise, shape=sourceBox,
						scale=1, sigma=0.5)
					densityInflow(flags=flags, density=heat, noise=noise, shape=sourceBox,
						scale=1, sigma=0.5)
					densityInflow(flags=flags, density=fuel, noise=noise, shape=sourceBox,
						scale=1, sigma=0.5)
					densityInflow(flags=flags, density=react, noise=noise, shape=sourceBox,
						scale=1, sigma=0.5)

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
						#resetOutflowCoarseGrid(ms)
						#resetOutflowFineGrid(ms)

					# global add buoyancy
					addBuoyancy(flags=flags, density=density, vel=vel, gravity=(gravity*smokeDensity))
					addBuoyancy(flags=flags, density=heat,    vel=vel, gravity=(gravity*smokeTempDiff))

					# global copy to fine
					ms.mapDataToFineGrid()

					# global copy to coarse
					ms.mapDataToCoarseGrid()

					# solve pressure fine
					solvePressureFineGrid(ms)

					# solve pressure coarse
					solvePressureCoarseGrid(ms)

					solvePressure( flags=flags, vel=vel, pressure=pressure )

					ms.gatherGlobalData()
					if (first_frame):
						first_frame = False
					else:
						ms.writeFluidData(str(cnt))
						# print ("cnt:", cnt)
						cnt = cnt + 1

					# copy to global
					# ms.mapCoarseDataToFineGrid()
					# ms.gatherGlobalData()

					updateFlame( react=react, flame=flame )

					# timings.display()
					ms.step()