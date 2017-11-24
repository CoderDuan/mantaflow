import gzip
import os
import sys
import urllib

import tensorflow.python.platform

import numpy
import tensorflow as tf

from manta import *

dim = 3
# resolution
resC = 6
resF = 6
# grid size
gsC = vec3(resC+2, resC+2, resC+2)
gsF = vec3(resF+2, resF+2, resF+2)
gsG = vec3(resC*resF+2, resC*resF+2, resC*resF+2)
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

setOpenBoundMultiGrid(ms)

doOpen = True

# buoyancy parameters
smokeDensity = -0.001 # alpha
smokeTempDiff = 0.1   # beta

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

flags.initDomain( boundaryWidth=bWidth )
flags.fillGrid()
if doOpen:
  setOpenBound( flags, bWidth, 'xXyYzZ', FlagOutflow | FlagEmpty )

# tensorflow settings
# only for 2d
SEED = 666
BATCH_SIZE = 64
CONV_SIZE = 5
VECTOR_DIM = 2

fineGridNum = ms.getFineGridNum()
find_grid_node = tf.placeholder(tf.float32,
  shape = (BATCH_SIZE, gsG.x, gsG.y, VECTOR_DIM))
coarse_grid_node = tf.placeholder(tf.float32,
  shape = (BATCH_SIZE, gsC.x, gsC.y, VECTOR_DIM))
coarse_grid_old_node = tf.placeholder(tf.float32,
  shape = (BATCH_SIZE, gsC.x, gsC.y, VECTOR_DIM))
ground_truth_node = tf.placeholder(tf.float32,
  shape = (BATCH_SIZE, gsG.x, gsG.y, VECTOR_DIM))

conv1_weight = tf.Variable(
  tf.truncated_normal([CONV_SIZE, CONV_SIZE, VECTOR_DIM, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=SEED)))
conv1_biases = tf.Variable(tf.zeros([32]))

conv = tf.nn.conv2d(find_grid_node,
                    conv1_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')


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
  # mantaMsg('1')
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
  # copy to global
  ms.mapCoarseDataToFineGrid()
  ms.gatherGlobalData()

  updateFlame( react=react, flame=flame )

  timings.display()
  ms.step()
