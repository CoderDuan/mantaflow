#ifndef _MULTIGRIDSOLVER_H
#define _MULTIGRIDSOLVER_H

#include "fluidsolver.h"
#include "grid.h"

namespace Manta {

PYTHON(name=MultiGridSolver)
class MultiGridSolver : public FluidSolver {
public:
	PYTHON() MultiGridSolver(Vec3i gridSize, int dim=3);

	PYTHON() void initMultiGrid();

	PYTHON() PbClass* createFlagGrid();
	PYTHON() PbClass* createDensityGrid();
	PYTHON() PbClass* createReactGrid();
	PYTHON() PbClass* createFuelGrid();
	PYTHON() PbClass* createHeatGrid();
	PYTHON() PbClass* createFlameGrid();
	PYTHON() PbClass* createPressureGrid();

protected:
	FluidSolver* mGlobalSolver;
	FluidSolver* mCoarseSolver;
	std::vector<FluidSolver*> mFineSolverList;

	Vec3i mGlobalSize;
	Vec3i mCoarseSize;
	Vec3i mFineSize;

	FlagGrid* mFlags;
	MACGrid* mVel;
	Grid<Real>* mDensity;
	Grid<Real>* mReact;
	Grid<Real>* mFuel;
	Grid<Real>* mHeat;
	Grid<Real>* mFlame;
	Grid<Real>* mPressure;

};

}

#endif