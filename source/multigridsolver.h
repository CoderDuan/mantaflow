#ifndef _MULTIGRIDSOLVER_H
#define _MULTIGRIDSOLVER_H

#include "fluidsolver.h"
#include "grid.h"

using namespace std;

namespace Manta {

PYTHON(name=MultiGridSolver)
class MultiGridSolver : public FluidSolver {
public:
	PYTHON() MultiGridSolver(Vec3i coarseGridSize, Vec3i fineGridSize, int dim=3);

	PYTHON() void initMultiGrid(int dim);

	PYTHON() PbClass* createFlagGrid();
	PYTHON() PbClass* createVelGrid();
	PYTHON() PbClass* createDensityGrid();
	PYTHON() PbClass* createReactGrid();
	PYTHON() PbClass* createFuelGrid();
	PYTHON() PbClass* createHeatGrid();
	PYTHON() PbClass* createFlameGrid();
	PYTHON() PbClass* createPressureGrid();

protected:
	FluidSolver* mGlobalSolver;
	FluidSolver* mCoarseSolver;
	FluidSolver* mFineSolver;

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

	FlagGrid* mCoarseFlags;
	MACGrid* mCoarseVel;
	Grid<Real>* mCoarseDensity;
	Grid<Real>* mCoarseReact;
	Grid<Real>* mCoarseFuel;
	Grid<Real>* mCoarseHeat;
	Grid<Real>* mCoarseFlame;
	Grid<Real>* mCoarsePressure;

	vector<FlagGrid*> mFineFlags;
	vector<MACGrid*> mFineVel;
	vector<Grid<Real>*> mFineDensity;
	vector<Grid<Real>*> mFineReact;
	vector<Grid<Real>*> mFineFuel;
	vector<Grid<Real>*> mFineHeat;
	vector<Grid<Real>*> mFineFlame;
	vector<Grid<Real>*> mFinePressure;

};

}

#endif