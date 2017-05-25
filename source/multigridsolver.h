#ifndef _MULTIGRIDSOLVER_H
#define _MULTIGRIDSOLVER_H

#include "grid.h"
#include "fluidsolver.h"


using namespace std;

namespace Manta {

struct FluidData
{
	/* data */
	FluidSolver* parent;
	FlagGrid* mFlags;
	MACGrid* mVel;
	Grid<Real>* mDensity;
	Grid<Real>* mReact;
	Grid<Real>* mFuel;
	Grid<Real>* mHeat;
	Grid<Real>* mFlame;
	Grid<Real>* mPressure;
	FluidData() {};
	FluidData(FluidSolver* parent);
};

PYTHON(name=MultiGridSolver)
class MultiGridSolver : public FluidSolver {
public:
	PYTHON() MultiGridSolver(Vec3i coarseGridSize, Vec3i fineGridSize, int dim=3);

	PYTHON() void initMultiGrid(int dim);

	PYTHON() void setMultiGridSolver(FluidSolver* coarseGridSolver, FluidSolver* fineGridSolver);

	PYTHON() void advectCoarseGrid();
	PYTHON() void calculateCoarseGrid();
	PYTHON() void solveCoarseGrid();

	PYTHON() void advectFineGrid();
	PYTHON() void calculateFineGrid();
	PYTHON() void solveFineGrid();

	// gather global data from fine grid for rendering
	PYTHON() void gatherGlobalData();

	PYTHON() PbClass* getFlagsObj();
	PYTHON() PbClass* getVelObj();
	PYTHON() PbClass* getDensityObj();
	PYTHON() PbClass* getReactObj();
	PYTHON() PbClass* getFuelObj();
	PYTHON() PbClass* getHeatObj();
	PYTHON() PbClass* getFlameObj();
	PYTHON() PbClass* getPressureObj();

	FlagGrid* getFlagsGrid() {return mFlags;}
	MACGrid* getVelGrid() {return mVel;}
	Grid<Real>* getDensityGrid() {return mDensity;}
	Grid<Real>* getReactGrid() {return mReact;}
	Grid<Real>* getFuelGrid() {return mFuel;}
	Grid<Real>* getHeatGrid() {return mHeat;}
	Grid<Real>* getFlameGrid() {return mFlame;}
	Grid<Real>* getPressureGrid() {return mPressure;}

	FlagGrid* getCoarseFlagsGrid() {return mCoarseFlags;}
	MACGrid* getCoarseVelGrid() {return mCoarseVel;}
	Grid<Real>* getCoarseDensityGrid() {return mCoarseDensity;}
	Grid<Real>* getCoarseReactGrid() {return mCoarseReact;}
	Grid<Real>* getCoarseFuelGrid() {return mCoarseFuel;}
	Grid<Real>* getCoarseHeatGrid() {return mCoarseHeat;}
	Grid<Real>* getCoarseFlameGrid() {return mCoarseFlame;}
	Grid<Real>* getCoarsePressureGrid() {return mCoarsePressure;}

protected:
	FluidSolver* mGlobalSolver;
	FluidSolver* mCoarseSolver;
	FluidSolver* mFineSolver;

	Vec3i mGlobalSize;
	Vec3i mCoarseSize;
	Vec3i mFineSize;

	FluidData mGlobalData;
	FluidData mCoarseData;
	vector<FluidData> mFineDataList;

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