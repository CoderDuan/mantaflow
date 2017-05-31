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

	// init all data grids
	PYTHON() void initMultiGrid(int dim, int bWidth=1);

	PYTHON() void setMultiGridSolver(FluidSolver* coarseGridSolver, FluidSolver* fineGridSolver);

	Vec3i getCoarseSize() {return mCoarseSize;}
	Vec3i getFineSize() {return mFineSize;}
	inline int fineGridIndex(int i, int j, int k) {
		return (i*mCoarseSize.y*mCoarseSize.z + j*mCoarseSize.z + k);
	}

	PYTHON() void calculateCoarseGrid();
	PYTHON() void solveCoarseGrid();

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

	int boundaryWidth;

	// Global grids
	FlagGrid* getFlagsGrid() {return mGlobalData.mFlags;}
	MACGrid* getVelGrid() {return mGlobalData.mVel;}
	Grid<Real>* getDensityGrid() {return mGlobalData.mDensity;}
	Grid<Real>* getReactGrid() {return mGlobalData.mReact;}
	Grid<Real>* getFuelGrid() {return mGlobalData.mFuel;}
	Grid<Real>* getHeatGrid() {return mGlobalData.mHeat;}
	Grid<Real>* getFlameGrid() {return mGlobalData.mFlame;}
	Grid<Real>* getPressureGrid() {return mGlobalData.mPressure;}

	// Coarse grids
	FlagGrid* getCoarseFlagsGrid() {return mCoarseData.mFlags;}
	MACGrid* getCoarseVelGrid() {return mCoarseData.mVel;}
	Grid<Real>* getCoarseDensityGrid() {return mCoarseData.mDensity;}
	Grid<Real>* getCoarseReactGrid() {return mCoarseData.mReact;}
	Grid<Real>* getCoarseFuelGrid() {return mCoarseData.mFuel;}
	Grid<Real>* getCoarseHeatGrid() {return mCoarseData.mHeat;}
	Grid<Real>* getCoarseFlameGrid() {return mCoarseData.mFlame;}
	Grid<Real>* getCoarsePressureGrid() {return mCoarseData.mPressure;}

	// Fine grids
	FlagGrid* getFineFlagsGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mFlags;}
	MACGrid* getFineVelGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mVel;}
	Grid<Real>* getFineDensityGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mDensity;}
	Grid<Real>* getFineReactGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mReact;}
	Grid<Real>* getFineFuelGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mFuel;}
	Grid<Real>* getFineHeatGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mHeat;}
	Grid<Real>* getFineFlameGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mFlame;}
	Grid<Real>* getFinePressureGrid(int i, int j, int k) {return mFineDataList[fineGridIndex(i,j,k)].mPressure;}

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
};

}

#endif