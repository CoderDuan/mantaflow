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
	// void copyFrom(FluidData from, Vec3i offset, Vec3i size);
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
	Vec3i getFineGridNum() {return mFineGridNum;}
	inline int fineGridIndex(int i, int j, int k) {
		return (i*mCoarseSize.y*mCoarseSize.z + j*mCoarseSize.z + k);
	}

	// calculate data of coarse grid using fine grids data
	PYTHON() void calculateCoarseGrid();

	// calculate data of fine grids using coarse grid data
	PYTHON() void calculateFineGrid();

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

	// boundary width: 0 by default
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
	Vec3i mFineGridNum;
	Vec3i mFineSizeEffective;

	FluidData mGlobalData;
	FluidData mCoarseData;

	MACGrid* mCoarseOldVel;

	vector<FluidData> mFineDataList;

	// calculate velocity of one coarse cell(i,j,k), using data of one fine grid(i,j,k)
	Vec3 calculateCoarseCell(int i, int j, int k);
};

}

#endif