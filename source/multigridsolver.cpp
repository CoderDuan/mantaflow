#include "multigridsolver.h"
#include "grid.h"
#include <sstream>
#include <fstream>

using namespace std;
namespace Manta {

MultiGridSolver::MultiGridSolver(Vec3i cgs, Vec3i fgs, int dim)
	: FluidSolver(Vec3i(cgs.x*fgs.x, cgs.y*fgs.y, cgs.z*fgs.z), dim, -1) {
	mGlobalSize = Vec3i(cgs.x*fgs.x, cgs.y*fgs.y, cgs.z*fgs.z);
	mCoarseSize = cgs;
	mFineSize = fgs;
	initMultiGrid(dim);
}

void MultiGridSolver::initMultiGrid(int dim) {
	PbType pt;
	pt.S = "FlagGrid";
	mFlags = (FlagGrid*)create(pt, PbTypeVec(), "");
	pt.S = "MACGrid";
	mVel = (MACGrid*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mDensity = (Grid<Real>*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mReact = (Grid<Real>*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mFuel = (Grid<Real>*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mHeat = (Grid<Real>*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mFlame = (Grid<Real>*)create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mPressure = (Grid<Real>*)create(pt, PbTypeVec(), "");

	mCoarseSolver = new FluidSolver(mCoarseSize, dim, -1);
	mFineSolver = new FluidSolver(mFineSize, dim, -1);

	// coarse grids:
	pt.S = "FlagGrid";
	mCoarseFlags = (FlagGrid*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "MACGrid";
	mCoarseVel = (MACGrid*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarseDensity = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarseReact = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarseFuel = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarseHeat = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarseFlame = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mCoarsePressure = (Grid<Real>*)mCoarseSolver->create(pt, PbTypeVec(), "");

	// fine grids:
	for (int i = 0; i < mCoarseSize.x; i++) {
		for (int j = 0; j < mCoarseSize.y; j++) {
			for (int k = 0; k < mCoarseSize.z; k++) {
					pt.S = "FlagGrid";
					mFineFlags.push_back((FlagGrid*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "MACGrid";
					mFineVel.push_back((MACGrid*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFineDensity.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFineReact.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFineFuel.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFineHeat.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFineFlame.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
					pt.S = "RealGrid";
					mFinePressure.push_back((Grid<Real>*)mFineSolver->create(pt, PbTypeVec(), ""));
			}
		}
	}

}

PbClass* MultiGridSolver::createFlagGrid() {
	return mFlags;
}

PbClass* MultiGridSolver::createVelGrid() {
	return mVel;
}

PbClass* MultiGridSolver::createDensityGrid() {
	return mDensity;
}

PbClass* MultiGridSolver::createReactGrid() {
	return mReact;
}

PbClass* MultiGridSolver::createFuelGrid() {
	return mFuel;
}

PbClass* MultiGridSolver::createHeatGrid() {
	return mHeat;
}

PbClass* MultiGridSolver::createFlameGrid() {
	return mFlame;
}

PbClass* MultiGridSolver::createPressureGrid() {
	return mPressure;
}

}