#include "multigridsolver.h"
#include "grid.h"
#include <sstream>
#include <fstream>

using namespace std;
namespace Manta {

FluidData::FluidData(FluidSolver* p) {
	parent = p;
	PbType pt;
	pt.S = "FlagGrid";
	mFlags = (FlagGrid*)p->create(pt, PbTypeVec(), "");
	pt.S = "MACGrid";
	mVel = (MACGrid*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mDensity = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mReact = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mFuel = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mHeat = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mFlame = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
	pt.S = "RealGrid";
	mPressure = (Grid<Real>*)p->create(pt, PbTypeVec(), "");
}

MultiGridSolver::MultiGridSolver(Vec3i cgs, Vec3i fgs, int dim)
	: FluidSolver(Vec3i(cgs.x*fgs.x, cgs.y*fgs.y, cgs.z*fgs.z), dim, -1) {
	mGlobalSize = Vec3i(cgs.x*fgs.x, cgs.y*fgs.y, cgs.z*fgs.z);
	mCoarseSize = cgs;
	mFineSize = fgs;
}

void MultiGridSolver::setMultiGridSolver(FluidSolver* cs, FluidSolver* fs) {
	if (cs->getGridSize() != mCoarseSize || fs->getGridSize() != mFineSize)
		printf("Invalid Coarse/Fine Solver grid size!\n");
	mCoarseSolver = cs;
	mFineSolver = fs;
}

void MultiGridSolver::initMultiGrid(int dim) {

	PbType pt;
	mGlobalData = FluidData(this);

	if (mCoarseSolver == NULL)
		return;
	mCoarseData = FluidData(mCoarseSolver);

	if (mFineSolver == NULL)
		return;

	// fine grids:
	for (int i = 0; i < mCoarseSize.x; i++) {
		for (int j = 0; j < mCoarseSize.y; j++) {
			for (int k = 0; k < mCoarseSize.z; k++) {
				mFineDataList.push_back(FluidData(mFineSolver));
			}
		}
	}

}

PbClass* MultiGridSolver::getFlagsObj() {
	return (PbClass*)mFlags;
}

PbClass* MultiGridSolver::getVelObj() {
	return (PbClass*)mVel;
}

PbClass* MultiGridSolver::getDensityObj() {
	return (PbClass*)mDensity;
}

PbClass* MultiGridSolver::getReactObj() {
	return (PbClass*)mReact;
}

PbClass* MultiGridSolver::getFuelObj() {
	return (PbClass*)mFuel;
}

PbClass* MultiGridSolver::getHeatObj() {
	return (PbClass*)mHeat;
}

PbClass* MultiGridSolver::getFlameObj() {
	return (PbClass*)mFlame;
}

PbClass* MultiGridSolver::getPressureObj() {
	return (PbClass*)mPressure;
}


void MultiGridSolver::advectCoarseGrid() {

}

void MultiGridSolver::calculateCoarseGrid() {

}

void MultiGridSolver::solveCoarseGrid() {

}

void MultiGridSolver::advectFineGrid() {

}

void MultiGridSolver::calculateFineGrid() {

}

void MultiGridSolver::solveFineGrid() {

}

void MultiGridSolver::gatherGlobalData() {
	for (int idx = 0; idx < mCoarseSize.x; idx++) {
		for (int idy = 0; idy < mCoarseSize.y; idy++) {
			for (int idz = 0; idz < mCoarseSize.z; idz++) {

			}
		}
	}
}

}