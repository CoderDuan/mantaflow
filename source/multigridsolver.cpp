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
}

void MultiGridSolver::setMultiGridSolver(FluidSolver* cs, FluidSolver* fs) {
	if (cs->getGridSize() != mCoarseSize || fs->getGridSize() != mFineSize)
		printf("Invalid Coarse/Fine Solver grid size!\n");
	mCoarseSolver = cs;
	mFineSolver = fs;
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

	if (mCoarseSolver == NULL)
		return;

	// coarse grids:
	pt.S = "FlagGrid";
	mCoarseFlags = (FlagGrid*)mCoarseSolver->create(pt, PbTypeVec(), "mCoarseFlags");
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

}