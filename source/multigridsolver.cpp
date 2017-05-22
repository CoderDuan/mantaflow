#include "multigridsolver.h"
#include "grid.h"
#include <sstream>
#include <fstream>

using namespace std;
namespace Manta {

MultiGridSolver::MultiGridSolver(Vec3i gridSize, int dim)
	: FluidSolver(gridSize, dim, -1) {

}

void MultiGridSolver::initMultiGrid() {
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