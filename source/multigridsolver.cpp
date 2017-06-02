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

void MultiGridSolver::initMultiGrid(int dim, int bWidth) {
	PbType pt;
	mGlobalData = FluidData(this);
	boundaryWidth = bWidth;

	if (mCoarseSolver == NULL)
		return;
	mCoarseData = FluidData(mCoarseSolver);
	mCoarseData.mFlags->initDomain();
	mCoarseData.mFlags->fillGrid();

	if (mFineSolver == NULL)
		return;

	// fine grids:
	for (int i = 0; i < mCoarseSize.x; i++) {
		for (int j = 0; j < mCoarseSize.y; j++) {
			for (int k = 0; k < mCoarseSize.z; k++) {
				FluidData mFineData = FluidData(mFineSolver);
				mFineData.mFlags->initDomain();

				mFineData.mFlags->fillGrid();
				mFineDataList.push_back(mFineData);
			}
		}
	}

	printf("initMultiGrid() finished.\n");
}

PbClass* MultiGridSolver::getFlagsObj() {
	return (PbClass*)mGlobalData.mFlags;
}

PbClass* MultiGridSolver::getVelObj() {
	return (PbClass*)mGlobalData.mVel;
}

PbClass* MultiGridSolver::getDensityObj() {
	return (PbClass*)mGlobalData.mDensity;
}

PbClass* MultiGridSolver::getReactObj() {
	return (PbClass*)mGlobalData.mReact;
}

PbClass* MultiGridSolver::getFuelObj() {
	return (PbClass*)mGlobalData.mFuel;
}

PbClass* MultiGridSolver::getHeatObj() {
	return (PbClass*)mGlobalData.mHeat;
}

PbClass* MultiGridSolver::getFlameObj() {
	return (PbClass*)mGlobalData.mFlame;
}

PbClass* MultiGridSolver::getPressureObj() {
	return (PbClass*)mGlobalData.mPressure;
}


void MultiGridSolver::calculateCoarseGrid() {
	// printf("calculateCoarseGrid()\n");
	for (int i = 0; i < mCoarseSize.x; i++) {
		for (int j = 0; j < mCoarseSize.y; j++) {
			for (int k = 0; k < mCoarseSize.z; k++) {
				mCoarseData.mVel->setAt(i,j,k,calculateCoarseCell(i,j,k));
			}
		}
	}
}

Vec3 MultiGridSolver::calculateCoarseCell(int i, int j, int k) {
	// printf("calculateCoarseCell()\n");
	FluidData &cell = mFineDataList[fineGridIndex(i,j,k)];
	Vec3 v(0,0,0);
	for (int x = 0; x < mFineSize.x; x++) {
		for (int y = 0; y < mFineSize.y; y++) {
			for (int z = 0; z < mFineSize.z; z++) {
				v += cell.mVel->getCentered(x,y,z);
			}
		}
	}

	return (v/(mFineSize.x*mFineSize.y*mFineSize.z));
}

void MultiGridSolver::calculateFineGrid() {

}

void MultiGridSolver::gatherGlobalData() {
	for (int idx = 0; idx < mCoarseSize.x; idx++) {
		for (int idy = 0; idy < mCoarseSize.y; idy++) {
			for (int idz = 0; idz < mCoarseSize.z; idz++) {
				Vec3i pos = Vec3i(idx, idy, idz)*mFineSize;
				FluidData &fdata = mFineDataList[fineGridIndex(idx,idy,idz)];
				mGlobalData.mFlags->copyFromFine(pos, *(fdata.mFlags), mFineSize);
				mGlobalData.mVel->copyFromFine(pos, *(fdata.mVel), mFineSize);
				mGlobalData.mDensity->copyFromFine(pos, *(fdata.mDensity), mFineSize);
				mGlobalData.mReact->copyFromFine(pos, *(fdata.mReact), mFineSize);
				mGlobalData.mFuel->copyFromFine(pos, *(fdata.mFuel), mFineSize);
				mGlobalData.mHeat->copyFromFine(pos, *(fdata.mHeat), mFineSize);
				mGlobalData.mFlame->copyFromFine(pos, *(fdata.mFlame), mFineSize);
				mGlobalData.mPressure->copyFromFine(pos, *(fdata.mPressure), mFineSize);
			}
		}
	}
}

}