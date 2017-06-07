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
	if (fgs.z > 1) { // 3D
		mGlobalSize = Vec3i((cgs.x-2)*(fgs.x-2)+2, (cgs.y-2)*(fgs.y-2)+2, (cgs.z-2)*(fgs.z-2)+2);
		mGridSize = mGlobalSize;
	} else {
		mGlobalSize = Vec3i((cgs.x-2)*(fgs.x-2)+2, (cgs.y-2)*(fgs.y-2)+2, cgs.z*fgs.z);
		mGridSize = mGlobalSize;
	}

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
	mGlobalData = FluidData(this);
	boundaryWidth = bWidth;

	if (mCoarseSolver == NULL)
		return;
	mCoarseData = FluidData(mCoarseSolver);
	mCoarseData.mFlags->initDomain();
	mCoarseData.mFlags->fillGrid();
	PbType pt;
	pt.S = "MACGrid";
	mCoarseOldVel = (MACGrid*)mCoarseSolver->create(pt, PbTypeVec(), "");

	if (mFineSolver == NULL)
		return;

	// init fine grids:
	// if the mCoarseSize is (x,y,z) then there are (x-2)*(y-2)*(z-2) fine grids
	if (is3D()) {
		for (int i = 1; i < mCoarseSize.x-1; i++) {
			for (int j = 1; j < mCoarseSize.y-1; j++) {
				for (int k = 1; k < mCoarseSize.z-1; k++) {
					FluidData mFineData = FluidData(mFineSolver);
					mFineData.mFlags->initDomain();

					mFineData.mFlags->fillGrid();
					mFineDataList.push_back(mFineData);
				}
			}
		}
	} else { //2D
		for (int i = 1; i < mCoarseSize.x-1; i++) {
			for (int j = 1; j < mCoarseSize.y-1; j++) {
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
	mCoarseOldVel->copyFrom(*(mCoarseData.mVel));
}

Vec3 MultiGridSolver::calculateCoarseCell(int i, int j, int k) {
	// printf("calculateCoarseCell()\n");
	FluidData &cell = mFineDataList[fineGridIndex(i,j,k)];
	Vec3 v(0,0,0);
	int cnt = 0;
	for (int x = 1; x < mFineSize.x-1; x++) {
		for (int y = 1; y < mFineSize.y-1; y++) {
			if (mFineSize.z == 1) { // 2D
				cnt++;
				v += cell.mVel->getCentered(x,y,0);
				continue;
			}
			// 3D
			for (int z = 1; z < mFineSize.z-1; z++) {
				cnt++;
				v += cell.mVel->getCentered(x,y,z);
			}
		}
	}

	v = v/cnt;

	v.x = v.x / (mFineSize.x-2);
	v.y = v.y / (mFineSize.y-2);
	if (mFineSize.z != 1)
		v.z = v.z / (mFineSize.z-2);

	return v;
}

void MultiGridSolver::calculateFineGrid() {
	// for (int idx = 0; idx < mCoarseSize.x; idx++) {
	// 	for (int idy = 0; idy < mCoarseSize.y; idy++) {
	// 		for (int idz = 0; idz < mCoarseSize.z; idz++) {
	// 			Vec3 dv = mCoarseData.mVel->getAt(idx, idy, idz)
	// 					- mCoarseOldVel->getAt(idx, idy, idz);
	// 			FluidData &fdata = mFineDataList[fineGridIndex(idx, idy, idz)];
	// 			fdata.mVel->addConst(dv);	
	// 		}
	// 	}
	// }
	if (is3D()) {	
		
	} else {
		for (int i = 1; i < mGlobalSize.x-1; i++) {
			for (int j = 1; j < mGlobalSize.y-1; j++) {

			}
		}
	}
}

void MultiGridSolver::gatherGlobalData() {
	for (int idx = 0; idx < mCoarseSize.x; idx++) {
		for (int idy = 0; idy < mCoarseSize.y; idy++) {
			for (int idz = 0; idz < mCoarseSize.z; idz++) {
				Vec3i pos;
				if (mFineSize.z == 1) { //2D
					pos = Vec3i(idx, idy, idz)*(mFineSize-Vec3i(2,2,0));
				} else { //3D
					pos = Vec3i(idx, idy, idz)*(mFineSize-Vec3i(2,2,2));
				}
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