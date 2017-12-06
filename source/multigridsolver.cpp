#include "multigridsolver.h"
#include "grid.h"
#include <sstream>
#include <fstream>
// #include "plugin/numpyconvert.cpp"

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

void FluidData::reset() {
	// Notice mFlag will not be reset
	mVel->clear();
	mDensity->clear();
	mReact->clear();
	mFuel->clear();
	mHeat->clear();
	mFlame->clear();
	mPressure->clear();
}

Vec3i MultiGridSolver::getCoarseSize () {return mCoarseSize;}
Vec3i MultiGridSolver::getFineSize   () {return mFineSize;}
Vec3i MultiGridSolver::getFineGridNum() {return mFineGridNum;}

MultiGridSolver::MultiGridSolver(Vec3i cgs, Vec3i fgs, Vec3i ggs, int dim)
	: FluidSolver(ggs, dim, -1) {
	if (is3D()) { // 3D
		mGlobalSize = ggs;
		mFineGridNum = cgs - Vec3i(2,2,2);
		mFineSizeEffective = fgs - Vec3i(2,2,2);
	} else {
		mGlobalSize = ggs;
		mFineGridNum = cgs - Vec3i(2,2,0);
		mFineSizeEffective = fgs - Vec3i(2,2,0);
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

void MultiGridSolver::initMultiGrid(int bWidth) {
	mGlobalData = FluidData(this);
	mGlobalData.mFlags->initDomain();
	mGlobalData.mFlags->fillGrid();

	boundaryWidth = bWidth;

	if (mCoarseSolver == NULL)
		return;
	mCoarseData = FluidData(mCoarseSolver);
	mCoarseData.mFlags->initDomain();
	mCoarseData.mFlags->fillGrid();
	PbType pt;
	pt.S = "MACGrid";

	mCoarseOldVel = (MACGrid*)mCoarseSolver->create(pt, PbTypeVec(), "");
	mCoarseOldVel_Enlarged = (MACGrid*)create(pt, PbTypeVec(), "");
	mCoarseNewVel_Enlarged = (MACGrid*)create(pt, PbTypeVec(), "");
	mGlobalVel_tmp = (MACGrid*)create(pt, PbTypeVec(), "");

	if (mFineSolver == NULL)
		return;

	// init fine grids:
	// if the mCoarseSize is (x,y,z) then there are (x-2)*(y-2)*(z-2) fine grids
	for (int i = 0; i < mFineGridNum.x; i++) {
		for (int j = 0; j < mFineGridNum.y; j++) {
			for (int k = 0; k < mFineGridNum.z; k++) {
				FluidData mFineData = FluidData(mFineSolver);
				mFineData.mFlags->initDomain();

				mFineData.mFlags->fillGrid();
				mFineDataList.push_back(mFineData);
			}
		}
	}

	printf("FineGridNum:%d*%d*%d\n", mFineGridNum.x, mFineGridNum.y, mFineGridNum.z);

	printf("initMultiGrid() finished.\n");
}

void MultiGridSolver::resetGrid() {
	mGlobalData.reset();
	mCoarseData.reset();

	mCoarseOldVel->clear();
	mGlobalVel_tmp->clear();

	for (int i = 0; i < mFineGridNum.x; i++) {
		for (int j = 0; j < mFineGridNum.y; j++) {
			for (int k = 0; k < mFineGridNum.z; k++) {
				FluidData &fdata = mFineDataList[fineGridIndex(i,j,k)];
				fdata.reset();
			}
		}
	}
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

PbClass* MultiGridSolver::getFineFlagsGridObj(int i, int j, int k) {
	return (PbClass*)mFineDataList[fineGridIndex(i,j,k)].mFlags;
}

extern void copyGridToArrayMAC(const MACGrid& _Source, PyArrayContainer target);
void MultiGridSolver::copyToArray_CoarseVel(PyArrayContainer oldVel,
											PyArrayContainer newVel) {
	copyGridToArrayMAC(*mCoarseOldVel_Enlarged, oldVel);
	copyGridToArrayMAC(*mCoarseNewVel_Enlarged, newVel);
}

void MultiGridSolver::mapDataToFineGrid() {
	//printf("%s\n", __func__);
	if (is3D()) {
		for (int i = 0; i < mFineGridNum.x; i++) {
			for (int j = 0; j < mFineGridNum.y; j++) {
				for (int k = 0; k < mFineGridNum.z; k++) {
					FluidData &fdata = mFineDataList[fineGridIndex(i,j,k)];
					Vec3i start = Vec3i(i,j,k)*mFineSizeEffective;
					fdata.mVel->copyFromGlobal(*(mGlobalData.mVel),
						start.x, start.y, start.z,
						mFineSize.x, mFineSize.y, mFineSize.z);
					fdata.mPressure->copyFromGlobal(*(mGlobalData.mPressure),
						start.x, start.y, start.z,
						mFineSize.x, mFineSize.y, mFineSize.z);
				}
			}
		}
	} else {
		for (int i = 0; i < mFineGridNum.x; i++) {
			for (int j = 0; j < mFineGridNum.y; j++) {
				FluidData &fdata = mFineDataList[fineGridIndex(i,j,0)];
				Vec3i start = Vec3i(i,j,0)*mFineSizeEffective;
				fdata.mVel->copyFromGlobal(*(mGlobalData.mVel),
					start.x, start.y, start.z,
					mFineSize.x, mFineSize.y, mFineSize.z);
				fdata.mPressure->copyFromGlobal(*(mGlobalData.mPressure),
					start.x, start.y, start.z,
					mFineSize.x, mFineSize.y, mFineSize.z);
			}
		}
	}
}

void MultiGridSolver::mapDataToCoarseGrid() {
	//printf("%s\n", __func__);
	for (int i = 0; i < mFineGridNum.x; i++) {
		for (int j = 0; j < mFineGridNum.y; j++) {
			for (int k = 0; k < mFineGridNum.z; k++) {
				std::pair<Vec3, float> v_and_p = calculateCoarseCell(i,j,k);

				mCoarseData.mVel->setAt(i+1, j+1, k+1, v_and_p.first);
				mCoarseData.mPressure->setAt(i+1, j+1, k+1, v_and_p.second);
			}
		}
	}
	mCoarseOldVel->copyFrom(*(mCoarseData.mVel));
}

std::pair<Vec3, float> MultiGridSolver::calculateCoarseCell(int i, int j, int k) {
	//printf("%s\n", __func__);
	FluidData &cell = mFineDataList[fineGridIndex(i,j,k)];
	Vec3 v(0,0,0);
	float pressure = 0;
	int cnt = 0;
	for (int x = 1; x < mFineSize.x-1; x++) {
		for (int y = 1; y < mFineSize.y-1; y++) {
			if (is3D()) { // 3D
				for (int z = 1; z < mFineSize.z-1; z++) {
					if ((x != 1 && x != mFineSize.x-2)
						&& (y != 1 && y != mFineSize.y-2)
						&& (z != 1 && z != mFineSize.z-2))
						continue;
					cnt++;
					v += cell.mVel->getAt(x,y,z);
					pressure += cell.mPressure->getAt(x,y,z);
				}
			} else { // 2D
				if ((x != 1 && x != mFineSize.x-2)
					&& (y != 1 && y != mFineSize.y-2))
					continue;
				cnt++;
				v += cell.mVel->getAt(x,y,0);
				pressure += cell.mPressure->getAt(x,y,0);
			}
		}
	}

	v = v/(float)cnt;

	pressure /= (float)cnt;

	return std::make_pair(v, pressure);
}

void MultiGridSolver::mapCoarseDataToFineGrid() {
	//printf("%s\n", __func__);
	for (int i = 0; i < mFineGridNum.x; i++) {
		for (int j = 0; j < mFineGridNum.y; j++) {
			for (int k = 0; k < mFineGridNum.z; k++) {
				FluidData &fdata = mFineDataList[fineGridIndex(i,j,k)];
				Vec3 dv = Vec3(0,0,0);
				if (is3D())
					dv = (mCoarseData.mVel->getAt(i+1, j+1, k+1)
						- mCoarseOldVel->getAt(i+1, j+1, k+1));
				else
					dv = (mCoarseData.mVel->getAt(i+1, j+1, k)
						- mCoarseOldVel->getAt(i+1, j+1, k));

				fdata.mVel->addConst(dv);
			}
		}
	}
}

void MultiGridSolver::gatherGlobalData() {
	//printf("%s\n", __func__);
	Vec3i offset = Vec3i(1,1,1);
	if (!is3D()) offset.z = 0;
	for (int idx = 0; idx < mFineGridNum.x; idx++) {
		for (int idy = 0; idy < mFineGridNum.y; idy++) {
			for (int idz = 0; idz < mFineGridNum.z; idz++) {
				Vec3i pos = Vec3i(idx, idy, idz) * mFineSizeEffective + offset;
				FluidData &fdata = mFineDataList[fineGridIndex(idx,idy,idz)];
				mGlobalData.mVel->copyFromFine(pos, *(fdata.mVel), mFineSize);
				mGlobalData.mPressure->copyFromFine(pos, *(fdata.mPressure), mFineSize);
			}
		}
	}
}

// gather mGlobalVel_tmp(fine data) and coarse vel data(old and new, enlarged)
void MultiGridSolver::gatherTrainData() {
	Vec3i offset = Vec3i(1,1,1);
	if (!is3D()) offset.z = 0;
	for (int idx = 0; idx < mFineGridNum.x; idx++) {
		for (int idy = 0; idy < mFineGridNum.y; idy++) {
			for (int idz = 0; idz < mFineGridNum.z; idz++) {
				Vec3i pos = Vec3i(idx, idy, idz) * mFineSizeEffective + offset;
				FluidData &fdata = mFineDataList[fineGridIndex(idx,idy,idz)];
				mGlobalData.mVel->copyFromFine(pos, *(fdata.mVel), mFineSize);
				mGlobalData.mPressure->copyFromFine(pos, *(fdata.mPressure), mFineSize);

				// calculate enlarged coarse vel data
				for (int i = 0; i < mFineSizeEffective.x; i++) {
					for (int j = 0; j < mFineSizeEffective.y; j++) {
						for (int k = 0; k < mFineSizeEffective.z; k++) {
							auto coarseOldVel = mCoarseOldVel->getAt(
								idx+offset.x, idy+offset.y, idz+offset.z);
							mCoarseOldVel_Enlarged->setAt(
								pos.x+i, pos.y+j, pos.z+k, coarseOldVel);
							
							auto coarseNewVel = mCoarseData.mVel->getAt(
								idx+offset.x, idy+offset.y, idz+offset.z);
							mCoarseNewVel_Enlarged->setAt(
								pos.x+i, pos.y+j, pos.z+k, coarseNewVel);
						}
					}
				}
			}
		}
	}
}

template<class T>
void MultiGridSolver::writeGridData(string filename, Grid<T>* grid) {
	// debMsg( "writing grid to text file " << filename, 1);
	ofstream ofs(filename.c_str());
	FOR_IJK(*grid) {
		ofs << (*grid)(i,j,k).x << ' ' << (*grid)(i,j,k).y << ' ' << (*grid)(i,j,k).z <<"\n";
	}
	ofs.close();
}

void MultiGridSolver::writeFluidData(string filename) {
	string global_filename = "data/global" + filename + ".txt";

	writeGridData("data/global" + filename + ".txt", mGlobalVel_tmp);
	writeGridData("data/coarse" + filename + ".txt", mCoarseNewVel_Enlarged);
	writeGridData("data/coarse_old" + filename + ".txt", mCoarseOldVel_Enlarged);
	writeGridData("data/groundtruth" + filename + ".txt", mGlobalData.mVel);

}

}