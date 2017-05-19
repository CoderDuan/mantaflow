#include "multigridsolver.h"

namespace Manta {

MultiGridSolver::MultiGridSolver(Vec3i gridSize, int dim)
	: FluidSolver(gridSize, dim, -1) {

}

void MultiGridSolver::initMultiGrid() {
	
}

PbClass* MultiGridSolver::createFlagGrid() {
	PbType t;
	t.S = "FlagGrid";
	mFlags = (FlagGrid*)create(t);
	return mFlags;
}

PbClass* MultiGridSolver::createDensityGrid() {
	return NULL;
}

PbClass* MultiGridSolver::createReactGrid() {
	return NULL;
}

PbClass* MultiGridSolver::createFuelGrid() {
	return NULL;
}

PbClass* MultiGridSolver::createHeatGrid() {
	return NULL;
}

PbClass* MultiGridSolver::createFlameGrid() {
	return NULL;
}

PbClass* MultiGridSolver::createPressureGrid() {
	return NULL;
}

}