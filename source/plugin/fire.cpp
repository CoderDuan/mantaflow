/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2016 Sebastian Barschkis, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * Fire modeling plugin
 *
 ******************************************************************************/

#include "general.h"
#include "grid.h"
#include "vectorbase.h"
#include "multigridsolver.h"

using namespace std;

namespace Manta {

KERNEL (bnd=1)
void KnProcessBurn(Grid<Real>& fuel, Grid<Real>& density, Grid<Real>& react,
				   Grid<Real>* red, Grid<Real>* green, Grid<Real>* blue,
				   Grid<Real>* heat, Real burningRate, Real flameSmoke,
				   Real ignitionTemp, Real maxTemp, Real dt, Vec3 flameSmokeColor)
{
	// Save initial values
	Real origFuel = fuel(i,j,k);
	Real origSmoke = density(i,j,k);
	Real smokeEmit = 0.0f;
	Real flame = 0.0f;
	
	// Process fuel
	fuel(i,j,k) -= burningRate * dt;
	if (fuel(i,j,k) < 0.0f)
		fuel(i,j,k) = 0.0f;
	
	// Process reaction coordinate
	if (origFuel > VECTOR_EPSILON) {
		react(i,j,k) *= fuel(i,j,k) / origFuel;
		flame = pow(react(i,j,k), 0.5f);
	} else {
		react(i,j,k) = 0.0f;
	}
	
	// Set fluid temperature based on fuel burn rate and "flameSmoke" factor
	smokeEmit = (origFuel < 1.0f) ? (1.0 - origFuel) * 0.5f : 0.0f;
	smokeEmit = (smokeEmit + 0.5f) * (origFuel - fuel(i,j,k)) * 0.1f * flameSmoke;
	density(i,j,k) += smokeEmit;
	clamp( density(i,j,k), (Real)0.0f, (Real)1.0f);
	
	// Set fluid temperature from the flame temperature profile
	if (heat && flame)
		(*heat)(i,j,k) = (1.0f - flame) * ignitionTemp + flame * maxTemp;
	
	// Mix new color
	if (smokeEmit > VECTOR_EPSILON) {
		float smokeFactor = density(i,j,k) / (origSmoke + smokeEmit);
		if(red)   (*red)(i,j,k)   = ((*red)(i,j,k)   + flameSmokeColor.x * smokeEmit) * smokeFactor;
		if(green) (*green)(i,j,k) = ((*green)(i,j,k) + flameSmokeColor.y * smokeEmit) * smokeFactor;
		if(blue)  (*blue)(i,j,k)  = ((*blue)(i,j,k)  + flameSmokeColor.z * smokeEmit) * smokeFactor;
	}
}

PYTHON() void processBurn(Grid<Real>& fuel, Grid<Real>& density, Grid<Real>& react,
						  Grid<Real>* red = NULL, Grid<Real>* green = NULL, Grid<Real>* blue = NULL,
						  Grid<Real>* heat = NULL, Real burningRate = 0.75f,
						  Real flameSmoke = 1.0f, Real ignitionTemp = 1.25f,
						  Real maxTemp = 1.75f, Real dt = 0.1f,
						  Vec3 flameSmokeColor = Vec3(0.7f, 0.7f, 0.7f))
{
	KnProcessBurn(fuel, density, react, red, green, blue, heat, burningRate,
				  flameSmoke, ignitionTemp, maxTemp, dt, flameSmokeColor);
}

PYTHON() void processBurnCoarseGrid(MultiGridSolver* mgs)
{
	Grid<Real>* fuel = mgs->getCoarseFuelGrid();
	Grid<Real>* density = mgs->getCoarseDensityGrid();
	Grid<Real>* react = mgs->getCoarseReactGrid();
	Grid<Real>* heat = mgs->getCoarseHeatGrid();

	KnProcessBurn(*fuel, *density, *react, NULL, NULL, NULL, heat,
		0.75f, 1.0f, 1.25f, 1.75f, 0.1f, Vec3(0.7f, 0.7f, 0.7f));

	return;
}

PYTHON() void processBurnFineGrid(MultiGridSolver* mgs)
{
	Vec3i size = mgs->getCoarseSize();
	for (int i = 0; i < size.x; i++) {
		for (int j = 0; j < size.y; j++) {
			for (int k = 0; k < size.z; k++) {
				Grid<Real>* fuel = mgs->getFineFuelGrid(i,j,k);
				Grid<Real>* density = mgs->getFineDensityGrid(i,j,k);
				Grid<Real>* react = mgs->getFineReactGrid(i,j,k);
				Grid<Real>* heat = mgs->getFineHeatGrid(i,j,k);
				KnProcessBurn(*fuel, *density, *react, NULL, NULL, NULL, heat,
					0.75f, 1.0f, 1.25f, 1.75f, 0.1f, Vec3(0.7f, 0.7f, 0.7f));			
			}
		}
	}
}

KERNEL (bnd=1)
void KnUpdateFlame(Grid<Real>& react, Grid<Real>& flame)
{
	if (react(i,j,k) > 0.0f)
		flame(i,j,k) = pow(react(i,j,k), 0.5f);
	else
		flame(i,j,k) = 0.0f;
}

PYTHON() void updateFlame(Grid<Real>& react, Grid<Real>& flame)
{
	KnUpdateFlame(react, flame);
}

PYTHON() void updateFlameCoarseGrid(MultiGridSolver* mgs) {
	KnUpdateFlame(*(mgs->getCoarseReactGrid()), *(mgs->getCoarseFlameGrid()));
}

PYTHON() void updateFlameFineGrid(MultiGridSolver* mgs) {
	Vec3i size = mgs->getCoarseSize();
	for (int i = 0; i < size.x; i++) {
		for (int j = 0; j < size.y; j++) {
			for (int k = 0; k < size.z; k++) {
				KnUpdateFlame(*(mgs->getFineReactGrid(i,j,k)),
					*(mgs->getFineFlameGrid(i,j,k)));	
			}
		}
	}
}

} // namespace
