/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef GRAINSIZEDUSTDESTRUCTIONRECIPE_HPP
#define GRAINSIZEDUSTDESTRUCTIONRECIPE_HPP

#include "DustDestructionRecipe.hpp"

////////////////////////////////////////////////////////////////////

/** GeneralDustDestructionRecipe derives from DustDestructionRecipe to implement a basic dust
    destruction recipe that calculates a sublimation temperature based on grain size and destroys
    the this dust species when this temperature is exceeded.

    Specifically, there is no destruction if \f$T_\mathrm{eq}<=T_\mathrm{sub}\f$; all grains are
    destroyed if \f$T_\mathrm{eq}>=T_\mathrm{sub}\f$. The values for \f$T_\mathrm{sub}\f$ are determined using Waxman & Drain 2000. As this program only considers a single point in time, it cannot account for the survival time of a dust grain. Therefore the FWHM of the primary source is used to calculate a single sublimation temperature. */
class GrainSizeDustDestructionRecipe : public DustDestructionRecipe
{
    ITEM_CONCRETE(GrainSizeDustDestructionRecipe, DustDestructionRecipe,
                  "a dust destruction recipe using a grain size dependent sublimation temperature")

        PROPERTY_DOUBLE(Prefactor, "This factor is defined by the type of dust")
        ATTRIBUTE_MIN_VALUE(Prefactor, "0")
        ATTRIBUTE_MAX_VALUE(Prefactor, "1000]")
        ATTRIBUTE_DEFAULT_VALUE(Prefactor, "42.747")

        PROPERTY_DOUBLE(FWHM, "The Full Width Half Maximum of the transcient's lightcurve.")
        ATTRIBUTE_QUANTITY(FWHM, "time")
        ATTRIBUTE_MIN_VALUE(FWHM, "[0 yr")
        ATTRIBUTE_DEFAULT_VALUE(FWHM, "0.33 yr")

    ITEM_END()

    //============= Construction - Setup - Destruction =============

protected:
    /** This function verifies that the maximum temperatures are not below the minimum
        temperatures. */
    void setupSelfBefore() override;

    //======================== Other Functions =======================

public:
    /** This function returns the non-destroyed density fraction for a grain population with the
        specified type (graphite or silicate) and equilibrium temperature using the linear
        dependence described in the class header. The radiation field is not
        used. */
    double densityFraction(bool graphite, double a, const Array& Jv, double T) const;

};

////////////////////////////////////////////////////////////////////

#endif
