/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#ifndef LIGHTCURVE_HPP
#define LIGHTCURVE_HPP

#include "Array.hpp"
#include "Range.hpp"
#include "SimulationItem.hpp"

//////////////////////////////////////////////////////////////////////

/** Lightcurve defines if the timedependence of the Spectral Energy Distribution (SED) should be recorded in a lightcurve and how the bins of this lightcurve are defined. */
class Lightcurve : public SimulationItem
{
    ITEM_CONCRETE(Lightcurve, SimulationItem, "record the time dependence of the SED")
        PROPERTY_BOOL(Active, "if true the time dependence of the SED is saved")
        ATTRIBUTE_DEFAULT_VALUE(Active, "False")

        PROPERTY_INT(numBins, "the number of bins for which the time dependence of the SED is recorded")
        ATTRIBUTE_DEFAULT_VALUE(numBins, "1")
        ATTRIBUTE_MIN_VALUE(numBins, "1")

        PROPERTY_DOUBLE(Range, "the timescale over which the bins are distributed")
        ATTRIBUTE_DEFAULT_VALUE(Range, "1 yr")
        ATTRIBUTE_QUANTITY(Range, "time")
        ATTRIBUTE_MIN_VALUE(FWHM, "0 yr")

    ITEM_END()
};

//////////////////////////////////////////////////////////////////////

#endif
