/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "FullInstrument.hpp"
#include "FluxRecorder.hpp"

////////////////////////////////////////////////////////////////////

void FullInstrument::setupSelfBefore()
{
    FrameInstrument::setupSelfBefore();

    // add SED to FrameInstrument's flux recorder's configuration
    instrumentFluxRecorder()->includeFluxDensityForDistant();
}

////////////////////////////////////////////////////////////////////
