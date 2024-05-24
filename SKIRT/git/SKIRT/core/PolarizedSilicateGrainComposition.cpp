/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "PolarizedSilicateGrainComposition.hpp"

//////////////////////////////////////////////////////////////////////

string PolarizedSilicateGrainComposition::name() const
{
    return "Polarized_Draine_Silicate";
}

//////////////////////////////////////////////////////////////////////

double PolarizedSilicateGrainComposition::bulkDensity() const
{
    return 3.0e3;
}

//////////////////////////////////////////////////////////////////////

string PolarizedSilicateGrainComposition::resourceNameForOpticalProps() const
{
    return "StokesSilicateOpticalProps";
}

//////////////////////////////////////////////////////////////////////

string PolarizedSilicateGrainComposition::resourceNameForMuellerMatrix() const
{
    return "StokesSilicateMuellerMatrix";
}

//////////////////////////////////////////////////////////////////////

string PolarizedSilicateGrainComposition::resourceNameForEnthalpies() const
{
    return "DraineSilicateEnthalpies";
}

//////////////////////////////////////////////////////////////////////
