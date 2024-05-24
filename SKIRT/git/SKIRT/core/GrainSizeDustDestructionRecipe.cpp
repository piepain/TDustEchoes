/*//////////////////////////////////////////////////////////////////
////     The SKIRT project -- advanced radiative transfer       ////
////       © Astronomical Observatory, Ghent University         ////
///////////////////////////////////////////////////////////////// */

#include "GrainSizeDustDestructionRecipe.hpp"
#include "FatalError.hpp"

////////////////////////////////////////////////////////////////////

void GrainSizeDustDestructionRecipe::setupSelfBefore()
{
    DustDestructionRecipe::setupSelfBefore();
}

////////////////////////////////////////////////////////////////////

double GrainSizeDustDestructionRecipe::densityFraction(bool graphite, double a, const Array& /*Jv*/, double T) const
//double GrainSizeDustDestructionRecipe::densityFraction(bool /*graphite*/, double a, double T) const
{
    // from vanVelzen+2016
    //if (graphite){
    //    if ( T>=(Prefactor() * 1900/(std::log(FWHM()/(30*3.21*a/0.1))+Prefactor())) ) return 0.;
    ///    //if (T<sublimationTemperature()) return 1.;
    //    return 1.;
    //
    //from Waxman Draine 2000
    if (graphite){
        if ( T>=(81200 / (std::log( FWHM() * 2.3e9 / (4641.589*a) )) ) ) return 0.;
        return 1.;
    }
    else{
        if ( T>=(68100 / (std::log( FWHM() * 2.3e10 / (4641.589*a) )) ) ) return 0.;
        return 1.;
    }
    //Waxman Draine 2000 with more accurate densityFraction
    //if (graphite){
    //    if ( T>=(81200 / (std::log( FWHM()/(2.3e9) * 8.7e12/a )) ) ) return 0.;
    //    return 1.;
    //}
    //else{
    //    if ( T>=(68100 / (std::log( FWHM()/(2.3e10) * 8.7e12/a )) ) ) return 0.;
    //    return 1.;
    //}
}

////////////////////////////////////////////////////////////////////
