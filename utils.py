""" 
Original by Remon van Gaalen, https://github.com/vgaalen/DustEcho
Edited and expanded upon by Pepijn Lankhorst & Leon Schoof, https://github.com/piepain/TDustEchoes
"""

import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.io import ascii

from scipy.interpolate import interp1d

import json

import matplotlib.colors as colors


unitL=u.erg/u.s/u.Hz
unitF=u.erg/u.s/u.Hz/(u.m**2)

def bb_fit(wave,T ,A):
    """
    Fitting the parameters of a blackbody model.

    Parameters
    ----------
    wave : array_like
        Wavelengths input in micrometers.
    T : float
        Temperature in Kelvin.
    A : float
        Normalization constant.

    Returns
    -------
    array_like
        Intensity values of the blackbody model.
    """
    T = T * u.K
    nu = (c.c/(wave * u.um)).to(u.Hz)
    h = (c.h).to(u.erg * u.s)
    k_b = c.k_B.to(u.erg / u.K)
    wave = wave * u.um
    intensity = A / ((wave**5).value * np.exp(h*c.c.to(u.um/u.s)/(wave*k_b*T))-1)    
    
    return intensity.value

def wien_inverse(wave_max):
    """
    Calculating the temperature for a given maxima of a blackbody 
    wave_max: float in micrometer, has to be void of unit. 
    
    Parameters
    ----------
    wave_max : float
        The wavelength of maximum emission in micrometers. The value should be unitless.

    Returns
    -------
    astropy.units.quantity.Quantity
        The temperature of the blackbody in Kelvin.

    """
    wave_max = ((wave_max * u.um).to(u.m))
    return  (2.898 *10**-3 *u.m*u.K)/ wave_max 

def ABmagtoFlux(ABmag,error):
    """
    Convert AB magnitude to flux in erg/s/cm^2/Hz

    Parameters
    ----------
    ABmag : float
        AB magnitude
    error : float
        error on the AB magnitude

    Returns
    -------
    float
        The flux corresponding to the AB magnitude.
    
    """
    return np.array((10**((ABmag+48.60)/-2.5),-3.34407e-20*np.exp(-0.921034*ABmag)*error))*u.erg/(u.s*u.Hz*u.cm**2)

def WISEmagtoFlux(WISEmag,error,band):
    """
    The WISE dataproducts give a magnitude with a zeropoint based on Vega.
    This can be converted using a simple offset.
    
    Parameters
    ----------
    WISEmag : float
        WISE magnitude
    error : float
        error on the WISE magnitude
    band : int
        WISE filterband (1,2,3 or 4)
    
    Returns
    -------
    float
        The WISE magnitude corresponding to the input flux.
    """
    if type(band)!=int or band==0:
        raise TypeError('Please give the WISE filterband as an integer')
    offset=[0,2.699,3.339,5.174,6.620]
    ABmag=WISEmag+offset[band]
    return ABmagtoFlux(ABmag,error)

def binning(bins,time,data,error,clean=True):
    """
    Bins the data in the given bins and returns the binned data

    input the dataset as seperate lists/1D-numpy arrays
    give the bins as a list of border values

    Parameters
    ----------
    bins : list
        list of border values of the bins
    time : list
        list of time values 
    data : list
        list of data values
    error : list
        list of error values
    clean : bool
        if True, the data will be cleaned from outliers before binning
    
    Returns
    -------
    numpy.ndarray
        Array containing the binned time, the binned data and the binned error, as [binned_time,binned_data,binned_error].

    """
    binned_time=[]
    binned_data=[]
    binned_error=[]
    mask=np.invert(np.isnan(data)+np.isnan(error))
    time=time[mask]
    data=data[mask]
    error=error[mask]

    for i in range(len(bins)-1):
        mask=(time>=bins[i])&(time<=bins[i+1])
        if clean:
            median=np.mean(data[mask])
            std=np.std(data[mask])
            #std=np.max(error[mask])
            mask*=(data>median-3*std)&(data<median+3*std)

        if np.linalg.norm(mask,0)!=0:
            mean=wmean(data[mask],error[mask])
            #binned_time.append(wmean(time[mask],error[mask])[0])
            binned_time.append(np.mean(time[mask]))
            binned_data.append(mean[0])
            binned_error.append(mean[1])
            #if clean:
            #    mask2=(data[mask]>0.1*binned_data[-1])&(data[mask]<10*binned_data[-1])
            #    if np.linalg.norm(mask2,0)!=0:
            #        mean=wmean(data[mask][mask2],error[mask][mask2])
            #        binned_data[-1]=mean[0]
            #        binned_error[-1]=mean[1]
        elif i!=0 and i<len(bins)-2:
            binned_time.append(bins[i])
            mask=(time>=bins[i-1])&(time<=bins[i+2])
            if clean:
                median=np.mean(data)
                std=np.std(data)
                #std=np.max(error)
                mask*=(data>median-std)&(data<median+std)

            mean=wmean(data[mask],error[mask])
            binned_data.append(mean[0])
            binned_error.append(mean[1])
        else:
            binned_time.append(bins[i])
            binned_data.append(0)
            binned_error.append(0)

    return np.array((binned_time,binned_data,binned_error))

def FluxtoLum(flux,distance=1171*u.Mpc):
    """
    Convert flux to luminosity
    
    Parameters
    ----------
    flux : astropy.units.quantity.Quantity
        flux
    distance : astropy.units.quantity.Quantity
        distance

    Returns
    -------
    astropy.units.quantity.Quantity
        The luminosity of the object.

    """
    return (4*np.pi*distance**2*flux).to(unitL)

def LumtoFlux(lum,distance=1171*u.Mpc):
    """
    Convert luminosity to flux
    
    Parameters
    ----------
    lum : astropy.units.quantity.Quantity
        luminosity
    distance : astropy.units.quantity.Quantity
        distance
        
    Returns
    -------
    astropy.units.quantity.Quantity
        The flux of the object.

    """
    return (lum/(4*np.pi*distance**2)).to(unitF)

def bb_temp(flux,wl,Tmin=1e-1,Tmax=100000,Tstep=1):
    """
    Returns the temperature of a blackbody with the same relative flux in the 2 bands given as inputs.

    Parameters
    ----------
    flux : tuple
        A tuple containing the fluxes in the two bands.
    wl : tuple
        A tuple containing the wavelengths corresponding to the two fluxes.
    Tmin : float, optional
        The minimum temperature to consider (default is 0.1 K).
    Tmax : float, optional
        The maximum temperature to consider (default is 100,000 K).
    Tstep : float, optional
        The step size for the temperature range (default is 1 K).

    Returns
    -------
    float
        The temperature of the blackbody that matches the relative fluxes in the given bands.
    """
    flux1,flux2=flux
    wl1,wl2=wl
    from astropy.modeling.physical_models import BlackBody
    x,y=0,0
    #Tmin=1e-1
    #while x<0.01 and y<0.01:
    #    Tmin*=10
    #    #Tmax*=10
    #    bb1=BlackBody(temperature=Tmin*u.K)
    #    x=bb1(wl1)/bb1(bb1.lambda_max)
    #    bb2=BlackBody(temperature=Tmin*u.K)
    #    y=bb2(wl2)/bb2(bb2.lambda_max)
    temp=np.arange(Tmin,Tmax,Tstep)
    BB=BlackBody(temperature=temp*u.K)
    index=np.argmin(np.abs((flux1/flux2)-(BB(wl1)/BB(wl2))))

    return temp[index]

def wmean(values,errors):
    """
    Calculate the weighted mean of a set of values with an uncertainty.

    Parameters
    ----------
    values : array_like
        The set of values.
    errors : array_like
        The uncertainties associated with the values.

    Returns
    -------
    list
        A list containing the weighted mean and its uncertainty, as [weighted_mean, uncertainty].
    """
    mask=np.invert(np.isnan(values)+np.isnan(errors))
    values=values[mask]
    errors=errors[mask]
    if np.linalg.norm(mask,0)==0.:
        return [0.,np.inf]
    if np.sum(np.divide(1,np.power(errors,2)))==0.:
        return [0.,np.inf]
    return np.average(values,weights=np.divide(1,np.power(errors,2))), np.sqrt(np.divide(1,np.sum(np.divide(1,np.power(errors,2)))))

def VegaColor(lum1, lum2,distance= 1171*u.Mpc):
    """
     Returns an object's 1-2 color based on Vega magnitudes from its flux in bands 1 and 2.
     
     Parameters
     ----------
     flux1 : float
        The flux in band 1
     flux2 : float
        The flux in band 2
    """
    
    fzero_w1 = 309.540
    fzero_w2 = 171.787

    flux1 = LumtoFlux(lum1,distance=distance).to(u.Jy)
    flux2 = LumtoFlux(lum2,distance=distance).to(u.Jy)

    m1 = -2.5*np.log(flux1.value/fzero_w1)
    m2 = -2.5*np.log(flux2.value/fzero_w2)

    return m1-m2

def BoltzmannSurface(L,T):
    """
    Returns the radiation surface of an object in square parsec from the Stefan-Boltzmann law.

    Parameters
    ----------
    L : float
        Luminosity in erg/s.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Radiation surface in square parsecs.
    """
    return (L*(u.erg/u.s)/(c.sigma_sb * (T*u.K)**4)).to(u.pc**2).value

def blackbody_Lnu(frequency, T=2e4*u.K, distance=1171*u.Mpc, r_outer=0.15*u.pc):
    """ 
    Returns the nu*L_nu of a black body using the Planck function, in erg / s.

    Parameters
    ----------
    frequency : float or numpy.ndarray
        The frequency in Hertz
    T : astropy.units.quantity.Quantity
        Temperature, including a unit
    distance : astropy.units.quantity.Quantity
        distance to the object, including a unit
    r_outer : astropy.units.quantity.Quantity
        outer radius of the object, including a unit   
        
    Returns
    -------
    float or numpy.ndarray
        The blackbody luminosity for the given frequency or frequencies. Type depends on the type of the frequencies parameter.
    """
    frequency *= u.Hz
    fraction = 2 * c.h * frequency**3 / c.c**2
    exponent = np.expm1((c.h * frequency / (c.k_B * T)).to_value(''))
    blackbody = fraction / exponent
    bb_flux = blackbody.to(u.erg/u.Hz/u.cm**2/u.s) /u.sr * 0.25 * np.pi * (np.arctan((r_outer/distance.to(u.pc))))**2
    bb_flux = bb_flux.to(u.erg/u.Hz/u.cm**2/u.s)
    bb_lum = bb_flux * 4 * np.pi * distance**2
    return (bb_lum.to(u.erg/u.Hz/u.s)).value

def lightcurve_transmission(filter_name, output_wavelengths, luminosity, data_loc='./data/filters'):
    '''
    This function gives the lightcurve adjusted for the transmission curve of the used filter.
    Transmission curve data files from http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=WISE&asttype=
    New filters can easily be added by downloading the curves from this website.
    Parameters
    ----------
    filter_name : str
        The name of the filter. Either 'WISE_W1', or 'WISE_W2'.
    output_wavelengths : numpy.ndarray
        The output wavelengths of the simulation.
    luminosity : numpy.ndarray
        The luminosity array as obtained from SKIRT simulation. Should be given in erg/s/Hz.
    data_loc : str, optional
        The folder in which the transmission curve data can be found. The default is './data/filters'.
    Returns
    -------
    numpy.ndarray
        The lightcurve for the filter, made using its transmission curve
    '''
    #Loading the file
    file = ascii.read(data_loc+'/'+filter_name+'.dat',names=['col1','col2'])
    wavelengths = np.array(file['col1']/1e4) #Angstrom to micrometer
    transmission = np.array(file['col2'])
    
    #Interpolating the wavelengths
    interpolator = interp1d(wavelengths,transmission,bounds_error=False,fill_value=0)
    interp_transmission = interpolator(output_wavelengths)
    #Normalizing the transmission curve after interpolating
    interp_transmission /= np.trapz(interp_transmission,output_wavelengths)
    
    #Finding the adjusted luminosity by multiplying and integrating the luminosity by the lightcurve.
    adjusted_lum = [np.trapz(interp_transmission * luminosity[i].value,output_wavelengths) for i in range(len(luminosity))] 
    return adjusted_lum * u.erg/u.s/u.Hz

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Edits a colourmap by cutting some colors from the beginning and the end.    
    Parameters
    ----------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The colourmap that should be edited
    minval : float, optional
        The fraction of the colourmap to cut from the beginning. The default is 0.0.
    maxval : float, optional
        The fraction of the colourmap to cut from the end. The default is 1.0.
    n : int, optional
        Number of discrete color values in the final colormap. The default is 100.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The edited colormap
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_lightcurve_data(tde_name = 'ASASSN-15lh', datafolder='data/ASASSN-15lh/'):
	"""
    Originally from Mummery, van Velzen et al. 2023 (DOI: 10.1093/mnras/stad3001), with some minor alterations.
	Input: 
		The TDEs name
        The folder in which the data is stored. The default is 'data/ASASSN-15lh/'.
	Returns:
		1. A dictionary with all of the light curve data, labelled by observing band. 
		2. A list of lightcurve filters with available data. 
        3. A list of the frequencies of the lightcurve filters.
        4. The MjD of the UV peak.
	"""

	fname = './{0}{1}.json'.format(datafolder,tde_name)
	tde_data = json.load(open(fname,'r'))# Load data. 

	# These conversion are needed because json doesn't store tuples.
	dt = [tuple(x) for x in tde_data['lightcurve']['dtype']]
	lc_obj = [tuple(x) for x in tde_data['lightcurve']['data']] 

	# Make a recarray. 
	lc_rec = np.array(lc_obj, dtype=dt)
	mjd0 = tde_data['peak_mjd']

	lc_dict = {}
	filters = tde_data['lightcurve']['filters']
	frequency_Hz = tde_data['lightcurve']['frequency_Hz']

	for flt in filters:
		idx = lc_rec['filter']==flt

		flux = lc_rec[idx]['flux_Jy']
		flux_corr = flux / tde_data['extinction']['linear_extinction'][flt]# Correct for extinction. 

		lc_dict[flt] = [lc_rec[idx]['mjd']-mjd0, flux_corr, lc_rec[idx]['e_flux_Jy']]
	return lc_dict, filters, frequency_Hz,mjd0

def data_luminosity(tde_name='ASASSN-15lh',datafolder='data/ASASSN-15lh/',distance=1171*u.Mpc):
    """
    Unpacks the 

    Parameters
    ----------
    tde_name : str, optional
        The name of the transient event. The default is 'ASASSN-15lh'.
    datafolder : str, optional
        The folder containing the data files. Default is 'data/ASASSN-15lh/'.
    distance : astropy.units.quantity.Quantity, optional
        The distance to the object. Default is 1171 Mpc.

    Returns
    -------
    numpy.ndarray
        An array containing the luminosity data with:
        - Column 0: Time
        - Column 1: Luminosity
        - Column 2: Luminosity error
        - Column 3: Frequency
    """
    lc_dict,filters,frequency_Hz,mjd0 = get_lightcurve_data(tde_name=tde_name,datafolder=datafolder)
    L = [[],[],[],[]]    
    for flt in filters:
        t,F,e_F = lc_dict[flt]
        F, e_F = FluxtoLum(F*u.Jy,distance=distance),FluxtoLum(e_F*u.Jy,distance=distance)
        idx = filters.index(flt)
        F,e_F = F*frequency_Hz[idx]*u.Hz, e_F*frequency_Hz[idx]*u.Hz
        L[0]=np.append(L[0],t)
        L[1]=np.append(L[1],F.value)
        L[2]=np.append(L[2],e_F.value)
        L[3]=np.append(L[3],[frequency_Hz[idx]]*len(F))
    return np.array(L)

def gauss(x, p):
    """ 
    A simple Gaussian fit function.

    Parameters
    ----------
    x : array_like
        The input values.
    p : array_like
        An array containing the parameters of the Gaussian function.
        p[0] corresponds to the mean, and p[1] corresponds to the standard deviation.

    Returns
    -------
    array_like
        The values of the Gaussian function evaluated at the given input values.

    """
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

def convolution(rinner_pc,lum_pick,mjd_pick):
    """
     Makes a convolution of a step function with length 2r/c and value c/(2r) with the luminosity data.
     Returns the time and flux of this convolution.
     Parameters
     ----------
     rinner_pc : float
          The outer radius of the dust shell.
     lum_pick: array
          Array containing the luminosity of the original source. 
     mjd_pick: array 
          Time in MjD.
    
    Returns
    -------
    tuple of array_likes
        1. The convolution's times
        2. The convolution's found luminosity
    """
    rinner_days = ((rinner_pc*u.pc / c.c ).to(u.day)).value
    
    transfunc_time = np.arange(0,2*rinner_days+100, 0.5)
    transfunc_amp = np.zeros(len(transfunc_time))
    transfunc_amp[transfunc_time<2*rinner_days]= 1/ (2*rinner_days)

    # padding with zeros
    lum_pick = np.append(0, lum_pick)
    lum_pick = np.append(lum_pick, 0)

    mjd_pick = np.append(mjd_pick[0]-1,mjd_pick)
    mjd_pick = np.append(mjd_pick, mjd_pick[-1]+1)
    
    # define output times for convolved function
    lc_conv_time = np.arange(mjd_pick[0]-10,mjd_pick[-1]+2*rinner_days+10, 2)  # grid with 2 day steps
    lc_conv_flux = np.zeros(len(lc_conv_time))

    # loop and find the flux of the light curve that contributes 
    for i, ctime in enumerate(lc_conv_time):
        flux_before = np.interp(ctime-transfunc_time, mjd_pick, lum_pick) # we look "back in time"
        lc_conv_flux[i] = np.trapz(flux_before*transfunc_amp, transfunc_time) # the tranfer function is normalized
    return lc_conv_time, lc_conv_flux