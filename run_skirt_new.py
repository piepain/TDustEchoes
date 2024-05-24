import os
import glob
import astropy.units as u
from astropy.io import fits
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad
from tqdm import tqdm

import sys
sys.path.append('./PTS')
import pts.simulation as sm
import pts.do
pts.do.initializePTS()

def writeSKI(infile,outfile,static,normalization_type,nsizes,Si):
    """ 
     Removes and adds parts of the .SKI file based on the type of normalisation and the number of layers.
     Make sure that in the SKI file, you have both options for the geometric medium, enclosed between <!--sublimation_start--> and <!--sublimation_end--> and <!--static_start--> and <!--static_end-->.
     The SKI file should contain both mass and optical depth normalization, enclosed by <!--mass and mass--> and <!--optd and optd--> respectively.
     The different dust types (silicates and graphites) should be enclosed between <!--Si_start--> and end, or <!--C_start--> and end.

     Parameters
     ----------
     infile : string
         The name of the file this function will change.
     outfile : string
         The name of the wanted output file
     static : bool
         Decides if the simulation includes sublimation
     normalization_type : string
         Describes the type of normalization wanted; either 'mass' or 'optical_depth'
     nsizes : int
         The number of different dust grain sizes within the dust structure.
     Si : float
         The fraction of the dust that should consist of silicates
                """
    with open(infile) as fin:
        content = fin.readlines()

    #Finding the places where layerstart and layerend are located for the sublimation and static cases
    start_sub = next((i for i, line in enumerate(content) if '<!--sublimation_start-->' in line), None)
    end_sub = next((i for i, line in enumerate(content) if '<!--sublimation_end-->' in line), None)

    if not static:
        #Finding and deleting the static part
        starts = [i for i, line in enumerate(content) if '<!--static_start-->' in line]
        ends = [i for i, line in enumerate(content) if '<!--static_end-->' in line]
        for start_index, end_index in zip(starts[::-1], ends[::-1]):
                del content[start_index + 1:end_index + 1]

        start_C = [i for i, line in enumerate(content) if '<!--C_start-->' in line]
        end_C =[ i for i, line in enumerate(content) if '<!--C_end-->' in line]
        start_Si = [i for i, line in enumerate(content) if '<!--Si_start-->' in line]
        end_Si = [i for i, line in enumerate(content) if '<!--Si_end-->' in line]
        
        if Si==0:
            #Only graphite present, so delete the silicates part
            for start_index, end_index in zip(start_Si[::-1], end_Si[::-1]):
                del content[start_index + 1:end_index + 1]
            layer = content[start_sub+1:end_sub+1]
            duplicate_layer = layer * nsizes
            content[start_sub+1:end_sub+1] = duplicate_layer

        elif Si==1:
            #Only silicates present, so delete the graphites part
            for start_index, end_index in zip(start_C[::-1], end_C[::-1]):
                del content[start_index + 1:end_index + 1]
            layer = content[start_sub+1:end_sub+1]
            duplicate_layer = layer * nsizes
            content[start_sub+1:end_sub+1] = duplicate_layer

        else:
            #Both Si and C are present, but copy them one by one to make array shapes and orders in get_lightcurve easier
            #Start with Si as that is the one furthest down, so the line numbers won´t get mixed up
            for start_index, end_index in zip(start_Si[::-1], end_Si[::-1]):
                layer_Si = content[start_index + 1:end_index + 1]
                duplicate_layer_Si = layer_Si * nsizes
                content[start_index + 1:end_index + 1] = duplicate_layer_Si

            for start_index, end_index in zip(start_C[::-1], end_C[::-1]):
                layer_C = content[start_index + 1:end_index + 1]
                duplicate_layer_C = layer_C * nsizes
                content[start_index + 1:end_index + 1] = duplicate_layer_C


    if static:
        #Finding and deleting the sublimation part
        starts = [i for i, line in enumerate(content) if '<!--sublimation_start-->' in line]
        ends = [i for i, line in enumerate(content) if '<!--sublimation_end-->' in line]
        for start_index, end_index in zip(starts[::-1], ends[::-1]):
            del content[start_index + 1:end_index + 1]

        if normalization_type == "mass":
            #Finding and deleting the optical depth normalization part
            starts = [i for i, line in enumerate(content) if '<!--optd' in line]
            ends = [i for i, line in enumerate(content) if 'optd-->' in line]
            for start_index, end_index in zip(starts[::-1], ends[::-1]):
                del content[start_index + 1:end_index + 1]

        elif normalization_type == "optical_depth":
            #Finding and deleting the mass normalization part
            starts = [i for i, line in enumerate(content) if '<!--mass' in line]
            ends = [i for i, line in enumerate(content) if 'mass-->' in line]
            for start_index, end_index in zip(starts[::-1], ends[::-1]):
                del content[start_index + 1:end_index + 1]

        else:
            raise AttributeError("The normalization type should either be 'mass' or 'optical_depth'.")

        #Deleting the grains if there are none of that type
        start_C = [i for i, line in enumerate(content) if '<!--C_start-->' in line]
        end_C =[ i for i, line in enumerate(content) if '<!--C_end-->' in line]
        start_Si = [i for i, line in enumerate(content) if '<!--Si_start-->' in line]
        end_Si = [i for i, line in enumerate(content) if '<!--Si_end-->' in line]

        if Si ==0:
            for start_index, end_index in zip(start_Si[::-1], end_Si[::-1]):
                del content[start_index + 1:end_index + 1]
        elif Si==1:
            for start_index, end_index in zip(start_C[::-1], end_C[::-1]):
                del content[start_index + 1:end_index + 1]
        
        #Change the powerlaw grain size distribution to a single grain size if needed
        if nsizes == 1: 
            for i, line in enumerate(content):
                if '<PowerLawGrainSizeDistribution' in line:
                    indent = line[:len(line) - len(line.lstrip())]  #Make sure to keep the indentation level
                    content[i] = f'{indent}<SingleGrainSizeDistribution size="1 micron"/>\n' #Grain size will be changed in get_lightcurve

    #Deleting all comments from the skifile
    delete_list = ["<!--mass", "mass-->","<!--optd", "optd-->","<!--sublimation_start-->","<!--sublimation_end-->","<!--static_start-->","<!--static_end-->","<!--Si_start-->","<!--Si_end-->","<!--C_start-->","<!--C_end-->"]
    #Making a new file containing the edited text
    with open(outfile, "w") as fout:
        for line in content:
            for word in delete_list:
                line = line.replace(word, "")
            fout.write(line)

def wavelengthListtoString(wavelengths):
    """
     Changes the numpy array or list containing all output wavelengths to a string which works in SKIRT.
     Parameters
    ----------
     wavelengths : array or list
         Array or list containing all the wavelengths you want to put into SKIRT.
    """
    list= [] # string list for in the ski file
   
    for i in range(len(wavelengths)):
        if i==0:
            list += f'{wavelengths[i]} micron,'
        elif i < len(wavelengths) -1:
            list += f' {wavelengths[i]} micron,'
        else:
            list += f' {wavelengths[i]} micron'
    return ''.join(list)

def get_lightcurve(L_data,T_data,t_data,output_t,output_wavelengths,shell,grain,spaceBins,normalization_type,total_mass=None,opt_depth=None,nsizes=1,static=False,FWHM=None,distance=1171*u.Mpc,skiname="combined_ski.txt",Si=False,prefix='v6',
                   OUTFILES='results/',SKIRTpath=None,suffix="",nphotons=1e5):
    """
     This function uses the SKIRT program (https://skirt.ugent.be/) to simulate a lightcurve for a variable blackbody source with a dust geometry around it. 
     The source is defined by the blackbody temperature (T) and the integrated luminosity. 
     Both these parameters have to be defined in every timestep for which the lightcurve should be determined though a SKIRT simulation. 
     The full lightcurve is determined afterwards through interpolation to all points in output_t.

     Parameters
     ----------
     L_data : array
         Array containing the integrated luminosity of the source in erg/s for every timestep.
     T_data : array
         Array containing the blackbody temperature of the source in K for every timestep.
     t_data : array
         Array containing the time of every timestep in MJD.
     output_t : array
         Array containing the times for which the lightcurve should be determined in MJD (through interpolation).
     output_wavelengths : array
         Array describing which wavelengths should be in the output and their relative half bin widths; [wavelength 1, wavelength 2, ..., wavelength n, relative half bin width].
     shell : array
         Array containing the parameters of the dust geometry; [inner radius, outer radius, exponent].
     grain : array
         Array containing 2 arrays, each containing the minimum and maximum dust grain size and the dust grain size distribution exponent for multiple layers, one for graphite and one for silicates. [[min size graphites, max size graphites, exponent graphites], [min size silicates, max size silicates, exponent silicates]]
         If there is only 1 dust size, the minimum grain size will be used.
     spaceBins : int
         The number of spatial bins used in the SKIRT simulation.
     normalization_type : string
         The normalization type SKIRT should use. Either 'mass' or 'optical_depth'.     
     total_mass : float
         The total mass of the dust in solar mass.
     opt_depth : array
         Array containing the optical depth (float) and the wavelength in micron (float) and axis (string) at which it should be taken; [optical depth, wavelength, axis].
     nsizes : int
         Integer specifying the number of dust grain sizes within the dust structure. One size by default.
     static : bool
         When True this parameter prevents any sublimation (or actually prevent non-physical regrowth of dust when the transient falls off)
         When False, this function does not support different grain sizes and distributions for silicates and graphites, and only works on mass normalization.
     FWHM : float
         The full width at half maximum of the transients lightcurve in days. Only necessary if static==False.
     distance : float, optional
         The distance to the transient in Mpc. If not given, the distance is set to 1171 Mpc.
     skiname : str, optional
         The name of the SKIRT config file (as a .txt file) that should be used. Default is 'combined_ski.txt'.
     Si : float, optional
         The fraction of the dust mass that is made up of silicate grains, the rest is made up of carbonaceous grains. Default is 0, which means that only carbonaceous grains are used.
     prefix : str, optional
         The prefix of the output files. Default is 'v6'.
     OUTFILES : str, optional
         The path to the output directory. Default is 'results/'.
     SKIRTpath : str, optional
         The path to the SKIRT executable. Default is None, which means that the default path is used.
     suffix : str, optional
         A suffix to the output file names (eg _1), to eliminate any problems with file rights in shared folders.
     nphotons : int, optional
         The number of photon packets simulated through SKIRT. Default is 1e5 for fast simulations, for better resolution (or more) 1e7 is recommended.
     """
    
    #Handling cases that will not work, inside an if statement so they can easily be collapsed
    if True:
        if opt_depth == None and total_mass == None:
            raise Exception("You must give a value for either the mass or the optical depth, or both. Now received values for neither.")
        if normalization_type=='mass' and total_mass == None:
            raise Exception("Please give a total mass if you want to use mass normalization. Change normalisation method to 'optical_depth' or add a total mass for the shell.")
        if normalization_type=='optical_depth' and opt_depth==None:
            raise Exception("Please give optical depth parameters if you want to use optical depth normalization. Change normalisation method to 'mass' or add the optical depth parameters.")

        if nsizes<1:
            raise Exception("The number of different grain sizes cannot be smaller than 1 (single layer).")
        if nsizes>1 and len(grain[0])<3 or len(grain[1])<3:
            raise Exception("Expected 3 values in both grain arrays. If you want to use a single grain size, please put in a placeholder for the maximum grain size and the size distribution exponent; these won't be used.")

        if not static and FWHM==None:
            raise Exception("If you want the simulation to not be static, please insert a value for the FWHM of the primary source.")
        if static==False and normalization_type!='mass':
            raise Exception("Non-static simulations with optical depth normalization are currently not supported. This can be circumvented by doing a single run at the desired optical depth and using the mass from one of the convergence probes.")

        if (len(grain)!=2 or type(grain[1])==int):
            raise Exception("The grain array should look like this: [[min size graphites, max size graphites, exponent graphites], [min size silicates, max size silicates, exponent silicates]] ")
        if Si<0 or Si>1:
            raise Exception("The silicate fraction should be a float in the range [0,1]")
    
    #Letting the user know what simulation the program is running
    if static:
        _='without sublimation'
    elif not static:
        _='with sublimation'
    print(f"Now simulating the lightcurve for a sphere with {nsizes} dust grain sizes, {normalization_type.replace("_"," ")} normalization, and {_}. ")

    #Creating the .SKI file to from the input .txt file to prepare for simulation
    writeSKI(skiname, "cleaned_ski"+str(suffix)+".ski",static, normalization_type, nsizes, Si)
    skifile = sm.SkiFile("cleaned_ski"+str(suffix)+".ski")

    #Set the number of photon packets
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation[@numPackets]','numPackets',str(nphotons))

    #Set up the spatial grid
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/grid/Sphere1DSpatialGrid/meshRadial/LogMesh[@numBins]','numBins',str(spaceBins))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/grid/Sphere1DSpatialGrid/meshRadial/LogMesh[@centralBinFraction]','centralBinFraction',str(shell[0]/shell[1]))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/grid/Sphere1DSpatialGrid[@maxRadius]','maxRadius',str(shell[1]))

    #Set up dust emission options
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/dustEmissionOptions/DustEmissionOptions/dustEmissionWLG/LogWavelengthGrid[@minWavelength]','minWavelength',str(np.min(output_wavelengths)*(1-output_wavelengths[-1])))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/dustEmissionOptions/DustEmissionOptions/dustEmissionWLG/LogWavelengthGrid[@maxWavelength]','maxWavelength',str(np.max(output_wavelengths)*(1+output_wavelengths[-1])))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/dustEmissionOptions/DustEmissionOptions/dustEmissionWLG/LogWavelengthGrid[@numWavelengths]','numWavelengths',str(2*len(output_wavelengths[:-1])))

    #Set the output wavelengths:
    wls = wavelengthListtoString(output_wavelengths[:-1])
    relbinwidth = output_wavelengths[-1]
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/instrumentSystem/InstrumentSystem/instruments/SEDInstrument/wavelengthGrid/ListWavelengthGrid[@wavelengths]','wavelengths','{0}'.format(wls))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/instrumentSystem/InstrumentSystem/instruments/SEDInstrument/wavelengthGrid/ListWavelengthGrid[@relativeHalfWidth]','relativeHalfWidth','{0}'.format(relbinwidth))

    #Set the distance
    if distance==1171*u.Mpc:
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/cosmology/FlatUniverseCosmology[@redshift]','redshift','0.2343272')
    else:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.cosmology.units as cu
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/cosmology/FlatUniverseCosmology[@redshift]','redshift',str(distance.to(cu.redshift,cu.redshift_distance(FlatLambdaCDM(70*u.km/u.s/u.Mpc, 0.3,0.7),kind='luminosity')).value))

    #The static case
    if static:
        #Set up the (lack of) dust destruction
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator[@hasDynamicDensities]','hasDynamicDensities',"false")
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/dynamicStateOptions/DynamicStateOptions/recipes/GrainSizeDustDestructionRecipe[@FWHM]','FWHM',"{0}".format(1000))

        #Set up the dust grain properties for the case of only graphite values in the grain array and the case where there are also silicates.
        if Si==0:
            if nsizes==1:
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/SingleGrainSizeDistribution[@size]','size','{0} micron'.format(grain[0][0]))
            else:
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@minSize]','minSize','{0} micron'.format(grain[0][0]))
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@maxSize]','maxSize','{0} micron'.format(grain[0][1]))
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@exponent]','exponent',str(grain[0][2]))
                #Set the number of dust grain size bins
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[1][@numSizes]','numSizes',str(nsizes))
        elif Si==1:
            if nsizes ==1:
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/SingleGrainSizeDistribution[@size]','size','{0} micron'.format(grain[1][0]))
            else:
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@minSize]','minSize','{0} micron'.format(grain[1][0]))
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@maxSize]','maxSize','{0} micron'.format(grain[1][1]))
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/PowerLawGrainSizeDistribution[@exponent]','exponent',str(grain[1][2]))
                #Set the number of dust grain size bins
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[1][@numSizes]','numSizes',str(nsizes))
        else:
            if nsizes==1:
                for i in range(len(grain)):
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[{0}]/sizeDistribution/SingleGrainSizeDistribution[@size]'.format(i+1),'size','{0} micron'.format(grain[i][0]))
            else:
                #Set up the dust grain properties for graphites (i=0) and silicates (i=1)
                for i in range(len(grain)):
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[{0}]/sizeDistribution/PowerLawGrainSizeDistribution[@minSize]'.format(i+1),'minSize','{0} micron'.format(grain[i][0]))
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[{0}]/sizeDistribution/PowerLawGrainSizeDistribution[@maxSize]'.format(i+1),'maxSize','{0} micron'.format(grain[i][1]))
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[{0}]/sizeDistribution/PowerLawGrainSizeDistribution[@exponent]'.format(i+1),'exponent',str(grain[i][2]))
                    #Set the number of dust grain size bins
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[{0}][@numSizes]'.format(i+1),'numSizes',str(nsizes))
            #Set up the silicate-graphite ratio
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[1][@dustMassPerHydrogenMass]','dustMassPerHydrogenMass',str((1-Si)*0.01)) #For graphite. *0.01 as this is the standard value
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation[2][@dustMassPerHydrogenMass]','dustMassPerHydrogenMass',str(format((Si)*0.01))) #For silicates

        #Set up the dust structure properties
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/geometry/ShellGeometry[@exponent]','exponent',str(shell[2]))
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/geometry/ShellGeometry[@minRadius]','minRadius','{0} pc'.format(shell[0]))
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/geometry/ShellGeometry[@maxRadius]','maxRadius','{0} pc'.format(shell[1]))

        if normalization_type=='mass':
            mass = total_mass
            #Set up the normalization properties
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/normalization/MassMaterialNormalization[@mass]','mass','{0} Msun'.format(mass))

        if normalization_type=='optical_depth':
            #Set up the normalization properties
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/normalization/OpticalDepthMaterialNormalization[@opticalDepth]','opticalDepth',str(opt_depth[0]))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/normalization/OpticalDepthMaterialNormalization[@wavelength]','wavelength','{0} micron'.format(str(opt_depth[1])))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium/normalization/OpticalDepthMaterialNormalization[@axis]','axis','{0}'.format(str(opt_depth[2])))
        radius = shell[0]

    #The sublimation case
    if not static:
        #Set up the dust destruction
        if Si>0 and Si<1:
            for i in range(2*nsizes):
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/materialMix/FragmentDustMixDecorator[@hasDynamicDensities]'.format(i+1),'hasDynamicDensities',"true")
        else:
            for i in range(nsizes):
                skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/materialMix/FragmentDustMixDecorator[@hasDynamicDensities]'.format(i+1),'hasDynamicDensities',"true")
    
        skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/dynamicStateOptions/DynamicStateOptions/recipes/GrainSizeDustDestructionRecipe[@FWHM]','FWHM',"{0}".format(FWHM))
        """
        The following might work for different grain sizes & size distributions for silicates and graphites, but has not been tested or completely finished.

        #Determining the mass for each dust grain size bin
        #With silicates
        if len(grain)==2:
            a_c=np.logspace(np.log10(grain[0][0]),np.log10(grain[0][1]),10) #For the graphites
            mass_c=[np.abs(quad(lambda x: x**(-1*grain[0][2]),a_c[0],10*a_c[0])[0])+np.abs(quad(lambda x: x**(-1*grain[0][2]),10*a[0],np.inf)[0])]
            a_si=np.logspace(np.log10(grain[1][0]),np.log10(grain[1][1]),10) #For the silicates           
            mass_si=[np.abs(quad(lambda x: x**(-1*grain[1][2]),a_si[0],10*a_si[0])[0])+np.abs(quad(lambda x: x**(-1*grain[1][2]),10*a[0],np.inf)[0])]
            for i in range(1,len(a_c)):
                mass_c.append(quad(lambda x: x**(-1*grain[0][2]),a_c[i],a_c[i-1])[0])
                mass_si.append(quad(lambda x: x**(-1*grain[1][2]),a_si[i],a_si[i-1])[0])
        """      
        #Determining the mass for each dust grain size bin
        if Si==0:
            a=np.logspace(np.log10(grain[0][1]),np.log10(grain[0][0]),nsizes)
            mass=[np.abs(quad(lambda x: x**(-1*grain[0][2]),a[0],10*a[0])[0])+np.abs(quad(lambda x: x**(-1*grain[0][2]),10*a[0],np.inf)[0])]
            for i in range(1,len(a)):
                mass.append(quad(lambda x: x**(-1*grain[0][2]),a[i],a[i-1])[0])
        elif Si==1:
            a=np.logspace(np.log10(grain[1][1]),np.log10(grain[1][0]),nsizes)
            mass=[np.abs(quad(lambda x: x**(-1*grain[1][2]),a[0],10*a[0])[0])+np.abs(quad(lambda x: x**(-1*grain[1][2]),10*a[0],np.inf)[0])]
            for i in range(1,len(a)):
                mass.append(quad(lambda x: x**(-1*grain[1][2]),a[i],a[i-1])[0])
        else:
            a=np.logspace(np.log10(grain[0][1]),np.log10(grain[0][0]),nsizes)
            mass=[np.abs(quad(lambda x: x**(-1*grain[0][2]),a[0],10*a[0])[0])+np.abs(quad(lambda x: x**(-1*grain[0][2]),10*a[0],np.inf)[0])]
            for i in range(1,len(a)):
                mass.append(quad(lambda x: x**(-1*grain[0][2]),a[i],a[i-1])[0])
            mass=np.concatenate((np.multiply((1-Si),mass),np.multiply(Si,mass)))
            a=np.concatenate((a,a))

        mass=np.multiply(mass,total_mass/np.sum(mass))

        for i in range(len(mass)):
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/normalization/MassMaterialNormalization[@mass]'.format(i+1),'mass','{0} Msun'.format(mass[i]))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/materialMix/FragmentDustMixDecorator/dustMix/ConfigurableDustMix/populations/GrainPopulation/sizeDistribution/SingleGrainSizeDistribution[@size]'.format(i+1),'size','{0} micron'.format(a[i]))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/geometry/ShellGeometry[@minRadius]'.format(i+1),'minRadius','{0} pc'.format(shell[0]))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/geometry/ShellGeometry[@maxRadius]'.format(i+1),'maxRadius','{0} pc'.format(shell[1]))
            skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/geometry/ShellGeometry[@exponent]'.format(i+1),'exponent',str(shell[2]))
        radius=[shell[0]]*len(mass)   

    #Run SKIRT
    #Note: the lightcurve file is based on relative times, therefore the zeropoint corresponds to direct light.
    lightcurve=[]
    temp=[]
    t_peak=t_data[np.argmax(L_data[1])]
    
    for t in tqdm(range(len(t_data)),desc="SKIRT Runs"):
        # Run SKIRT
        lightcurve_t,SED_t,radius_t,temp_t,wavelengths,simulation=runSKIRT(L_data[1,t],T_data[1,t],t_data[t],skifile,OUTFILES=OUTFILES,SKIRTpath=SKIRTpath)
        # Save the output
        if np.sum(lightcurve_t)==0.:
            return 0,0,0,0, simulation
        lightcurve.append(lightcurve_t)
        temp.append(temp_t)
        # Update the dust grain size distribution if applicable
        if static==False and normalization_type=='mass':
            # If the peak has passed, the dust should no longer sublimate
            if t_data[t]>=t_peak:
                static=True
            # If the peak has not passed, the dust grain size distribution should be updated
            for i in range(len(radius_t)):
                # Only update if the inner radius of a shell has increased (due to finite spatial resolution in SKIRT this is not always the case, even before the transient peak)
                if radius_t[i]<=shell[0]:
                    pass
                elif radius_t[i]<radius[i]:
                    pass
                elif radius[i]<radius_t[i] and radius_t[i]<shell[1]:
                    mass_fraction=quad(lambda r: r**(-1*shell[2]), radius_t[i], shell[1])[0]/quad(lambda r: r**(-1*shell[2]), radius[i], shell[1])[0]
                    if mass_fraction<1.:
                        mass=np.multiply(mass_fraction,mass)
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/geometry/ShellGeometry[@minRadius]'.format(i+1),'minRadius','{0} pc'.format(radius_t[i]))
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/normalization/MassMaterialNormalization[@mass]'.format(i+1),'mass','{0} Msun'.format(mass[i]))
                    radius[i]=radius_t[i]
                # If the inner radius of a shell has increased beyond the outer radius, the shell should be removed (as true removal is challanging in this implementation, the shell is set to a very small radius and mass)
                else:
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/geometry/ShellGeometry[@minRadius]'.format(i+1),'minRadius','{0} pc'.format(shell[1]-1e-5))
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/materialMix/FragmentDustMixDecorator[@initialDensityFraction]'.format(i+1),'initialDensityFraction','0')
                    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/mediumSystem/MediumSystem/media/GeometricMedium[{0}]/normalization/MassMaterialNormalization[@mass]'.format(i+1),'mass','1e-50 Msun')
                    radius[i]=radius_t[i]
    # Create an array with the timesteps as written by SKIRT (changes to the SKIRT config file should be mirrored here)
    t_bin=1*u.day
    timesteps=[]
    for i in np.arange(len(lightcurve[0][0,1:])):
        timesteps.append(i*t_bin.value)

    # Interpolate between timesteps to get emmission for every day
    representation=np.zeros((len(lightcurve[0][0,:]),len(wavelengths),3,len(t_data)+4))

    for wl in tqdm(range(len(wavelengths)),leave=True):
        for i in range(1,len(lightcurve[0][0,:])-1): # first column contains wavelengths and final contains overflow, therefore these are discarded
            y=[]
            time=[]
            for t in range(len(t_data)):
                if lightcurve[t][wl,i]!=0.:
                    y.append(lightcurve[t][wl,i])
                    time.append(t_data[t])
            if np.linalg.norm(y)==0.:
                pass
            if len(y)>3:
                a=splrep(time,y,k=1)
            elif len(y)>1:
                a=splrep(time,y,k=1)
            else:
                continue
            representation[i,wl,0,:len(a[0])]=a[0]
            representation[i,wl,1,:len(a[1])]=a[1]
            representation[i,wl,2,0]=a[2]

    lc_total=np.zeros((len(output_t),len(wavelengths)))
    for i in tqdm(range(len(output_t)),desc="Compile Lightcurve",leave=False,position=2):
        for t in np.arange(0,int(np.floor(output_t[i]-t_data[0])),1):
            for wl in range(len(wavelengths)):
                if np.linalg.norm(representation[int(t/1)+1,wl,0,:])!=0:
                    lc_total[i,wl]+=splev(output_t[i]-t,(representation[int(t/1)+1,wl,0,:],representation[int(t/1)+1,wl,1,:],int(representation[int(t/1)+1,wl,2,0])))

    return lc_total*u.Jy,wavelengths,temp,radius, simulation


def runSKIRT(L,T,MJD,skifile,OUTFILES="",SKIRTpath=None):
    """
    This function uses the SKIRT program (https://skirt.ugent.be/) to simulate one timestep in a lightcurve for a variable blackbody source with a dust geometry around it. 
    The source is defined by the blackbody temperature (T) and the integrated luminosity. 

    Parameters
    ----------
    L : float
        The integrated luminosity of the source in erg/s.
    T : float
        The blackbody temperature of the source in K.
    MJD : float
        The absolute time of the requested lightcurve in MJD.
    skifile : str
        The SKIRT config file that should be used.
    OUTFILES : str, optional
        The path to the output directory. Default is ''.
    SKIRTpath : str, optional
        The path to the SKIRT executable. Default is None, which means that the default path is used.
    """

    # Initialize skirt
    if SKIRTpath==None:
        skirt = sm.Skirt(path="SKIRT/release/SKIRT/main/skirt")
    else:
        skirt = sm.Skirt(path=SKIRTpath)

    # Update the SKIRT config file with this timestep's parameters
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/sourceSystem/SourceSystem/sources/PointSource/normalization/IntegratedLuminosityNormalization[@integratedLuminosity]','integratedLuminosity',"{0} erg/s".format(L))
    skifile.setStringAttribute('//skirt-simulation-hierarchy/MonteCarloSimulation/sourceSystem/SourceSystem/sources/PointSource/sed/BlackBodySED[@temperature]','temperature',"{0:.2E} K".format(T))

    # Create the output directory and clear it if necessary
    if os.path.isdir(OUTFILES+str(int(MJD))+"/")==False:
        os.mkdir(OUTFILES+str(int(MJD))+"/")
    else:
        files = glob.glob(OUTFILES+str(int(MJD))+'/*')
        for f in files:
            os.remove(f)

    # Run SKIRT
    skifile.saveTo(OUTFILES+str(int(MJD))+"/run.ski")
    simulation = skirt.execute(OUTFILES+str(int(MJD))+"/run.ski",outDirPath=OUTFILES+str(int(MJD))+"/", console='brief')

    if os.path.isfile(OUTFILES+str(int(MJD))+'/run_instrument1_sed.dat')==False or os.path.isfile(OUTFILES+str(int(MJD))+'/run_instrument1_lc.dat')==False:
        print("ERROR: SKIRT did not produce output files")
        return 0,0,0,0,0, simulation
    # Load the output
    SED=np.loadtxt(OUTFILES+str(int(MJD))+'/run_instrument1_sed.dat')
    lightcurve=np.loadtxt(OUTFILES+str(int(MJD))+'/run_instrument1_lc.dat')

    # Find the inner radius of every dust shell and the temperature at that radius
    temp=[]
    radius=[]
    i=0
    while os.path.isfile(OUTFILES+str(int(MJD))+'/run_medium-temperature_{0}_T_xy.fits'.format(i)):
        datafile=fits.open(OUTFILES+str(int(MJD))+'/run_medium-temperature_{0}_T_xy.fits'.format(i))[0]
        temperature=datafile.data
        size=int(np.ceil(0.5*len(temperature[0,:])))
        # Handle the no sublimation case
        if temperature[size,size]!=0.:
            radius.append(0)
            temp.append(temperature[size,size])
            i+=1
        # Handle the full sublimation case
        elif np.linalg.norm(temperature)==0.:
            radius.append(size*datafile.header['CDELT1'])
            temp.append(0.)
            i+=1
        # All other cases
        else:
            for r in range(size):
                if temperature[size,size+r]!=0.:
                    radius.append(r*datafile.header['CDELT1'])
                    temp.append(temperature[size,size+r])
                    i+=1
                    break
    
    wavelengths=SED[:,0]
    return lightcurve,SED,radius,temp,wavelengths,simulation


