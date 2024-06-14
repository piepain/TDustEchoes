# Dust Reprocessing Echo Analiser Module for Tidal disruption events

This radiative transfer simulation is designed to model Dust Echos of Tidal Disruption Events.

This work adds onto van Gaalen's Dust Echo modeling module; see: https://github.com/vgaalen/DustEcho  
This work builds upon SKIRT developed by Astronomical Observatory, Ghent University; see https://skirt.ugent.be/ or https://github.com/SKIRT/SKIRT9.

## Installation Instructions

This software has only been tested on Unix based systems.  
Installation on another OS is possible, but not supported by these instructions.  
Requirements:

 - A recent C++ compiler with full support for the C++14 standard:
        On macOS, use Apple Clang v10 (included in Xcode v10) or later.
        On other Unix systems, use GNU g++ v5.4.0 or later or Intel icc v19.0.2 or later.
 - CMake v3.2.2 or later.
 - git
 - python3
 - the following python-packages. A venv is included in the repo (see requirements.txt)
    - astropy
    - scipy
    - matplotlib
    - datetime
    - tqdm
    - reportlab
    - lxml
    - jupyter notebook / jupyterlab
    - json

#### Get the source code

Clone the github repository, build SKIRT's C++ code, and retrieve SKIRT's resources (these contain files too large for github and are therefore fetched seperately)

```
git clone https://github.com/piepain/TDustEchoes [YOURDIR]
cd [YOURDIR]/SKIRT/git
chmod +rx makeSKIRT.sh
chmod +rx downloadResources.sh
./makeSKIRT.sh
./downloadResources.sh
```

The downloadResources.sh executable prompts you to download different resource packages, only SKIRT9_Resources_Core is required.

#### Trying out the simulation

This repository includes a notebook to analyse dust echoes and show the simulation's capabilities.  
DustEchoes.ipynb shows how to generate a simulated lightcurve for a source, and shows how to make simulations for multiple values for several variables.  
This can easily be recreated for any other source by taking the UV light curves from van Velzen's repository (https://github.com/sjoertvv/manyTDE), its distance, and its temperature. By adding the infrared data as well, the simulated dust echoes can also be compared to measurements.
data_analysis.ipynb is a class that performs some analysis functions, and can also create figures. This is still fairly bare-bones though, so it is probably easier to use the DustEchoes notebook.
