## Wind Turbine Sound Surrogate (TurbSound)

This repository contains code for the model described in 

Gu, J., Glaws, A., Harrison-Atlas, D., Bortolotti, P., Kaliski, K., and Lopez, A. "The impact of sound ordinances on the land-based wind technical potential of the United States." In press at Nature Energy.

___

#### Description

The Wind Turbine Sound Surrogate is a data-driven model for predicting turbine aeroacoustic sound levels based on atmospheric and environmental conditions, including wind speed/direction, turbulence, temperature, humidity, air pressure, and a ground coverage factor. The neural network model is located in the `SoundSurrogate.py` file. The `utils.py` file contains support functionality for data processing and normalization. Additionally, an interface is provided in `TurbSound.py` to easily leverage the surrogate to get sound levels for a timeseries of weather data, to compute various statistical metrics for sound levels, and to compute turbine setback distances based on sound thresholds. We also provide an example notebook and dataset to demonstrate how the tool is used.

A conda environment YML file `environment.yml` has been provided for your convenience. To build this conda environment use the command

`conda env create -f environment.yml`

and then

`conda activate turb-sound.yml`

#### Acknowledgments
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
