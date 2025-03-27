# DX11

## Overview
This repository contains an adaptation of the **DX11 algorithm** for **weekly time series**, based on Keerthi's detrending approach from the study:

> **Contrasted Contribution of Intraseasonal Time Scales to Surface Chlorophyll Variations in a Bloom and an Oligotrophic Regime**  
> [Read the paper here](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JC015701)

## Features
- Designed for **weekly** chlorophyll-a time series processing
- Implements **Keerthi's detrending** methodology
- Can be **easily adapted** to **monthly data** by:
  - Modifying the **convolution kernel length**
  - Adjusting the **Henderson filter values**

## How to Use
To adapt the code for **monthly** data:
1. Change the **length of the convolution kernel** accordingly.
2. Modify the **Henderson filter values** to suit the new time scale.

## References
Keerthi et al., 2019: [DOI: 10.1029/2019JC015701](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019JC015701)

## License
This project follows the applicable licensing terms. Please check the repository for details.
