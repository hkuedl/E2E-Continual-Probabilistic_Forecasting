# E2E-Continual-Probabilistic_Forecasting

This repository contains codes for our paper "Decision-focused Learning for Local Energy Communities Management Under Uncertainty", authored by Yangze Zhou, Yihong Zhou, Thomas Morstyn, and Yi Wang.

## Environments
The environments for the code can be installed by
```
conda env create -f environments.yml
```

## Data
The load data used for experiments can be found in ```./Data/GEF_data```, and the solar power data used for experiments can be found in ```./Data/PV```.

## Code
Our code consists of two parts: one for the parametric approach-based probabilistic forecasting and the other for the non-parametric approach. Each part includes several notebooks for different experiments. For example, in the parametric approach:

| File name      | Description |
| ----------- | ----------- |
| main parameter dynamic price.ipynb      | To obtain the results for 4 settings|
| main parameter(Compare epsion).ipynb    |To obtain the results with different radius of ambiguity set $\epsilon$|
| main parameter(Compare N).ipynb   | To obtain the results with different samples number $N$|
| online_non_parameter.py  | Function related to online learning for parametric approach|
| test_parameter.py  | Function related to evaluation for parametric approach|

 You can obtain the results for different settings by changing the parameter "flag_dynamic_price" and "flag_dynamic_mode".

 ## Results
 You can find the results for the parametric and non-parametric approaches in the folders labeled ```parameter``` and ```Non parameter```, respectively.

For the ablation experiments related to $N$ and $\epsilon$, please refer to the folders labeled ```Different N``` and ```Different Epsion```.
