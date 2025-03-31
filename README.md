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
There are two parts of our code, which are for parametric approach probabilistic forecasting and non-parametric approach, respectively.
For each part, there are some notebooks for different experiments, take the parametric approach as an example:
| File name      | Description |
| ----------- | ----------- |
| main parameter dynamic price.ipynb      | To obtain the results for 4 settings|
| main parameter(Compare epsion).ipynb    |To obtain the results with different radius of ambiguity set $\epsilon$|
| main parameter(Compare N).ipynb   | To obtain the results with different samples number $N$|
| online_non_parameter.py  | Function related to online learning for parametric approach|
| test_parameter.py  | Function related to evaluation for parametric approach|


