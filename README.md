# template-morphing
Use CERN's ROOT framework to fit different distributions to a signal and interpolate in between fits

For py2.7.15
 - most recent version of ROOT (CERN)
 - numpy
 - torch
 - matplotlib

For py3.9.6
 - most/all required libraries for X. Ding's CCGAN architecture: https://github.com/UBCDingXin/improved_CcGAN.git


1. first make ```root/``` directory in ```template-morphing/``` where all ```.root``` files will go

2. user must be in src to have all the proper definitions and reachable code
```cd src```

3. in py2.7.15 env, run ```python ROOTtoNP_Gaus.py``` to generate 2D gaussians with number of bins as the spread defined in L32-35

4. edit hyperparameters.json for vicinity parameters

5. in py3.9.6 env, run ```python TM_Gaus.py``` to perform the template morphing procedure on generated 2D Gaussians

Needs two separate environments: py2.7.15 and py3.9.6:
