# template-morphing
Use CERN's ROOT framework to fit different distributions to a signal and interpolate in between fits

1. first make ```root/``` directory in ```template-morphing/``` where all ```.root``` files will go

2. user must be in src to have all the proper definitions and reachable code
```cd src```

3. a) put all necessary parameters in ```config.json``` for general tool usage<\br>
   b) set all fitting parameters in ```/src/build/fitter-init.json```

4. run ```python fit_suite.py config.json``` to use the tool
