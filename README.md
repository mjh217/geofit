# GeoFit

This software package solves the equation governing 1-D vertical conduction of heat through continental lithosphere by starting from the surface and integrating downwards until it meets the mantle adaiabat. Each geotherm is definded by the following input parameters:
1. Surface heat flow
2. Crustal thickness
3. Bulk radiogenic heat production in the crust
4. Mantle potential temperature (default = 1330°C)
5. Bulk radiogenic heat production in the lithospheric mantle (default = 0)

And a user can also select amongst different models for the neccessary density and thermal paramters ($\alpha$, $C_P$, $k$, $\rho_0$), some of which permit inclusion of temperature- and pressure-dependence of these parameters. The code then uses a Bayesian probability framework to evaluate the posterior distribution of these input parameters, given both a user-provided assumption of their prior probability distribution and a set of underlying pressure-temperature, $P-T$ constraints from within the lithosphere that come from independent datasets including thermobarometric analysis of mantle xenoliths/xenocrysts and/or conversion of seismic velocities into temperature. In addition to posterior distributions of the input parameters, it also provides estimates of the thickness of the mechanical and thermal lithosphere-asthenosphere boundaries (LABs) and the range of potential temperatures as a function of depth.

# Local files

- [defaults.py](defaults.py) - Contains reasonable default values for the various parameters.
- [forwardmodel.py](forwardmodel.py) - Contains code to solve the forward problem of deriving a geotherm given a set of input parameters.
- [model.py](model.py) - Contains models governing the various density and thermal parameters such as $\alpha$, $C_P$, $k$, $\rho_0$.
- [bayesian.py](bayesian.py) - Contains functions to support Bayesian inversion (e.g., definition of likelihood, prior probabilities).
- [invert.py](invert.py) - An executable script to read in various command-line arguments and $P-T$ constraints, and invert them to find a family of potential geotherms.

An example snippet to run the code on the command line is:
```python invert.py TP_data.csv --invertTp --iwrite```

# How to cite

Code development was undertaken at the Australian National University and was funded by Geoscience Australia as a contribution to the *Resourcing Australia's Prosperity* initiative. A supporting publication is being prepared for submission:

Hoggard, M.J., R. Hawkins, T. Duvernay, S. Bagchi, F.D. Richards, Z.J. Sudholz, A.L. Jaques, G.M. Yaxley, K. Czarnota & M. Haynes (in prep). Paleogeothermal structure of the Australian continent.
