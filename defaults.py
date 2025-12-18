#
# Some default parameters
#
rho_crust = 2.7e3 # kg/m^3
rho_mantle = 3.3e3 # kg/m^3

alpha_crust = 3e-5 # K^-1
cp_crust = 0.75e3 # J/kg/K
k_crust = 2.5 # W/m/K

alpha_mantle = 3e-5 # K^-1
cp_mantle = 1.187e3 # J/kg/K
k_mantle = 2.5 # W/m/K

bulk_modulus_crust = 90e9 # Pa
bulk_modulus_mantle = 130e9 # Pa

grain_size_mantle = 0.1 # cm

Tp = 1330.0 + 273.15 # K

P0 = 101325.0 # Pa
T0 = 0.0 + 273.15 # K --- not currently used correctly

Z_MAX = 400e3 # k
Z_CRUSTMAX = 100e3 # m
Z_MAX_ADIABAT = Z_MAX + Z_CRUSTMAX
