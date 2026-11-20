"""Default physical parameters and model limits (SI units unless stated)."""

C2K = 273.15  # Celsius to Kelvin offset [K]

# Surface boundary conditions
T_SURFACE = 0.0 + C2K  # [K]
P_SURFACE = 101325.0  # [Pa], 1 atm

# Mantle potential temperature, i.e. the adiabat extrapolated to the surface
TP = 1330.0 + C2K  # [K]

# Properties for the constant-parameter models
RHO_CRUST = 2.7e3  # density [kg/m^3]
ALPHA_CRUST = 3e-5  # thermal expansivity [1/K]
CP_CRUST = 0.75e3  # specific heat capacity [J/kg/K]
K_CRUST = 2.5  # thermal conductivity [W/m/K]

RHO_MANTLE = 3.3e3
ALPHA_MANTLE = 3e-5
CP_MANTLE = 1.187e3
K_MANTLE = 2.5

# Mantle properties for the Hoggard model
BULK_MODULUS_MANTLE = 130e9  # [Pa]
GRAIN_SIZE_MANTLE = 0.1  # [cm]

# Depth limits
Z_MAX = 400e3  # deepest point of a geotherm [m]
Z_CRUST_MAX = 100e3  # allowance for the crust when integrating the adiabat [m]
Z_MAX_ADIABAT = Z_MAX + Z_CRUST_MAX  # the adiabat is integrated from the surface to here [m]
