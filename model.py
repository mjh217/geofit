import abc
import numpy as np

class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, depth, pressure, temperature):
        """
        Depth - Meters
        Pressure - Pa
        Temperature - Kelvin

        Returns all necessary material parameters as a tuple
           i.e., (alpha, cp, rho, cond)
        """
        pass

    def output_shape(*args):
        size = 0
        for arg in args:
            if not isinstance(arg, (int, float, Model)):
                size = max(size, len(arg))

        return 1.0 if size == 0 else np.ones(size)


class ConstantModel(Model):
    def __init__(self, alpha, cp, rho, cond):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.cond = cond

    def compute(self, depth, pressure, temperature):
        shape = self.output_shape(depth, pressure, temperature)
        return self.alpha * shape, self.cp * shape, self.rho * shape, self.cond * shape


class OsakoModel(Model):
    #
    # This is a Mantle Model corresponding to mmod = 4 in the fitplot code
    #
    def __init__(self, alpha, cp, rho):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho

    def compute(self, depth, pressure, temperature):
        cond = (1.175 + 1264.0 / (temperature)) * np.exp(0.038 * pressure / 1e9)

        shape = self.output_shape(depth, pressure, temperature)
        return self.alpha * shape, self.cp * shape, self.rho * shape, cond * shape


class HasterokModel(Model):
    #
    # TODO: Reference for this model

    def __init__(self, alpha, cp, rho, crustk=2.5):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.crustk = crustk

    def compute(self, depth, pressure, temperature):
        hasterok_n = 6.4 - (2.3 * np.log(self.crustk))
        hasterok_beta = 0.1
        hasterok_k1 = (
            self.crustk
            * ((hasterok_n - 1.0) / hasterok_n)
            * (1.0 + hasterok_beta * pressure)
        )
        hasterok_k2 = (
            self.crustk
            * (np.exp(-(temperature - 25.0) / 300.0))
            * (1.0 / hasterok_n)
            * (1.0 + hasterok_beta * pressure)
        )
        cond = hasterok_k1 + hasterok_k2  # Variable temperature-dependent crustal value

        shape = self.output_shape(depth, pressure, temperature)
        return self.alpha * shape, self.cp * shape, self.rho * shape, cond * shape



def dV0dV_crust(P):
    # Isothermal volume change using a polynomial fit to the third-order Birch-Murnaghan EoS from Grose & Afonso using values of K0 = 90 GPa, K'T = 4.8 (i.e. crustal; issue #7)
    return 1.00000236 + (1.10952146e-11 - 2.12421383e-22 * P) * P

def dV0dV_mantle(P):
    # Isothermal volume change using a polynomial fit to the third-order Birch-Murnaghan EoS from Grose & Afonso using values of K0 = 130 GPa, K'T = 4.8 (i.e. mantle; issue #7)
    return 1.00009143 + (7.57454484e-12 - 8.00340143e-23 * P) * P



class HoggardModel(Model):
    def __init__(self, rho=3.3e3, grain_size=0.1, bulk_modulus=130e9):
        self.rho0 = rho

        # Kelvin to Celsius conversion factor (K)
        self.C2K = 273.15

        # Thermal expansivity T-dependence constants for fosterite from Bouhifd et al. (1996)
        self.alpha0 = 2.832e-5
        self.alpha1 = 0.758e-8

        # Cp(T) constants from Korenaga & Korenaga (2016)
        self.c0 = 1580.0
        self.c1 = 12230.0
        self.c2 = 1694e6

        # Pressure dependence of lattice conductivity from Hofmeister (2007) for "hard materials (mantle phases)"
        self.dlogkdp = 4.0 / bulk_modulus

        # Lattice thermal diffusivity (kappa) from Grose & Afonso (2013) fitted to "mantle olivine" data of Pertermann & Hofmeister (2006)
        self.kappa0 = 0.565
        self.kappa1 = 0.67
        self.kappa2 = 590.0
        self.kappa3 = 1.4
        self.kappa4 = 135.0

        # Radiative thermal diffusivity coefficients (depending on grain size in cm) from Grose & Afonso (2013) fitted to relationship of Hofmeister (2005) -- original typos corrected
        self.Ar = 1.8 * (1.0 - np.exp(-(grain_size**1.3) / 0.15)) - (1.0 - np.exp(-np.sqrt(grain_size) / 5.0) )
        self.Br = 11.7 * np.exp(-grain_size / 0.159) + 6.0 * np.exp(-(grain_size**3) / 10.0 )
        self.Ta = (490.0 + 1850.0 * np.exp(-(grain_size**0.315) / 0.825) + 875.0 * np.exp(-grain_size / 0.18) )
        self.Tb = 2700.0 + 9000.0 * np.exp(-np.sqrt(grain_size) / 0.205)
        self.xA = 167.5 + 505.0 * np.exp(-np.sqrt(grain_size) / 0.85)
        self.xB = 465.0 + 1700.0 * np.exp(-(grain_size**0.94) / 0.175)

        # Anderson-Gruneisen parameter from Grose & Afonso (2013) based on Anderson & Isaak (1995) for Fo90
        self.ag = 6.0

    def cp(self, T):
        # Cp(T) relationship from Korenaga & Korenaga (2016), T in Kelvin. NB. No pressure dependence (very weak over upper mantle P range -- Hofmeister, 2007)
        return self.c0 - self.c1 / np.sqrt(T) - self.c2 / T**3

    def kappalat(self, T):
        # Lattice thermal diffusivity (kappa) as function of temperature relationship from Grose & Afonso (2013) using T in Celsius
        return (self.kappa0 + self.kappa1 * np.exp((self.C2K - T) / self.kappa2) + self.kappa3 * np.exp((self.C2K - T) / self.kappa4) ) * 1e-6

    def krad(self, T):
        # Radiative thermal diffusivity relationship from Grose & Afonso (2013) using T in Celsius (P dependence currently unconstrained)
        return self.Ar * np.exp(-(((T - self.C2K) - self.Ta) ** 2) / (2.0 * self.xA**2)) + self.Br * np.exp(-(((T - self.C2K) - self.Tb) ** 2) / (2.0 * self.xB**2))

    def conductivity(self, P, T):
        # Combine lattice and radiative conductivity components 
        return self.kappalat(T) * self.rho(P, T) * self.cp(T) * np.exp(self.dlogkdp * P) + self.krad(T)

    def alpha(self, T):
        # Temperature dependence of thermal expansivity from Bouhifd et al. (1996) using T in Kelvin
        return self.alpha0 + self.alpha1 * T

    def rho(self, P, T):
        # Tempertarure and pressure dependence of density from Grose & Afonso (2013) with T in Kelvin
        dV0dV = dV0dV_mantle(P)
        alpha_ratio = dV0dV * np.exp((self.ag + 1.0) * (1.0 / dV0dV - 1.0))
        s = self.alpha0 * (T - self.C2K) + 0.5 * self.alpha1 * (T**2 - self.C2K**2)
        return self.rho0 * dV0dV * (1.0 - alpha_ratio * s)

    def compute(self, depth, pressure, temperature):
        alpha = self.alpha(temperature)
        cp = self.cp(temperature)
        rho = self.rho(pressure, temperature)
        cond = self.conductivity(pressure, temperature)

        shape = self.output_shape(depth, pressure, temperature)
        return alpha * shape, cp * shape, rho * shape, cond * shape



class WhittingtonModel(Model):
    def __init__(self, rho=2.7e3):
        self.rho0 = rho

        # Kelvin to Celsius conversion factor (K)
        self.C2K = 273.15

        # Temperature dependence of thermal expansivity constants from Stephenson et al. (2024) estimated using data in Skinner (1966) and Cooper & Simmons (1977)
        self.alpha0 = 1.0e-5
        self.alpha1 = 2.9e-8

        # Anderson-Gruneisen parameter
        self.ag = 6.0

    def kappa(self, T):
        # Temperature dependence of thermal diffusivity (Equations 1 & 2) from Whittington et al. (2009) with T in Kelvin
        if np.isscalar(T):
            kappa = 567.3 / T - 0.062 if T < 846.0 else 0.732 - 0.000135 * T
            return kappa / 1e6
        else:
            return [self.kappa(i) for i in T]

    def cp(self, T):
        # Temperature dependence of specific heat capacity (Equations 3 & 4) from Whittington et al. (2009) with T in Kelvin
        if np.isscalar(T):
            if T < 846.0:
                return (199.5 + 0.0857 * T - 5e6 / T**2) / 221.78e-3
            else:
                return (229.32 + 0.0323 * T - 47.9e-6 / T**2) / 221.78e-3
        else:
            return [self.cp(i) for i in T]

    def alpha(self, T):
        # Temperature dependence of thermal expansivity using T in Kelvin
        return self.alpha0 + self.alpha1 * T

    def rho(self, P, T):
        # Relationship for tempertarure and pressure dependence of density from Grose & Afonso (2013) with T in Kelvin
        dV0dV = dV0dV_crust(P)
        alpha_ratio = dV0dV * np.exp((self.ag + 1.0) * (1.0 / dV0dV - 1.0))
        s = self.alpha0 * (T - self.C2K) + 0.5 * self.alpha1 * (T**2 - self.C2K**2)
        return self.rho0 * dV0dV * (1.0 - alpha_ratio * s)

    def conductivity(self, P, T):
        return self.kappa(T) * self.rho(P, T) * self.cp(T)

    def compute(self, depth, pressure, temperature):
        alpha = self.alpha(temperature)
        cp = self.cp(temperature)
        rho = self.rho(pressure, temperature)
        cond = self.conductivity(pressure, temperature)

        shape = self.output_shape(depth, pressure, temperature)
        return alpha * shape, cp * shape, rho * shape, cond * shape




def load_from_string(model_name, alpha, cp, rho, cond, **kwargs):
    model_name = model_name.lower()
    if model_name == "constant":
        return ConstantModel(alpha, cp, rho, cond)

    elif model_name == "osako":
        return OsakoModel(alpha, cp, rho)

    elif model_name == "hoggard":
        return HoggardModel(rho, **kwargs)

    elif model_name == "whittington":
        return WhittingtonModel(rho)

    else:
        raise ValueError(f"Model name {model_name} not implemented")
