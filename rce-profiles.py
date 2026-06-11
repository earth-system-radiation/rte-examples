import numpy as np 
import xarray as xr
import metpy.calc as mpcalc
from   metpy.units import units
import metpy.constants as mpconst

def get_pressure_grids(surface_pressure=1000e2, top_pressure=1, num=128):
    r"""Create matching pressures at full-levels and half-levels.

    The half-levels range from ``surface_pressure`` to 1 Pa.

    Parameters:
        surface_pressure (float): Pressure of the lowest half-level [Pa].
        top_pressure (float): Pressure at the highest half-level [Pa].
        num (int): Number of **full** pressure levels.

    Returns:
        ndarray, ndarray: Full-level pressure, half-level pressure [Pa].

    """
    phlog = np.linspace(np.log(surface_pressure), np.log(top_pressure), num)
    plog  = 0.5 * (phlog[1:] + phlog[:-1])

    return np.exp(plog), np.exp(phlog)

def create_profile_moist(Ts, rh, p, Tstrat = 200, ps=None):
    ''' 
    Low level (numpy) function to create an idealized profile with given surface temperature 
    following a moist adiabatic lapse rate until a 200K isothermal stratosphere
    Relies on metpy

    IN:
        T_s [K]: given surface temperature
        rh [%]: given relative humidity, constant throughout atmosphere
        
    OUT:
        T_profile [K]: temperature profile
        vmr [unitless]: water vapor volume mixing ratio
    
    '''
    assert((p >= 100).all()) # metpy limitation(?)
    if ps is None: ps = p[0]
    T_profile =  np.maximum(mpcalc.moist_lapse(
                                    p * units("Pa") , 
                                    Ts * units("K") , 
                                    ps * units("Pa")).magnitude, 
                            Tstrat)
    mass_mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(p * units("Pa"), 
                                                                   T_profile * units("K"), 
                                                                   .5 * units("dimensionless")).to('kg/kg').magnitude 
    # mmr = vmr * molar mass / molar mass of dry air
    vmr = mass_mixing_ratio * \
          (mpconst.dry_air_molecular_weight.to('kg/mole')/mpconst.water_molecular_weight.to('kg/mole')
          ).magnitude   
    
    return T_profile, vmr

def construct_profile(ps = 1000e2, rh = 0.5, 
                      gas_concs = None, 
                      p0 = 100, num=128, 
                      Ts = 285, Tstrat = 200):
    """
    Create an xr.Dataset for a moist adiabatic troposphere under an isothermal stratosphere based
       on specification of surface temperature, relative humidity, and the surface and 
       stratospheric temperatures.Temperature on layers and levels. 
    Includes water vapor only by default
    """
    play, plev = get_pressure_grids(ps, p0, num)
    Tlay, h2o_vmr = create_profile_moist(Ts, rh, play, Tstrat = Tstrat, ps = ps)
    Tlev,  _      = create_profile_moist(Ts, rh, plev, Tstrat = Tstrat, ps = ps)
    ds = xr.Dataset(
                data_vars = dict(
                      pres_layer = (["layer"], play),
                      pres_level = (["level"], plev),
                      temp_layer = (["layer"], Tlay),
                      temp_level = (["level"], Tlev),
                      h2o  = (["layer"], h2o_vmr),
                      surface_temperature   = ([], Ts),
                      ps   = ([], ps), 
                      rh   = ([], rh),
                ),
            ) 
    if gas_concs is not None:
        for k, v in gas_concs.items():
            ds[k] = v

    return  ds

base = construct_profile(Ts = 295)
# 
# Concentrations of gases (other than water) required by RRTMGP
#
gas_concs = \
    {"co2": 428e-6, 
     "o3": 3.6478 * (base.pres_layer*0.01)**0.83209 * np.exp(-(base.pres_layer*0.01) / 11.3515) * 1e-6, 
     "ch4": 1.94e-6, 
     "n2o": 0.339e-6,
     "cfc11": 216e-12,
     "cfc12": 480e-12,
     "o2": 0.2095, 
     "co": 0,
     }

g = xr.concat([construct_profile(Ts = Ts, gas_concs = gas_concs) for Ts in np.arange(273, 305)], 
          dim = "col")

####
# Testing 
###
from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp import rte
from pyrte_rrtmgp.rrtmgp import GasOptics

gas_optics_lw = GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)
gas_optics_sw = GasOptics(
    gas_optics_file=GasOpticsFiles.SW_G224
)


gas_optics_lw.compute(g)

