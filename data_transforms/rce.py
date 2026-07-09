import numpy as np 
import xarray as xr
import metpy.calc as mpcalc
from   metpy.units import units
import metpy.constants as mpconst

OUTPUT_FILE = "rce-states.nc"

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
                      o3  = (["layer"], \
                        3.6478 * (play*0.01)**0.83209 \
                        * np.exp(-(play*0.01) / 11.3515) \
                        * 1e-6), 
                      surface_temperature = ([], Ts),
                      rh   = ([], rh),
                      surface_emissivity = ([], 1.),
                      surface_albedo = ([], 0.),
                      solar_zenith_angle = ([], np.acos(2./3.) * 180./np.pi), # https://doi.org/10.1175/JAS-D-13-0392.1
                ),
            ) 
    if gas_concs is not None:
        for k, v in gas_concs.items():
            ds[k] = v

    return  ds


def create_files():
    gas_concs = \
        {"co2": 280e-6, 
         "ch4": 1.94e-6, 
         "n2o": 0.339e-6,
         "n2": 0.7808,
         "o2": 0.2095, 
         "co": 0,
        }
    baseline = \
      xr.concat([construct_profile(Ts = Ts, gas_concs = gas_concs) for Ts in np.arange(273, 305)], 
              dim = "col")

    gas_concs["co2"] *= 2.
    co22x = \
      xr.concat([construct_profile(Ts = Ts, gas_concs = gas_concs) for Ts in np.arange(273, 305)], 
              dim = "col")
    
    ds = xr.concat([baseline, co22x], dim = "variant")  
    ds["pres_layer"] =  ds["pres_layer"].isel(variant=0)
    ds["pres_level"] =  ds["pres_level"].isel(variant=0)
    for v in ["co2", "ch4", "n2o", "co", "n2", "o2"]:
        ds[v] = ds[v].mean(dim="col")
    ds["total_solar_irradiance"] = mpconst.earth_solar_irradiance.m
    ds["expt_names"] = xr.DataArray(data = ["PI CO2", "2xCO2"], dims = ["variant"])
    return ds.transpose("variant", "layer", "level", "col")
