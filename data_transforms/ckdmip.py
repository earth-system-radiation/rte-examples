import sys
import xarray as xr
from .pooch_config import list_files
import metpy.constants as mpconst

from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)
from pyrte_rrtmgp.rrtmgp import GasOptics

OUTPUT_FILE = "ckdmip-states.nc"

def transform_files(): 
	f = xr.open_mfdataset(list_files("ckdmip"), 
					concat_dim = "variant", 
					combine='nested', 
					engine="netcdf4").\
			rename_dims({ \
				"level":"layer", 
				"column":"col",
				}). \
			rename_vars({ \
				"temperature_hl":"temp_level", 
				"temperature_fl":"temp_layer", 
				"pressure_hl":"pres_level", 
				"pressure_fl":"pres_layer", 
				}).\
			drop_vars(['latitude', 'longitude', 'time', 'level', 'half_level'])

	f = f.rename_dims({ \
				"half_level":"level", 
				}). \
			rename_vars({v:v.split("_")[0] for v in f.variables if "mole_fraction_fl" in v}).\
			drop_vars([v for v in f.variables if "mole_fraction_hl" in v]). \
			drop_attrs()
	# Atmosphere is ordered top to bottom - lowest level is surface tempterature 
	f["surface_temperature"] = f.temp_level.isel(level=-1)
	f["expt_names"] = xr.DataArray(data = [n.split("/")[-1] for n in list_files("ckdmip")], 
		dims = ["variant"])
	f["co"] = 0. 
	f["surface_emissivity"] = 1.
	f["surface_albedo"] = 0. 
	f["solar_zenith_angle"] = 0. 
	f["total_solar_irradiance"] = mpconst.earth_solar_irradiance.m
	#
	# pyRTE logic for computing top_at_1 is fragile - use only one set of pressures across variants
	#
	f["pres_layer"] = f["pres_layer"].isel(variant=0)
	f["pres_level"] = f["pres_level"].isel(variant=0)

	####
	# Limit pressures to those covered by RRTMGP
	###
	gas_optics_lw = GasOptics(
	    gas_optics_file=GasOpticsFiles.LW_G256, 
	)

	f["pres_layer"] = xr.where(f.pres_layer > gas_optics_lw.press_min + sys.float_info.epsilon, 
							   f.pres_layer, 
							   gas_optics_lw.press_min + sys.float_info.epsilon)

	return f 
