import xarray as xr
import pyrte_rrtmgp.rte as rte
from pyrte_rrtmgp.rrtmgp.data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp.rrtmgp import GasOptics

gas_optics_lw = GasOptics(
    gas_optics_file=GasOpticsFiles.LW_G256
)
gas_optics_sw = GasOptics(
    gas_optics_file=GasOpticsFiles.SW_G224
)

FILE_LIST = ["rfmip-states.nc", "rce-states.nc", "ckdmip-states.nc"]
for f in FILE_LIST:
    print(f"working on {f}")
    atmosphere = xr.open_dataset(f, engine="netcdf4")

    fluxes = xr.merge([
        xr.merge([
            gas_optics_lw.compute(atmosphere, add_to_input = False), 
            atmosphere.surface_emissivity, 
        ]).rte.solve(add_to_input = False),  
        xr.merge([
            gas_optics_sw.compute(atmosphere, add_to_input = False), 
            atmosphere.surface_albedo, 
            atmosphere.solar_zenith_angle, 
        ]).rte.solve(add_to_input = False)
    ]) 
