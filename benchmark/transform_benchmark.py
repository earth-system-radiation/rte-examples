# %% Download and transform the rte-benchmarks reference outputs
from pathlib import Path

import pooch
import xarray as xr

VERSION = "1.0"
ARCHIVE_URL = (
    f"https://github.com/m-brath/rte-benchmarks/archive/refs/tags/v{VERSION}.tar.gz"
)
ARCHIVE_NAME = f"rte-benchmarks-v{VERSION}.tar.gz"

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
CASE_NAMES = ("ckdmip", "rce", "rfmip")


def transform_benchmark():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download and extract the rte-benchmarks archive
    pooch.retrieve(
        url=ARCHIVE_URL,
        known_hash=None,
        fname=ARCHIVE_NAME,
        path=RAW_DIR,
        processor=pooch.Untar(extract_dir=RAW_DIR),
    )

    archive_root = sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_dir() and path.name.startswith("rte-benchmarks-")
    )[-1]

    # Transform the reference outputs
    for case_name in CASE_NAMES:
        for band in ("LW", "SW"):
            source_dir = archive_root / "results" / case_name / band
            source_file = sorted(source_dir.glob("Reference_fluxes_Nf*.nc"))[0]
            raw = xr.open_dataset(source_file).load()

            reference = xr.Dataset(
                {
                    f"{band.lower()}_flux_up": (
                        ("variant", "level", "col"),
                        raw["flux_clearsky_up"]
                        .transpose("variant", "level", "column")
                        .values,
                        {"units": raw["flux_clearsky_up"].attrs.get("units", "")},
                    ),
                    f"{band.lower()}_flux_dn": (
                        ("variant", "level", "col"),
                        (-raw["flux_clearsky_down"])
                        .transpose("variant", "level", "column")
                        .values,
                        {"units": raw["flux_clearsky_down"].attrs.get("units", "")},
                    ),
                }
            )

            output_path = SCRIPT_DIR / f"arts-{band.lower()}-{case_name}.nc"
            reference.to_netcdf(output_path, engine="netcdf4")


if __name__ == "__main__":
    transform_benchmark()

# %%
