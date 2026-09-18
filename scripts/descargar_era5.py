"""
Descarga viento real ERA5 (10m u,v) del Copernicus Climate Data Store,
golfo Persico, 2023-01 a 2026-08, 4 veces al dia (00/06/12/18 UTC).
Se pide en dos tramos (2023-2025 y 2026) para acotar el tamano de cada
peticion.
"""
import os
from pathlib import Path

os.environ["CDSAPI_RC"] = str(Path(__file__).parent / "cdsapirc.txt")
import cdsapi

OUT_DIR = Path(__file__).parent / "era5"
OUT_DIR.mkdir(exist_ok=True)

AREA = [30.5, 47.5, 23.5, 56.5]  # N, W, S, E
TIMES = ["00:00", "06:00", "12:00", "18:00"]
DAYS = [f"{d:02d}" for d in range(1, 32)]

client = cdsapi.Client()


def fetch(years, months, out_name):
    out_path = OUT_DIR / out_name
    if out_path.exists():
        print(f"  {out_name} ya existe, se omite")
        return
    print(f"Pidiendo {out_name}: years={years} months={months} ...")
    r = client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": years,
            "month": months,
            "day": DAYS,
            "time": TIMES,
            "area": AREA,
            "format": "netcdf",
        },
    )
    r.download(str(out_path))
    print(f"  Guardado: {out_path}")


fetch(["2023", "2024", "2025"], [f"{m:02d}" for m in range(1, 13)], "era5_2023_2025.nc")
fetch(["2026"], [f"{m:02d}" for m in range(1, 9)], "era5_2026.nc")

print("\nDescarga ERA5 completa.")
