"""
Procesa los NetCDF de ERA5 descargados: velocidad media mensual del viento
(media de las magnitudes instantaneas, no magnitud del vector medio -- mismo
cuidado que con NOAA GFS0P25), por celda de la rejilla de 10 km.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

ERA5_DIR = Path(__file__).parent / "era5"
OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

ds1 = xr.open_dataset(ERA5_DIR / "era5_2023_2025.nc")
ds2 = xr.open_dataset(ERA5_DIR / "era5_2026.nc")
ds = xr.concat([ds1, ds2], dim="valid_time").sortby("valid_time")
print(f"Timesteps ERA5: {ds.sizes['valid_time']}  ({str(ds.valid_time.min().values)[:10]} -> {str(ds.valid_time.max().values)[:10]})")

speed = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
ds = ds.assign(speed=speed)

df = ds[["u10", "v10", "speed"]].to_dataframe().reset_index()
df["mes"] = pd.to_datetime(df["valid_time"]).dt.to_period("M").astype(str)

monthly = df.groupby(["latitude", "longitude", "mes"], as_index=False).agg(
    u10=("u10", "mean"), v10=("v10", "mean"), wind_speed=("speed", "mean")
)
print(f"Filas lat-lon-mes ERA5: {len(monthly)}")

# --- Trasladar a la rejilla fina de 10 km por vecino mas cercano ---
canon = monthly[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
canon_pts = gpd.GeoDataFrame(canon, geometry=gpd.points_from_xy(canon["longitude"], canon["latitude"]), crs="EPSG:4326")

grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
grid["centroid"] = grid.geometry.centroid
grid_pts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs)

nearest = gpd.sjoin_nearest(grid_pts, canon_pts, how="left")[["cell_id", "latitude", "longitude"]]
print(f"Celdas finas mapeadas: {len(nearest)}")

viento_fino = nearest.merge(monthly, on=["latitude", "longitude"], how="left")
viento_fino = viento_fino[["cell_id", "mes", "u10", "v10", "wind_speed"]]
viento_fino.to_csv(OUT_DIR / "viento_celda_mes_era5.csv", index=False)
print(f"Guardado: viento_celda_mes_era5.csv ({len(viento_fino)} filas)")

# Comparacion rapida con el viento GFS ya usado
gfs = pd.read_csv(OUT_DIR / "viento_celda_mes.csv")
gfs["cell_id"] = gfs["cell_id"].astype(str)
viento_fino["cell_id"] = viento_fino["cell_id"].astype(str)
comp = gfs.merge(viento_fino, on=["cell_id", "mes"], suffixes=("_gfs", "_era5"))
print(f"\nComparacion GFS vs ERA5 (n={len(comp)}):")
print(f"  Viento medio GFS:  {comp['wind_speed_gfs'].mean():.2f} m/s")
print(f"  Viento medio ERA5: {comp['wind_speed_era5'].mean():.2f} m/s")
print(f"  Correlacion: {comp['wind_speed_gfs'].corr(comp['wind_speed_era5']):.3f}")

# --- Fusionar con el panel del modelo, sustituyendo wind_speed ---
panel = pd.read_csv(OUT_DIR / "panel_modelo_fase6_con_ais.csv")
panel["cell_id"] = panel["cell_id"].astype(str)
panel = panel.drop(columns=["wind_speed", "u10", "v10"], errors="ignore")
final = panel.merge(viento_fino, on=["cell_id", "mes"], how="left")
final.to_csv(OUT_DIR / "panel_modelo_fase6_era5.csv", index=False)
print(f"\nGuardado: panel_modelo_fase6_era5.csv ({len(final)} filas, "
      f"{final['wind_speed'].isna().mean() * 100:.2f}% sin viento asignado)")
