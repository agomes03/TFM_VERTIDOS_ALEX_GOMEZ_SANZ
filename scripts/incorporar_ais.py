"""
Trafico AIS real (Global Fishing Watch, 4Wings API - AIS vessel presence)
como covariable mensual por celda. Se agrega a la rejilla nativa de GFW
(resolucion LOW, ~0.1 grados) sumando horas-buque por celda-mes, y se
traslada a nuestra rejilla fina de 10 km por vecino mas cercano (mismo
patron que con el viento).
"""
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import gfwapiclient as gfw

BBOX = (47.5, 23.5, 56.5, 30.5)  # golfo_persico
TOKEN_PATH = Path(__file__).parent / "gfw_token.txt"
OUT_DIR = Path(__file__).parent / "salida_golfo_persico"


async def fetch_month(client, geojson, start, end):
    result = await client.fourwings.create_ais_presence_report(
        spatial_resolution="LOW",
        temporal_resolution="MONTHLY",
        start_date=start,
        end_date=end,
        geojson=geojson,
    )
    df = result.df()
    if df.empty:
        return pd.DataFrame(columns=["lat", "lon", "hours"])
    return df.groupby(["lat", "lon"], as_index=False)["hours"].sum()


async def main():
    token = TOKEN_PATH.read_text().strip()
    client = gfw.Client(access_token=token)

    minx, miny, maxx, maxy = BBOX
    geojson = {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}

    meses = pd.period_range("2023-01", pd.Timestamp.now().to_period("M"), freq="M")
    rows = []
    for period in meses:
        start = period.start_time.strftime("%Y-%m-%d")
        end = (period.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            agg = await fetch_month(client, geojson, start, end)
        except Exception as e:
            print(f"  [ERROR] {period}: {e}")
            continue
        agg["mes"] = str(period)
        rows.append(agg)
        print(f"  {period}: {len(agg)} celdas GFW (0.1°), {agg['hours'].sum():.0f} horas-buque totales")

    traffic_raw = pd.concat(rows, ignore_index=True)
    traffic_raw.to_csv(OUT_DIR / "trafico_ais_gfw_celda01_mes.csv", index=False)
    print(f"\nGuardado bruto: trafico_ais_gfw_celda01_mes.csv ({len(traffic_raw)} filas)")

    # --- Trasladar a nuestra rejilla fina por vecino mas cercano ---
    canon = traffic_raw[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    canon_pts = gpd.GeoDataFrame(canon, geometry=gpd.points_from_xy(canon["lon"], canon["lat"]), crs="EPSG:4326")

    grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
    grid["centroid"] = grid.geometry.centroid
    grid_pts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs)

    nearest = gpd.sjoin_nearest(grid_pts, canon_pts, how="left")[["cell_id", "lat", "lon"]]
    print(f"Celdas finas mapeadas a celda GFW mas cercana: {len(nearest)}")

    traffic_fino = nearest.merge(traffic_raw, on=["lat", "lon"], how="left")
    traffic_fino = traffic_fino.rename(columns={"hours": "ais_hours"})[["cell_id", "mes", "ais_hours"]]
    traffic_fino["ais_hours"] = traffic_fino["ais_hours"].fillna(0.0)
    traffic_fino.to_csv(OUT_DIR / "trafico_ais_celda_mes.csv", index=False)
    print(f"Guardado: trafico_ais_celda_mes.csv ({len(traffic_fino)} filas)")

    # --- Fusionar con el panel del modelo ---
    panel = pd.read_csv(OUT_DIR / "panel_modelo_fase6.csv")
    panel["cell_id"] = panel["cell_id"].astype(str)
    traffic_fino["cell_id"] = traffic_fino["cell_id"].astype(str)
    final = panel.merge(traffic_fino, on=["cell_id", "mes"], how="left")
    final.to_csv(OUT_DIR / "panel_modelo_fase6_con_ais.csv", index=False)
    print(f"Guardado: panel_modelo_fase6_con_ais.csv ({len(final)} filas, "
          f"{final['ais_hours'].isna().mean() * 100:.2f}% sin AIS)")


asyncio.run(main())
