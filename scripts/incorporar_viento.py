"""
Incorpora viento (velocidad/componentes u,v a 10m) como covariable mensual por
celda al panel de exposicion. ERA5 (ECMWF/ERA5/MONTHLY y DAILY en GEE) dejo de
actualizarse en 2020, y ERA5-Land esta enmascarado sobre mar abierto (el Golfo
Persico es casi todo mar) -- verificado ambos con consultas directas a Earth
Engine. Se usa en su lugar NOAA/GFS0P25 (modelo operativo GFS, forecast_hours=0
= analisis), que cubre 2015-2026 completo y SI tiene datos sobre el mar.
"""
import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

EE_PROJECT = os.environ["EE_PROJECT"]
ee.Initialize(project=EE_PROJECT)

BBOX = (47.5, 23.5, 56.5, 30.5)  # golfo_persico
GFS_SCALE = 27830  # resolucion nativa aprox. de GFS0P25 (0.25 grados)

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"


def monthly_wind_points(aoi_geom, start, end):
    """Imagen media mensual (forecast_hours=0) de u10/v10 muestreada a la
    resolucion nativa de GFS0P25 sobre el AOI -> lista de puntos con valores.

    wind_speed se calcula como la MEDIA DE LAS MAGNITUDES instantaneas, no
    como la magnitud del vector medio: si el viento cambia de direccion
    durante el mes, promediar u,v primero y sacar la magnitud despues
    subestima la velocidad real (los vectores se cancelan parcialmente).
    Verificado con un punto real: subestima ~8% en un mes de prueba."""
    coll = (ee.ImageCollection("NOAA/GFS0P25")
            .filterDate(start, end)
            .filter(ee.Filter.eq("forecast_hours", 0))
            .select(["u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"],
                    ["u10", "v10"]))

    def add_speed(img):
        speed = img.expression("sqrt(u10*u10 + v10*v10)", {"u10": img.select("u10"), "v10": img.select("v10")})
        return img.addBands(speed.rename("wind_speed_inst"))

    img = coll.map(add_speed).select(["u10", "v10", "wind_speed_inst"]).mean().rename(["u10", "v10", "wind_speed"])
    sample = img.sample(region=aoi_geom, scale=GFS_SCALE, geometries=True)
    return sample.getInfo()


if __name__ == "__main__":
    aoi = ee.Geometry.Rectangle(list(BBOX))

    grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
    grid["centroid"] = grid.geometry.centroid
    grid_pts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs)

    meses = pd.period_range("2023-01", pd.Timestamp.now().to_period("M"), freq="M")

    rows = []
    for period in meses:
        start = period.start_time.strftime("%Y-%m-%d")
        end = (period.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            info = monthly_wind_points(aoi, start, end)
        except Exception as e:
            print(f"  [ERROR] {period}: {e}")
            continue

        feats = info.get("features", [])
        if not feats:
            print(f"  {period}: sin puntos de viento (omitido)")
            continue

        wind_pts = gpd.GeoDataFrame(
            [{
                "u10": f["properties"].get("u10"),
                "v10": f["properties"].get("v10"),
                "wind_speed": f["properties"].get("wind_speed"),
                "geometry": Point(f["geometry"]["coordinates"]),
            } for f in feats if f.get("geometry")],
            crs="EPSG:4326",
        )

        joined = gpd.sjoin_nearest(grid_pts, wind_pts, how="left")[
            ["cell_id", "u10", "v10", "wind_speed"]
        ]
        joined["mes"] = str(period)
        rows.append(joined)
        print(f"  {period}: {len(wind_pts)} puntos GFS -> {len(joined)} celdas | viento medio {joined['wind_speed'].mean():.2f} m/s")

    viento_celda_mes = pd.concat(rows, ignore_index=True)
    viento_celda_mes.to_csv(OUT_DIR / "viento_celda_mes.csv", index=False)
    print(f"\nGuardado: viento_celda_mes.csv ({len(viento_celda_mes)} filas)")

    # --- Fusionar con el panel de exposicion ---
    exposicion = pd.read_csv(OUT_DIR / "exposicion_mensual_celda.csv")
    exposicion["cell_id"] = exposicion["cell_id"].astype(str)
    viento_celda_mes["cell_id"] = viento_celda_mes["cell_id"].astype(str)

    final = exposicion.merge(viento_celda_mes, on=["cell_id", "mes"], how="left")
    final.to_csv(OUT_DIR / "exposicion_mensual_celda_con_viento.csv", index=False)
    print(f"Guardado: exposicion_mensual_celda_con_viento.csv ({len(final)} filas, "
          f"{final['wind_speed'].isna().mean() * 100:.2f}% sin viento asignado)")
