"""
Covariables espaciales estaticas por celda: distancia a costa y distancia a
la ruta/esquema de separacion de trafico maritimo mas cercano.

Nota sobre trafico AIS real: se intento World Bank "Global Shipping Traffic
Density" (datacatalog.worldbank.org esta caido con error 5xx ahora mismo,
verificado por dos vias distintas) y el servicio ArcGIS ImageServer que lo
sirve (capabilities=TilesOnly, no expone exportImage/getSamples para pixeles
en bruto). Tambien se descarto Global Fishing Watch por requerir token de API
(credencial que el usuario tendria que dar de alta y pegar el). Como
sustituto real y sin credenciales, se usa OpenStreetMap (Overpass API): la
distancia a la ruta o esquema de separacion de trafico (TSS) mapeado mas
cercano es un proxy razonable de "estar en una ruta de tráfico marítimo",
aunque no mide densidad continua de trafico como si lo haria un raster AIS.
"""
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

BBOX = (47.5, 23.5, 56.5, 30.5)  # golfo_persico
UTM_CRS = "EPSG:32640"  # UTM 40N, razonable para todo el Golfo Persico

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
COASTLINE_PATH = Path(__file__).parent / "ne_10m_coastline.geojson"


def fetch_shipping_lanes(bbox):
    minx, miny, maxx, maxy = bbox
    query = (
        f'[out:json][timeout:90];'
        f'(way["seamark:type"~"separation_lane|separation_zone|separation_boundary|fairway"]'
        f'({miny},{minx},{maxy},{maxx}););out geom;'
    )
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query}, timeout=120,
        headers={"User-Agent": "tfm-vertidos-research/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    lines = []
    for el in data.get("elements", []):
        coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
        if len(coords) >= 2:
            lines.append({"tipo": el.get("tags", {}).get("seamark:type"), "geometry": {"type": "LineString", "coordinates": coords}})
    gdf = gpd.GeoDataFrame(
        [{"tipo": l["tipo"], "geometry": shape(l["geometry"])} for l in lines],
        crs="EPSG:4326",
    )
    return gdf


if __name__ == "__main__":
    grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
    grid["centroid"] = grid.geometry.centroid
    grid_pts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs).to_crs(UTM_CRS)

    # --- Distancia a costa (Natural Earth 10m coastline, recortada al AOI + margen) ---
    minx, miny, maxx, maxy = BBOX
    margin = 2.0
    coast = gpd.read_file(COASTLINE_PATH, bbox=(minx - margin, miny - margin, maxx + margin, maxy + margin))
    coast_utm = coast.to_crs(UTM_CRS)
    coast_union = coast_utm.union_all()

    grid_pts["dist_costa_km"] = grid_pts.geometry.distance(coast_union) / 1000.0
    print(f"Distancia a costa: min={grid_pts['dist_costa_km'].min():.2f} km, "
          f"media={grid_pts['dist_costa_km'].mean():.2f} km, max={grid_pts['dist_costa_km'].max():.2f} km")

    # --- Distancia a ruta/TSS marítima más cercana (OpenStreetMap) ---
    lanes = fetch_shipping_lanes(BBOX)
    print(f"Rutas/TSS obtenidas de OSM: {len(lanes)}")
    if not lanes.empty:
        lanes_utm = lanes.to_crs(UTM_CRS)
        lanes_union = lanes_utm.union_all()
        grid_pts["dist_ruta_maritima_km"] = grid_pts.geometry.distance(lanes_union) / 1000.0
        print(f"Distancia a ruta marítima: min={grid_pts['dist_ruta_maritima_km'].min():.2f} km, "
              f"media={grid_pts['dist_ruta_maritima_km'].mean():.2f} km, max={grid_pts['dist_ruta_maritima_km'].max():.2f} km")
    else:
        grid_pts["dist_ruta_maritima_km"] = np.nan
        print("[AVISO] Sin rutas/TSS de OSM en el AOI.")

    covariables = grid_pts[["cell_id", "dist_costa_km", "dist_ruta_maritima_km"]]
    covariables.to_csv(OUT_DIR / "covariables_espaciales_celda.csv", index=False)
    print(f"\nGuardado: covariables_espaciales_celda.csv ({len(covariables)} celdas)")

    # --- Fusionar con el panel celda-mes (estas variables son estaticas: se repiten cada mes) ---
    panel = pd.read_csv(OUT_DIR / "exposicion_mensual_celda_con_viento.csv")
    panel["cell_id"] = panel["cell_id"].astype(str)
    covariables["cell_id"] = covariables["cell_id"].astype(str)

    final = panel.merge(covariables, on="cell_id", how="left")
    final.to_csv(OUT_DIR / "panel_modelo_fase6.csv", index=False)
    print(f"Guardado: panel_modelo_fase6.csv ({len(final)} filas, "
          f"{final['dist_costa_km'].isna().mean() * 100:.2f}% sin dist_costa, "
          f"{final['dist_ruta_maritima_km'].isna().mean() * 100:.2f}% sin dist_ruta)")
