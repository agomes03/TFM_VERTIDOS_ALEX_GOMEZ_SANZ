"""
Relanzamiento local (fuera de Colab) de la fase 1 del TFM, con el area fijada
en Golfo Persico, usando la misma logica ya corregida en el notebook:
paginacion completa contra la API publica de Cerulean (sin necesidad de
API key), limpieza, y construccion de la rejilla espacio-temporal +
tabla de positivos por celda-escena.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import mapping, box

BASE_URL = "https://api.cerulean.skytruth.org"
COLLECTION = "public.slick_plus"

DATETIME_START = "2023-01-01T00:00:00Z"
DATETIME_END = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DATETIME_RANGE = f"{DATETIME_START}/{DATETIME_END}"

STUDY_AREAS = {
    "golfo_persico": (47.5, 23.5, 56.5, 30.5),
}

PAGE_LIMIT = 5000
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5
CELL_KM = 10

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def fetch_all_features(bbox, datetime_range, verbose=True, max_pages=200):
    url = f"{BASE_URL}/collections/{COLLECTION}/items"
    base_params = {
        "bbox": ",".join(map(str, bbox)),
        "datetime": datetime_range,
        "limit": PAGE_LIMIT,
        "f": "json",
    }
    headers = {}

    features = []
    next_url, next_params = url, dict(base_params)
    offset = 0
    page = 0

    while next_url and page < max_pages:
        attempt = 0
        while True:
            try:
                resp = requests.get(next_url, params=next_params, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    print(f"  [ERROR] Fallo tras {MAX_RETRIES} intentos: {e}")
                    return features
                if verbose:
                    print(f"  [WARN] Error de red ({e}); reintentando ({attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_WAIT_SECONDS)

        if isinstance(data, list):
            page_features, links = data, []
        elif isinstance(data, dict) and "features" in data:
            page_features, links = data.get("features", []), data.get("links", [])
        else:
            print(f"  [ERROR] Formato inesperado. Tipo: {type(data)}")
            break

        features.extend(page_features)
        page += 1
        if verbose:
            print(f"  Pagina {page}: +{len(page_features)} registros (acumulado {len(features)})")

        next_link = next((l.get("href") for l in links if l.get("rel") == "next"), None)
        if next_link:
            next_url, next_params = next_link, None
        elif len(page_features) == PAGE_LIMIT:
            offset += PAGE_LIMIT
            next_url, next_params = url, dict(base_params, offset=offset)
        else:
            next_url = None

    if page >= max_pages and verbose:
        print(f"  [AVISO] max_pages={max_pages} alcanzado.")
    if verbose:
        print(f"  Total registros recibidos: {len(features)}")
    return features


def clean_geodataframe(features, area_name):
    if not features:
        return gpd.GeoDataFrame()

    cleaned = []
    for f in features:
        if not isinstance(f, dict):
            continue
        geom = f.get("geometry")
        if isinstance(geom, str):
            parsed = False
            try:
                geom = json.loads(geom)
                parsed = True
            except json.JSONDecodeError:
                pass
            if not parsed:
                try:
                    wkt_string = geom
                    if wkt_string.startswith("SRID="):
                        wkt_string = wkt_string.split(';', 1)[1]
                    geom = mapping(wkt.loads(wkt_string))
                    parsed = True
                except Exception:
                    pass
            if not parsed:
                continue
        if geom is None or isinstance(geom, dict):
            cleaned.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {k: v for k, v in f.items() if k != "geometry"},
            })

    if not cleaned:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(cleaned, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
    if "id" in gdf.columns:
        gdf = gdf.drop_duplicates(subset="id")

    if "slick_timestamp" in gdf.columns:
        gdf["slick_timestamp"] = pd.to_datetime(gdf["slick_timestamp"], errors="coerce", utc=True)
        gdf["anio"] = gdf["slick_timestamp"].dt.year
        gdf["mes"] = gdf["slick_timestamp"].dt.month
        gdf["dia_semana"] = gdf["slick_timestamp"].dt.dayofweek

    gdf["area_km2_aprox"] = gdf.geometry.to_crs("ESRI:54034").area / 1e6
    projected_centroids = gdf.geometry.to_crs("ESRI:54034").centroid.to_crs(gdf.crs)
    gdf["centroid_lon"] = projected_centroids.x
    gdf["centroid_lat"] = projected_centroids.y
    gdf["zona_estudio"] = area_name
    return gdf


def build_grid(bbox, cell_km=CELL_KM):
    minx, miny, maxx, maxy = bbox
    lat_mid = (miny + maxy) / 2
    dlat = cell_km / 111.0
    dlon = cell_km / (111.0 * np.cos(np.radians(lat_mid)))
    xs = np.arange(minx, maxx, dlon)
    ys = np.arange(miny, maxy, dlat)
    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            rows.append({"cell_id": f"{i}_{j}", "geometry": box(x, y, x + dlon, y + dlat)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


if __name__ == "__main__":
    print("Rango temporal pedido:", DATETIME_RANGE)

    for area_name, bbox in STUDY_AREAS.items():
        print(f"\n=== Descargando: {area_name} (bbox={bbox}) ===")
        features = fetch_all_features(bbox, DATETIME_RANGE)
        print(f"  Total detecciones brutas: {len(features)}")

        timestamps = [f.get("properties", {}).get("slick_timestamp") for f in features
                      if f.get("properties", {}).get("slick_timestamp")]
        if timestamps:
            print(f"  Rango de fechas recibido: {min(timestamps)} -> {max(timestamps)}")

        raw_path = OUT_DIR / f"raw_{area_name}.geojson"
        with open(raw_path, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": features}, fh)

        gdf = clean_geodataframe(features, area_name)
        print(f"  Tras limpieza: {len(gdf)} registros")

        if gdf.empty:
            continue

        gdf.to_file(OUT_DIR / f"cerulean_{area_name}_limpio.geojson", driver="GeoJSON")
        gdf.drop(columns="geometry").to_csv(OUT_DIR / f"cerulean_{area_name}_limpio.csv", index=False)

        resumen = {
            "zona": area_name,
            "n_detecciones": len(gdf),
            "fecha_min": str(gdf["slick_timestamp"].min()),
            "fecha_max": str(gdf["slick_timestamp"].max()),
            "confianza_media": float(gdf["machine_confidence"].mean()),
            "pct_alta_confianza_0.5": float((gdf["machine_confidence"] > 0.5).mean() * 100),
            "area_km2_media": float(gdf["area_km2_aprox"].mean()),
            "n_escenas_s1_unicas": int(gdf["s1_scene_id"].nunique()) if "s1_scene_id" in gdf.columns else None,
        }
        print("\nResumen:", json.dumps(resumen, indent=2, ensure_ascii=False))
        pd.DataFrame([resumen]).to_csv(OUT_DIR / f"resumen_{area_name}.csv", index=False)

        # Rejilla + tabla de positivos por celda-escena
        grid = build_grid(bbox)
        pts = gpd.GeoDataFrame(
            gdf[["centroid_lon", "centroid_lat"]],
            geometry=gpd.points_from_xy(gdf["centroid_lon"], gdf["centroid_lat"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="left", predicate="within")
        gdf["cell_id"] = joined["cell_id"].values

        n_sin_celda = gdf["cell_id"].isna().sum()
        print(f"  Detecciones sin celda asignada: {n_sin_celda}")

        positives_panel = (
            gdf.dropna(subset=["cell_id"])
            .groupby(["zona_estudio", "cell_id", "s1_scene_id"])
            .size()
            .reset_index(name="n_events")
        )
        positives_panel.to_csv(OUT_DIR / "positivos_celda_escena.csv", index=False)
        grid.to_file(OUT_DIR / f"rejilla_{area_name}.geojson", driver="GeoJSON")

        print(f"  Rejilla: {len(grid)} celdas de ~{CELL_KM} km")
        print(f"  positivos_celda_escena.csv: {len(positives_panel)} filas")

    print(f"\nTodo guardado en: {OUT_DIR}")
