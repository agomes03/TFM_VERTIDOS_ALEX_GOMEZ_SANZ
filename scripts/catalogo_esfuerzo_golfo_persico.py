"""
Catalogo de esfuerzo (todas las pasadas de Sentinel-1 sobre el AOI, con y sin
deteccion) para Golfo Persico, corriendo localmente contra Earth Engine ya
autenticado. Misma logica que la seccion 9 del notebook optico.
"""
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape, Polygon

EE_PROJECT = os.environ["EE_PROJECT"]
ee.Initialize(project=EE_PROJECT)

STUDY_AREAS = {
    "golfo_persico": (47.5, 23.5, 56.5, 30.5),
}
CELL_KM = 10
FECHA_INICIO = "2023-01-01"
FECHA_FIN = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
S1_COLLECTION_ID = "COPERNICUS/S1_GRD"

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
OUT_DIR.mkdir(exist_ok=True, parents=True)


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


def fetch_s1_scene_catalog(aoi_bbox, start_date, end_date, chunk_days=30, verbose=True):
    aoi = ee.Geometry.Rectangle(list(aoi_bbox))
    all_scenes = []
    cur = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    while cur < end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days), end)
        coll = (
            ee.ImageCollection(S1_COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        attempt = 0
        while True:
            try:
                info = coll.getInfo()
                break
            except Exception as e:
                attempt += 1
                if attempt > 3:
                    print(f"  [ERROR] Bloque {cur.date()}-{chunk_end.date()} fallo tras 3 intentos: {e}")
                    info = {"features": []}
                    break
                print(f"  [WARN] {e}; reintentando ({attempt}/3)...")
                time.sleep(5)

        feats = info.get("features", [])
        for feat in feats:
            props = feat.get("properties", {})
            # S1_GRD no trae 'geometry' a nivel de feature (sale None); la
            # huella real esta en properties['system:footprint'] como
            # LinearRing, que hay que envolver como Polygon.
            footprint = props.get("system:footprint")
            if not footprint or "coordinates" not in footprint:
                continue
            geom = Polygon(footprint["coordinates"])
            all_scenes.append({
                "s1_scene_id": feat["id"].split("/")[-1],
                "timestamp_ms": props.get("system:time_start"),
                "orbit_pass": props.get("orbitProperties_pass"),
                "geometry": geom,
            })
        if verbose:
            print(f"  {cur.date()} -> {chunk_end.date()}: {len(feats)} escenas (acumulado {len(all_scenes)})")
        cur = chunk_end

    return all_scenes


if __name__ == "__main__":
    exposure_panels = {}

    for area_name, bbox in STUDY_AREAS.items():
        print(f"\n=== Catalogo de esfuerzo S1: {area_name} ({FECHA_INICIO} -> {FECHA_FIN}) ===")
        scenes = fetch_s1_scene_catalog(bbox, FECHA_INICIO, FECHA_FIN)
        print(f"  Total escenas S1 (con y sin deteccion): {len(scenes)}")
        if not scenes:
            print(f"  [AVISO] Sin escenas para {area_name}; se omite.")
            continue

        scenes_gdf = gpd.GeoDataFrame(
            [{"s1_scene_id": s["s1_scene_id"], "timestamp_ms": s["timestamp_ms"], "orbit_pass": s["orbit_pass"],
              "geometry": s["geometry"]} for s in scenes],
            crs="EPSG:4326",
        )

        grid = build_grid(bbox)
        pases = gpd.sjoin(grid, scenes_gdf, how="inner", predicate="intersects")[
            ["cell_id", "s1_scene_id", "timestamp_ms", "orbit_pass"]
        ].drop_duplicates()

        exposure_panels[area_name] = pases
        out_path = OUT_DIR / f"esfuerzo_{area_name}.csv"
        pases.to_csv(out_path, index=False)
        print(f"  {len(scenes_gdf)} escenas -> {len(pases)} pares celda-escena. Guardado: {out_path}")

    # --- Unir con los positivos ya generados (positivos_celda_escena.csv) ---
    positives = pd.read_csv(OUT_DIR / "positivos_celda_escena.csv")
    positives["cell_id"] = positives["cell_id"].astype(str)

    full_panels = []
    for area_name, pases in exposure_panels.items():
        pos = positives.loc[positives["zona_estudio"] == area_name, ["cell_id", "s1_scene_id", "n_events"]]
        panel = pases.copy()
        panel["cell_id"] = panel["cell_id"].astype(str)
        panel = panel.merge(pos, on=["cell_id", "s1_scene_id"], how="left")
        panel["n_events"] = panel["n_events"].fillna(0).astype(int)
        panel["has_event"] = (panel["n_events"] > 0).astype(int)
        panel["zona_estudio"] = area_name
        full_panels.append(panel)

    panel_completo = pd.concat(full_panels, ignore_index=True)
    panel_completo.to_csv(OUT_DIR / "panel_celda_escena_completo.csv", index=False)
    print(f"\nPanel celda-escena completo: {len(panel_completo)} filas, "
          f"{panel_completo['has_event'].mean() * 100:.3f}% con deteccion")

    panel_completo["fecha"] = pd.to_datetime(panel_completo["timestamp_ms"], unit="ms", utc=True)
    panel_completo["mes"] = panel_completo["fecha"].dt.to_period("M").astype(str)

    exposure_monthly = (
        panel_completo.groupby(["zona_estudio", "cell_id", "mes"])
        .agg(n_pases=("s1_scene_id", "nunique"), n_events=("n_events", "sum"))
        .reset_index()
    )
    exposure_monthly["tasa_deteccion"] = exposure_monthly["n_events"] / exposure_monthly["n_pases"]
    exposure_monthly.to_csv(OUT_DIR / "exposicion_mensual_celda.csv", index=False)
    print(f"Guardado: exposicion_mensual_celda.csv ({len(exposure_monthly)} filas celda-mes)")

    print(f"\nTodo en: {OUT_DIR}")
