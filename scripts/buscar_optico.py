"""
Emparejamiento optico Sentinel-2 para verificacion visual de una muestra de
detecciones Cerulean de alta confianza.

Aviso fisico: Sentinel-1 pasa sobre el golfo Persico a ~02:30 y ~14:30 UTC,
Sentinel-2 a ~07:00-08:00 UTC. Aun con la mejor coincidencia posible hay
horas de desfase durante las cuales la mancha deriva por corriente y viento.
Por eso se registra el desfase real en horas de cada pareja, y la
verificacion es cualitativa, no una validacion pixel a pixel.
"""
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ee.Initialize(project=os.environ["EE_PROJECT"])

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
S2 = "COPERNICUS/S2_SR_HARMONIZED"

MAX_CLOUD = 20        # % nubes maximo en la escena
DAYS_WINDOW = 1       # ventana estrecha: la mancha se mueve
BUFFER_KM = 8         # margen alrededor de la mancha para el recorte
N_CANDIDATAS = 220    # detecciones a examinar

det = pd.read_csv(OUT_DIR / "cerulean_golfo_persico_limpio.csv",
                  usecols=["id", "slick_timestamp", "machine_confidence", "area_km2_aprox",
                           "centroid_lon", "centroid_lat", "s1_scene_id", "slick_url"])
det["slick_timestamp"] = pd.to_datetime(det["slick_timestamp"], utc=True)

# Candidatas: alta confianza y grandes (mas probable verlas en optico)
cand = det[(det["machine_confidence"] > 0.92) & (det["area_km2_aprox"] > 8)].copy()
cand = cand.sort_values("machine_confidence", ascending=False).head(N_CANDIDATAS)
print(f"Candidatas a examinar: {len(cand)}")

resultados = []
for k, r in enumerate(cand.itertuples(), 1):
    ts = r.slick_timestamp
    dlon = BUFFER_KM / (111.0 * np.cos(np.radians(r.centroid_lat)))
    dlat = BUFFER_KM / 111.0
    aoi = ee.Geometry.Rectangle([r.centroid_lon - dlon, r.centroid_lat - dlat,
                                 r.centroid_lon + dlon, r.centroid_lat + dlat])
    start = (ts - pd.Timedelta(days=DAYS_WINDOW)).strftime("%Y-%m-%d")
    end = (ts + pd.Timedelta(days=DAYS_WINDOW + 1)).strftime("%Y-%m-%d")

    try:
        coll = (ee.ImageCollection(S2).filterBounds(aoi).filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD)))
        n = coll.size().getInfo()
        if n == 0:
            continue
        info = coll.sort("CLOUDY_PIXEL_PERCENTAGE").first().getInfo()
        props = info["properties"]
        s2_ts = pd.to_datetime(props["system:time_start"], unit="ms", utc=True)
        desfase_h = abs((s2_ts - ts).total_seconds()) / 3600.0

        resultados.append({
            "slick_id": r.id,
            "s1_timestamp": ts,
            "s2_image_id": info["id"],
            "s2_timestamp": s2_ts,
            "desfase_horas": round(desfase_h, 2),
            "nubes_pct": round(props.get("CLOUDY_PIXEL_PERCENTAGE", np.nan), 2),
            "machine_confidence": round(r.machine_confidence, 4),
            "area_km2": round(r.area_km2_aprox, 2),
            "lon": r.centroid_lon, "lat": r.centroid_lat,
            "slick_url": r.slick_url,
            "n_candidatas_s2": n,
        })
    except Exception as e:
        print(f"  [warn] {r.id}: {type(e).__name__}")
        continue

    if k % 25 == 0:
        print(f"  {k}/{len(cand)} examinadas, {len(resultados)} con optico")

res = pd.DataFrame(resultados).sort_values("desfase_horas")
res.to_csv(OUT_DIR / "optico_emparejamiento.csv", index=False)
print(f"\nEmparejadas: {len(res)} de {len(cand)} candidatas")
if len(res):
    print(f"Desfase temporal (horas): min {res['desfase_horas'].min():.1f} | "
          f"mediana {res['desfase_horas'].median():.1f} | max {res['desfase_horas'].max():.1f}")
    print(f"Con desfase < 12 h: {(res['desfase_horas'] < 12).sum()}")
    print(f"Con desfase < 6 h:  {(res['desfase_horas'] < 6).sum()}")
    print("\nMejores 10 parejas:")
    print(res.head(10)[["slick_id", "desfase_horas", "nubes_pct", "area_km2", "machine_confidence"]].to_string(index=False))
