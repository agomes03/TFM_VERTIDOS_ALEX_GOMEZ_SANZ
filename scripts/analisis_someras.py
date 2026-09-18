"""
Cuantifica que fraccion de las detecciones del catalogo cae en aguas muy
someras, donde el fondo visible y las estructuras sedimentarias favorecen los
falsos positivos (look-alikes). Se compara contra la disponibilidad de esas
profundidades en el propio area de estudio, para no confundir "hay muchas
detecciones en aguas someras" con "hay mucha agua somera".
"""
import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geopandas as gpd

ee.Initialize(project=os.environ["EE_PROJECT"])

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
BATIMETRIA = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock")

# --- 1. Profundidad de cada celda de la rejilla (una sola consulta por lote) ---
grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
grid["cell_id"] = grid["cell_id"].astype(str)
cent = grid.geometry.centroid
puntos = [ee.Feature(ee.Geometry.Point([x, y]), {"cell_id": c})
          for x, y, c in zip(cent.x, cent.y, grid["cell_id"])]

print(f"Consultando profundidad de {len(puntos)} celdas por lotes...")
filas = []
LOTE = 900
for i in range(0, len(puntos), LOTE):
    fc = ee.FeatureCollection(puntos[i:i + LOTE])
    muestreado = BATIMETRIA.reduceRegions(fc, ee.Reducer.mean(), 1000)
    info = muestreado.getInfo()
    for f in info["features"]:
        p = f["properties"]
        filas.append({"cell_id": p["cell_id"], "profundidad_m": p.get("mean")})
    print(f"  {min(i + LOTE, len(puntos))}/{len(puntos)}")

prof = pd.DataFrame(filas)
prof = prof[prof["profundidad_m"].notna()]
prof["profundidad_m"] = prof["profundidad_m"].astype(float)
# Solo celdas de mar (profundidad negativa)
prof_mar = prof[prof["profundidad_m"] < 0].copy()
prof_mar["prof_abs"] = prof_mar["profundidad_m"].abs()
prof_mar.to_csv(OUT_DIR / "profundidad_celda.csv", index=False)
print(f"\nCeldas de mar: {len(prof_mar)} de {len(prof)}")

# --- 2. Detecciones por celda ---
positivos = pd.read_csv(OUT_DIR / "positivos_celda_escena.csv")
positivos["cell_id"] = positivos["cell_id"].astype(str)
det_por_celda = positivos.groupby("cell_id")["n_events"].sum().reset_index()

# --- 3. Esfuerzo por celda (para normalizar) ---
exposicion = pd.read_csv(OUT_DIR / "exposicion_mensual_celda.csv")
exposicion["cell_id"] = exposicion["cell_id"].astype(str)
esf_por_celda = exposicion.groupby("cell_id")["n_pases"].sum().reset_index()

df = prof_mar.merge(det_por_celda, on="cell_id", how="left").merge(esf_por_celda, on="cell_id", how="left")
df["n_events"] = df["n_events"].fillna(0)
df = df[df["n_pases"].notna() & (df["n_pases"] > 0)]

bins = [0, 5, 10, 20, 40, 1e9]
labels = ["0-5 m", "5-10 m", "10-20 m", "20-40 m", ">40 m"]
df["franja"] = pd.cut(df["prof_abs"], bins=bins, labels=labels, right=False)

tabla = df.groupby("franja", observed=True).agg(
    celdas=("cell_id", "count"),
    detecciones=("n_events", "sum"),
    pasadas=("n_pases", "sum"),
).reset_index()
tabla["pct_celdas"] = tabla["celdas"] / tabla["celdas"].sum() * 100
tabla["pct_detecciones"] = tabla["detecciones"] / tabla["detecciones"].sum() * 100
tabla["tasa_por_1000_pasadas"] = tabla["detecciones"] / tabla["pasadas"] * 1000

print("\n=== Detecciones por franja de profundidad ===")
print(tabla.round(2).to_string(index=False))

tabla.to_csv(OUT_DIR / "detecciones_por_profundidad.csv", index=False)

someras = df[df["prof_abs"] < 10]
resto = df[df["prof_abs"] >= 10]
tasa_som = someras["n_events"].sum() / someras["n_pases"].sum() * 1000
tasa_res = resto["n_events"].sum() / resto["n_pases"].sum() * 1000
print(f"\nTasa en aguas <10 m:  {tasa_som:.2f} detecciones / 1000 pasadas")
print(f"Tasa en aguas >=10 m: {tasa_res:.2f} detecciones / 1000 pasadas")
print(f"Razon: {tasa_som / tasa_res:.2f}x")

with open(OUT_DIR / "analisis_someras_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Detecciones por franja de profundidad (ETOPO1) ===\n\n")
    f.write(tabla.round(2).to_string(index=False))
    f.write(f"\n\nTasa en aguas <10 m:  {tasa_som:.2f} detecciones / 1000 pasadas\n")
    f.write(f"Tasa en aguas >=10 m: {tasa_res:.2f} detecciones / 1000 pasadas\n")
    f.write(f"Razon: {tasa_som / tasa_res:.2f}x\n")
