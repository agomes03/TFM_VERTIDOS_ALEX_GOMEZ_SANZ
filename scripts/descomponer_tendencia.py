"""
Descomposicion de la tendencia temporal residual (rate ratio ~1,04/anio).

Pregunta: el aumento de la tasa de deteccion que persiste tras controlar por
esfuerzo, trafico y viento, corresponde a un foco real de vertidos o a un
cambio en el detector de Cerulean?

Cuatro contrastes con implicaciones opuestas:

  A. CONCENTRACION ESPACIAL. Un foco real se concentra en pocas celdas; un
     cambio del detector afecta a todo el area por igual.
  B. FIRMA DEL DETECTOR. Si cambio el modelo o su umbral, deberian moverse la
     distribucion de confianza y el tamano de las manchas detectadas.
  C. ESTRATIFICACION. Un foco real deberia concentrarse donde hay actividad
     (trafico alto); un cambio del detector afecta igual con y sin trafico.
  D. INTENSIDAD DE PROCESADO. El campo orchestrator_run identifica la
     ejecucion del pipeline que genero cada deteccion.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
DATOS = BASE / "salida_golfo_persico"
OUT = BASE / "salida_golfo_persico"

CORTE = "2025-07"  # mes en que arranca el tramo de mayor tasa

print("Cargando...")
det = pd.read_csv(DATOS / "cerulean_golfo_persico_limpio.csv",
                  usecols=["id", "slick_timestamp", "machine_confidence", "area_km2_aprox",
                           "orchestrator_run", "cls", "centroid_lon", "centroid_lat"])
det["ts"] = pd.to_datetime(det["slick_timestamp"], utc=True)
det["mes"] = det["ts"].dt.to_period("M").astype(str)
det["tramo"] = np.where(det["mes"] < CORTE, "Antes", "Despues")

panel = pd.read_csv(DATOS / "panel_modelo_fase6_era5.csv")
panel["cell_id"] = panel["cell_id"].astype(str)
panel["tramo"] = np.where(panel["mes"] < CORTE, "Antes", "Despues")

resultados = {}

# =====================================================================
# A. Concentracion espacial del incremento
# =====================================================================
print("\n" + "=" * 62)
print("A. CONCENTRACION ESPACIAL DEL INCREMENTO")
print("=" * 62)

tasa = panel.groupby(["cell_id", "tramo"], as_index=False).agg(
    ev=("n_events", "sum"), pa=("n_pases", "sum"))
piv = tasa.pivot(index="cell_id", columns="tramo", values=["ev", "pa"]).fillna(0)
piv.columns = [f"{a}_{b}" for a, b in piv.columns]
piv = piv[(piv["pa_Antes"] > 0) & (piv["pa_Despues"] > 0)].copy()
piv["tasa_antes"] = piv["ev_Antes"] / piv["pa_Antes"]
piv["tasa_despues"] = piv["ev_Despues"] / piv["pa_Despues"]
piv["delta_tasa"] = piv["tasa_despues"] - piv["tasa_antes"]

# Incremento atribuible a cada celda: cuantas detecciones "de mas" aporta
# respecto a las que habria tenido manteniendo su tasa anterior
piv["esperado"] = piv["tasa_antes"] * piv["pa_Despues"]
piv["exceso"] = piv["ev_Despues"] - piv["esperado"]

exceso_total = piv.loc[piv["exceso"] > 0, "exceso"].sum()
n_celdas = len(piv)
print(f"Celdas observadas en ambos tramos: {n_celdas:,}")
print(f"Celdas con incremento de tasa: {(piv['delta_tasa'] > 0).sum():,} "
      f"({(piv['delta_tasa'] > 0).mean():.1%})")
print(f"Celdas con descenso:           {(piv['delta_tasa'] < 0).sum():,} "
      f"({(piv['delta_tasa'] < 0).mean():.1%})")

orden = piv.sort_values("exceso", ascending=False)
acum = orden.loc[orden["exceso"] > 0, "exceso"].cumsum() / exceso_total
concentracion = {}
for pct in (0.01, 0.05, 0.10, 0.25, 0.50):
    k = max(1, int(n_celdas * pct))
    concentracion[pct] = orden["exceso"].head(k).sum() / exceso_total
    print(f"  El {pct:.0%} de celdas con mayor exceso aporta el "
          f"{concentracion[pct]:.1%} del incremento")

# Gini del exceso positivo, como medida de concentracion
x = np.sort(orden.loc[orden["exceso"] > 0, "exceso"].values)
n = len(x)
gini = (2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n
print(f"\nIndice de Gini del exceso: {gini:.3f}")
print("  (0 = repartido por igual entre celdas; 1 = concentrado en una sola)")
resultados["concentracion"] = concentracion
resultados["gini"] = gini
resultados["pct_celdas_suben"] = float((piv["delta_tasa"] > 0).mean())

piv.reset_index().to_csv(OUT / "tendencia_por_celda.csv", index=False)

# =====================================================================
# B. Firma del detector
# =====================================================================
print("\n" + "=" * 62)
print("B. FIRMA DEL DETECTOR")
print("=" * 62)

a = det.loc[det["tramo"] == "Antes"]
d = det.loc[det["tramo"] == "Despues"]

u1, p1 = stats.mannwhitneyu(a["machine_confidence"], d["machine_confidence"])
u2, p2 = stats.mannwhitneyu(a["area_km2_aprox"], d["area_km2_aprox"])
print(f"Confianza  — mediana antes {a['machine_confidence'].median():.4f} | "
      f"despues {d['machine_confidence'].median():.4f}  (p={p1:.2e})")
print(f"Area (km2) — mediana antes {a['area_km2_aprox'].median():.3f} | "
      f"despues {d['area_km2_aprox'].median():.3f}  (p={p2:.2e})")

# Detecciones pequenas: un detector mas sensible las multiplicaria
for umbral in (0.5, 1.0, 2.0):
    pa = (a["area_km2_aprox"] < umbral).mean()
    pd_ = (d["area_km2_aprox"] < umbral).mean()
    print(f"  Manchas < {umbral} km2: {pa:.1%} antes -> {pd_:.1%} despues")

conf_mes = det.groupby("mes").agg(
    conf_mediana=("machine_confidence", "median"),
    area_mediana=("area_km2_aprox", "median"),
    n=("id", "count")).reset_index()
conf_mes.to_csv(OUT / "tendencia_firma_detector.csv", index=False)

resultados["conf_antes"] = float(a["machine_confidence"].median())
resultados["conf_despues"] = float(d["machine_confidence"].median())
resultados["area_antes"] = float(a["area_km2_aprox"].median())
resultados["area_despues"] = float(d["area_km2_aprox"].median())
resultados["p_conf"] = float(p1)
resultados["p_area"] = float(p2)

# =====================================================================
# C. Estratificacion por trafico y profundidad
# =====================================================================
print("\n" + "=" * 62)
print("C. ESTRATIFICACION")
print("=" * 62)

prof = pd.read_csv(OUT / "profundidad_celda.csv")
prof["cell_id"] = prof["cell_id"].astype(str)
p2_ = panel.merge(prof[["cell_id", "prof_abs"]], on="cell_id", how="left")

# Trafico: mediana del periodo previo, para no clasificar con datos del futuro
traf_antes = panel[panel["tramo"] == "Antes"].groupby("cell_id")["ais_hours"].mean()
mediana_traf = traf_antes[traf_antes > 0].median()
p2_["traf_alto"] = p2_["cell_id"].map(traf_antes > mediana_traf)

filas = []
for nombre, mascara in [
    ("Trafico alto", p2_["traf_alto"] == True),
    ("Trafico bajo o nulo", p2_["traf_alto"] == False),
    ("Aguas < 10 m", p2_["prof_abs"] < 10),
    ("Aguas >= 10 m", p2_["prof_abs"] >= 10),
]:
    sub = p2_[mascara]
    ta = sub[sub["tramo"] == "Antes"]
    td = sub[sub["tramo"] == "Despues"]
    r_a = ta["n_events"].sum() / ta["n_pases"].sum() * 1000
    r_d = td["n_events"].sum() / td["n_pases"].sum() * 1000
    filas.append({"Estrato": nombre, "TasaAntes": r_a, "TasaDespues": r_d,
                  "Ratio": r_d / r_a if r_a else np.nan})
    print(f"  {nombre:22} {r_a:6.2f} -> {r_d:6.2f}  (x{r_d / r_a:.2f})")

estratos = pd.DataFrame(filas)
estratos.to_csv(OUT / "tendencia_estratos.csv", index=False)
resultados["estratos"] = estratos.to_dict(orient="records")

# =====================================================================
# D. Intensidad de procesado
# =====================================================================
print("\n" + "=" * 62)
print("D. INTENSIDAD DE PROCESADO (orchestrator_run)")
print("=" * 62)

runs = det.groupby("mes").agg(
    ejecuciones=("orchestrator_run", "nunique"),
    detecciones=("id", "count")).reset_index()
esf = panel.groupby("mes", as_index=False).agg(pasadas=("n_pases", "sum"),
                                               eventos=("n_events", "sum"))
runs = runs.merge(esf, on="mes", how="left")
runs["tasa"] = runs["eventos"] / runs["pasadas"] * 1000
runs["det_por_ejecucion"] = runs["detecciones"] / runs["ejecuciones"]
runs["tramo"] = np.where(runs["mes"] < CORTE, "Antes", "Despues")

ra = runs[runs["tramo"] == "Antes"]
rd = runs[runs["tramo"] == "Despues"]
print(f"Ejecuciones distintas por mes: {ra['ejecuciones'].mean():.0f} antes -> "
      f"{rd['ejecuciones'].mean():.0f} despues  (x{rd['ejecuciones'].mean() / ra['ejecuciones'].mean():.2f})")
print(f"Pasadas por mes:               {ra['pasadas'].mean():,.0f} antes -> "
      f"{rd['pasadas'].mean():,.0f} despues  (x{rd['pasadas'].mean() / ra['pasadas'].mean():.2f})")
print(f"Detecciones por ejecucion:     {ra['det_por_ejecucion'].mean():.1f} antes -> "
      f"{rd['det_por_ejecucion'].mean():.1f} despues")

r_ej_tasa = stats.spearmanr(runs["ejecuciones"], runs["tasa"])
r_pas_tasa = stats.spearmanr(runs["pasadas"], runs["tasa"])
print(f"\nCorrelacion de Spearman con la tasa de deteccion:")
print(f"  ejecuciones por mes: rho={r_ej_tasa.statistic:.3f} (p={r_ej_tasa.pvalue:.2e})")
print(f"  pasadas por mes:     rho={r_pas_tasa.statistic:.3f} (p={r_pas_tasa.pvalue:.2e})")

runs.to_csv(OUT / "tendencia_procesado.csv", index=False)
resultados["ejec_antes"] = float(ra["ejecuciones"].mean())
resultados["ejec_despues"] = float(rd["ejecuciones"].mean())
resultados["rho_ejec_tasa"] = float(r_ej_tasa.statistic)
resultados["p_rho_ejec"] = float(r_ej_tasa.pvalue)
resultados["rho_pasadas_tasa"] = float(r_pas_tasa.statistic)

# =====================================================================
# Sintesis
# =====================================================================
print("\n" + "=" * 62)
print("SINTESIS")
print("=" * 62)
señales_detector = 0
señales_foco = 0
if resultados["pct_celdas_suben"] > 0.55:
    señales_detector += 1
    print("  [detector] el incremento afecta a la mayoria de las celdas")
else:
    señales_foco += 1
    print("  [foco]     el incremento se limita a una minoria de celdas")

if gini < 0.6:
    señales_detector += 1
    print(f"  [detector] el exceso esta poco concentrado (Gini {gini:.2f})")
else:
    señales_foco += 1
    print(f"  [foco]     el exceso esta concentrado (Gini {gini:.2f})")

if p1 < 0.01 or p2 < 0.01:
    señales_detector += 1
    print("  [detector] la firma de las detecciones cambia entre tramos")

ratio_alto = estratos.loc[estratos["Estrato"] == "Trafico alto", "Ratio"].iloc[0]
ratio_bajo = estratos.loc[estratos["Estrato"] == "Trafico bajo o nulo", "Ratio"].iloc[0]
if abs(ratio_alto - ratio_bajo) < 0.25:
    señales_detector += 1
    print("  [detector] el aumento es similar con y sin trafico")
else:
    señales_foco += 1
    print("  [foco]     el aumento difiere segun el trafico")

print(f"\n  Senales compatibles con cambio del detector: {señales_detector}")
print(f"  Senales compatibles con foco real:           {señales_foco}")

with open(OUT / "tendencia_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Descomposicion de la tendencia temporal residual ===\n\n")
    f.write(f"Corte: {CORTE}\n\n")
    f.write("A. CONCENTRACION ESPACIAL\n")
    f.write(f"  Celdas comparables: {n_celdas}\n")
    f.write(f"  Con incremento de tasa: {resultados['pct_celdas_suben']:.1%}\n")
    for pct, v in concentracion.items():
        f.write(f"  Top {pct:.0%} de celdas aporta el {v:.1%} del incremento\n")
    f.write(f"  Gini del exceso: {gini:.3f}\n\n")
    f.write("B. FIRMA DEL DETECTOR\n")
    f.write(f"  Confianza mediana: {resultados['conf_antes']:.4f} -> {resultados['conf_despues']:.4f} (p={p1:.2e})\n")
    f.write(f"  Area mediana: {resultados['area_antes']:.3f} -> {resultados['area_despues']:.3f} km2 (p={p2:.2e})\n\n")
    f.write("C. ESTRATIFICACION\n")
    f.write(estratos.round(3).to_string(index=False))
    f.write("\n\nD. INTENSIDAD DE PROCESADO\n")
    f.write(f"  Ejecuciones/mes: {resultados['ejec_antes']:.0f} -> {resultados['ejec_despues']:.0f}\n")
    f.write(f"  Spearman ejecuciones-tasa: rho={resultados['rho_ejec_tasa']:.3f} (p={resultados['p_rho_ejec']:.2e})\n")
    f.write(f"  Spearman pasadas-tasa:     rho={resultados['rho_pasadas_tasa']:.3f}\n\n")
    f.write(f"Senales de cambio del detector: {señales_detector} | de foco real: {señales_foco}\n")

import json
with open(OUT / "tendencia_resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=float)

print(f"\nGuardado en {OUT}")
