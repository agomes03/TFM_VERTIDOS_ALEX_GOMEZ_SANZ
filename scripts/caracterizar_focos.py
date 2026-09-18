"""
Donde estan las celdas que concentran el incremento?

Si el 10 % de las celdas aporta el 87 % del aumento, la pregunta decisiva es
si esas celdas comparten algo (trafico, cercania a costa, contiguidad
espacial) o si aparecen repartidas al azar. Lo primero apunta a focos reales;
lo segundo, a ruido o a un artefacto del procesado.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "salida_golfo_persico"

piv = pd.read_csv(OUT / "tendencia_por_celda.csv")
piv["cell_id"] = piv["cell_id"].astype(str)

celda = pd.read_csv(BASE / "salida_golfo_persico" / "dim_celda.csv", encoding="utf-8-sig")
celda["CeldaID"] = celda["CeldaID"].astype(str)
piv = piv.merge(celda.rename(columns={"CeldaID": "cell_id"}), on="cell_id", how="left")

panel = pd.read_csv(BASE / "salida_golfo_persico" / "panel_modelo_fase6_era5.csv")
panel["cell_id"] = panel["cell_id"].astype(str)
traf = panel[panel["mes"] < "2025-07"].groupby("cell_id")["ais_hours"].mean().rename("ais_previo")
piv = piv.merge(traf, on="cell_id", how="left")

n = len(piv)
k = max(1, int(n * 0.10))
piv = piv.sort_values("exceso", ascending=False)
piv["grupo"] = "Resto"
piv.iloc[:k, piv.columns.get_loc("grupo")] = "Top 10 % del incremento"

top = piv[piv["grupo"] != "Resto"]
resto = piv[piv["grupo"] == "Resto"]

print("=" * 62)
print("PERFIL DE LAS CELDAS QUE CONCENTRAN EL INCREMENTO")
print("=" * 62)
print(f"Top 10 %: {len(top)} celdas | Resto: {len(resto)}")
print()

filas = []
for var, etiqueta, unidad in [
    ("ais_previo", "Trafico AIS previo", "horas-buque/mes"),
    ("DistanciaCostaKm", "Distancia a costa", "km"),
    ("ProfundidadM", "Profundidad", "m"),
    ("tasa_antes", "Tasa previa", "det/pasada"),
]:
    a = top[var].dropna()
    b = resto[var].dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    u, p = stats.mannwhitneyu(a, b)
    filas.append({"Variable": etiqueta, "Unidad": unidad,
                  "MedianaTop": a.median(), "MedianaResto": b.median(),
                  "Ratio": a.median() / b.median() if b.median() else np.nan,
                  "p": p})
    print(f"  {etiqueta:22} top {a.median():>10.3f} | resto {b.median():>10.3f}  (p={p:.2e})")

perfil = pd.DataFrame(filas)
perfil.to_csv(OUT / "tendencia_perfil_focos.csv", index=False)

# --- Contiguidad: las celdas del top se tocan entre si? ---
print()
print("=" * 62)
print("AGRUPAMIENTO ESPACIAL DEL TOP 10 %")
print("=" * 62)

ij = top["cell_id"].str.split("_", expand=True).astype(int)
coords = set(zip(ij[0], ij[1]))
vecinos = 0
for i, j in coords:
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (i + di, j + dj) in coords:
            vecinos += 1
vecinos //= 2  # cada par contado dos veces

# Esperado si las mismas celdas se repartieran al azar por el area de mar
mar = piv[piv["EsMar"] == True] if "EsMar" in piv.columns else piv
rng = np.random.default_rng(42)
todas_ij = piv["cell_id"].str.split("_", expand=True).astype(int)
todas = list(zip(todas_ij[0], todas_ij[1]))
sim = []
for _ in range(200):
    muestra = set(map(tuple, rng.choice(todas, size=len(coords), replace=False)))
    v = 0
    for i, j in muestra:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (i + di, j + dj) in muestra:
                v += 1
    sim.append(v // 2)
sim = np.array(sim)
p_agrup = (sim >= vecinos).mean()

print(f"  Pares de celdas contiguas observados: {vecinos}")
print(f"  Esperados al azar: {sim.mean():.0f} (DT {sim.std():.0f})")
print(f"  p (permutacion, 200 sim.): {p_agrup:.4f}")
print(f"  -> {'AGRUPADAS: forman focos' if p_agrup < 0.05 else 'dispersas'}")

# --- Cuanto del incremento cae en celdas que ya tenian trafico? ---
umbral_traf = piv["ais_previo"].replace(0, np.nan).median()
con_traf = top[top["ais_previo"] > umbral_traf]["exceso"].sum()
total_top = top["exceso"].sum()
print()
print(f"  Del incremento del top 10 %, el {con_traf / total_top:.1%} cae en celdas")
print(f"  con trafico AIS previo por encima de la mediana.")

resumen = {
    "n_top": len(top),
    "vecinos_observados": int(vecinos),
    "vecinos_esperados": float(sim.mean()),
    "p_agrupamiento": float(p_agrup),
    "pct_exceso_en_celdas_con_trafico": float(con_traf / total_top),
}
import json
with open(OUT / "tendencia_focos.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

with open(OUT / "tendencia_resumen.txt", "a", encoding="utf-8") as f:
    f.write("\n\nE. PERFIL DE LAS CELDAS DEL TOP 10 %\n")
    f.write(perfil.round(4).to_string(index=False))
    f.write(f"\n\n  Pares contiguos: {vecinos} observados frente a {sim.mean():.0f} esperados al azar "
            f"(p={p_agrup:.4f})\n")
    f.write(f"  Exceso en celdas con trafico previo alto: {con_traf / total_top:.1%}\n")

piv[["cell_id", "Latitud", "Longitud", "grupo", "exceso", "tasa_antes", "tasa_despues",
     "ais_previo", "DistanciaCostaKm", "ProfundidadM"]].to_csv(
    OUT / "tendencia_celdas_perfil.csv", index=False)
print(f"\nGuardado en {OUT}")
