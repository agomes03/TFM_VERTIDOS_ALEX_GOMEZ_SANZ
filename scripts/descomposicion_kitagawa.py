"""
Descomposicion de Kitagawa del aumento de la tasa agregada.

La tasa global es una media de las tasas de celda ponderada por el esfuerzo:

    T = sum_i ( w_i * t_i ),  con w_i = pasadas_i / pasadas_totales

Su variacion entre dos periodos se descompone en dos terminos:

    dT = sum_i ( (w_i' - w_i) * t_i )      <- efecto COMPOSICION
       + sum_i ( w_i' * (t_i' - t_i) )     <- efecto TASA

El primero recoge cuanto sube la tasa agregada solo porque el satelite pasa
proporcionalmente mas por celdas que ya detectaban mucho; el segundo, cuanto
sube porque las celdas detectan mas de lo que detectaban.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "salida_golfo_persico"

piv = pd.read_csv(OUT / "tendencia_por_celda.csv")
piv["cell_id"] = piv["cell_id"].astype(str)
piv = piv[(piv["pa_Antes"] > 0) & (piv["pa_Despues"] > 0)].copy()

pa_tot_a = piv["pa_Antes"].sum()
pa_tot_d = piv["pa_Despues"].sum()
piv["w_a"] = piv["pa_Antes"] / pa_tot_a
piv["w_d"] = piv["pa_Despues"] / pa_tot_d
piv["t_a"] = piv["ev_Antes"] / piv["pa_Antes"]
piv["t_d"] = piv["ev_Despues"] / piv["pa_Despues"]

T_a = (piv["w_a"] * piv["t_a"]).sum()
T_d = (piv["w_d"] * piv["t_d"]).sum()
dT = T_d - T_a

efecto_comp = ((piv["w_d"] - piv["w_a"]) * piv["t_a"]).sum()
efecto_tasa = (piv["w_d"] * (piv["t_d"] - piv["t_a"])).sum()

print("=" * 62)
print("DESCOMPOSICION DE KITAGAWA")
print("=" * 62)
print(f"Tasa agregada antes:   {T_a * 1000:7.3f} por 1.000 pasadas")
print(f"Tasa agregada despues: {T_d * 1000:7.3f} por 1.000 pasadas")
print(f"Variacion total:       {dT * 1000:+7.3f}  (x{T_d / T_a:.3f})")
print()
print(f"  Efecto COMPOSICION: {efecto_comp * 1000:+7.3f}  "
      f"({efecto_comp / dT:6.1%} de la variacion)")
print("     el satelite pasa proporcionalmente mas por celdas que ya detectaban mucho")
print(f"  Efecto TASA:        {efecto_tasa * 1000:+7.3f}  "
      f"({efecto_tasa / dT:6.1%} de la variacion)")
print("     las celdas detectan mas (o menos) de lo que detectaban")
print()
print(f"  Suma de comprobacion: {(efecto_comp + efecto_tasa) * 1000:+7.3f} "
      f"(debe coincidir con la variacion total)")

# A donde se desplazo el esfuerzo?
piv["dw"] = piv["w_d"] - piv["w_a"]
piv["decil_tasa_previa"] = pd.qcut(piv["t_a"].rank(method="first"), 10,
                                    labels=[f"D{i}" for i in range(1, 11)])
mov = piv.groupby("decil_tasa_previa", observed=True).agg(
    tasa_previa_media=("t_a", "mean"),
    peso_antes=("w_a", "sum"),
    peso_despues=("w_d", "sum")).reset_index()
mov["cambio_peso_pct"] = (mov["peso_despues"] / mov["peso_antes"] - 1) * 100

print()
print("=" * 62)
print("REDISTRIBUCION DEL ESFUERZO POR DECIL DE TASA PREVIA")
print("=" * 62)
print("  D1 = celdas que menos detectaban antes | D10 = las que mas")
print()
for r in mov.itertuples():
    print(f"  {r.decil_tasa_previa:4} tasa previa {r.tasa_previa_media * 1000:7.2f}‰  "
          f"peso {r.peso_antes:.4f} -> {r.peso_despues:.4f}  "
          f"({r.cambio_peso_pct:+6.1f} %)")

mov.to_csv(OUT / "tendencia_redistribucion_esfuerzo.csv", index=False)

resultados = {
    "tasa_antes_por_mil": float(T_a * 1000),
    "tasa_despues_por_mil": float(T_d * 1000),
    "factor": float(T_d / T_a),
    "efecto_composicion_por_mil": float(efecto_comp * 1000),
    "efecto_tasa_por_mil": float(efecto_tasa * 1000),
    "pct_composicion": float(efecto_comp / dT),
    "pct_tasa": float(efecto_tasa / dT),
}
import json
with open(OUT / "tendencia_kitagawa.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

with open(OUT / "tendencia_resumen.txt", "a", encoding="utf-8") as f:
    f.write("\n\nG. DESCOMPOSICION DE KITAGAWA\n")
    f.write(f"  Tasa agregada: {T_a * 1000:.3f} -> {T_d * 1000:.3f} por 1.000 pasadas (x{T_d / T_a:.3f})\n")
    f.write(f"  Efecto composicion: {efecto_comp * 1000:+.3f} ({efecto_comp / dT:.1%})\n")
    f.write(f"  Efecto tasa:        {efecto_tasa * 1000:+.3f} ({efecto_tasa / dT:.1%})\n\n")
    f.write("  Redistribucion del esfuerzo por decil de tasa previa:\n")
    f.write(mov.round(5).to_string(index=False))

print(f"\nGuardado en {OUT}")
