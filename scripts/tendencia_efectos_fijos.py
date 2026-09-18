"""
Contraste final: efectos fijos de celda.

Si el aumento de la tasa agregada procede sobre todo de que el esfuerzo se
redistribuyo hacia celdas que ya detectaban mucho (efecto composicion), al
introducir un efecto fijo por celda —que absorbe todo lo que distingue a una
celda de otra y no varia en el tiempo— el coeficiente de tendencia deberia
caer sustancialmente.

Si, por el contrario, la tendencia refleja un aumento real dentro de las
celdas, el coeficiente se mantendria.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent
OUT = BASE / "salida_golfo_persico"

df = pd.read_csv(BASE / "salida_golfo_persico" / "panel_modelo_fase6_era5.csv")
df["cell_id"] = df["cell_id"].astype(str)
df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["t"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais"] = np.log1p(df["ais_hours"])
df = df[df["n_pases"] > 0].dropna(
    subset=["wind_speed", "dist_costa_km", "log_ais", "n_events", "n_pases"]).copy()

# Efectos fijos solo tienen sentido donde hay variacion que explicar: se
# restringe a celdas con senal, para que el modelo sea manejable y estable.
senal = df.groupby("cell_id")["n_events"].sum()
celdas = senal[senal >= 5].index
sub = df[df["cell_id"].isin(celdas)].copy()
print(f"Celdas con >=5 detecciones: {len(celdas):,}")
print(f"Filas celda-mes: {len(sub):,}")
print(f"Detecciones incluidas: {int(sub['n_events'].sum()):,} "
      f"({sub['n_events'].sum() / df['n_events'].sum():.1%} del total)\n")

offset = np.log(sub["n_pases"].values)
y = sub["n_events"].values

VARS = ["wind_speed", "dist_costa_km", "log_ais", "mes_sin", "mes_cos", "t"]


def estandarizar(X, cols):
    X = X.copy()
    for c in cols:
        s = X[c].std()
        if s > 0:
            X[c] = (X[c] - X[c].mean()) / s
    return X


# --- Modelo 1: especificacion actual, sin efectos fijos ---
X1 = estandarizar(sub[VARS], ["wind_speed", "dist_costa_km", "log_ais", "t"])
X1 = sm.add_constant(X1)
m1 = sm.GLM(y, X1, family=sm.families.Poisson(), offset=offset).fit()
rr1 = np.exp(m1.params["t"])
print("=" * 62)
print("MODELO 1 — sin efectos fijos (especificacion de la memoria)")
print("=" * 62)
print(f"  Coeficiente de tendencia: {m1.params['t']:+.4f}  ->  rate ratio {rr1:.4f}")
print(f"  p = {m1.pvalues['t']:.3e}")

# --- Modelo 2: con efectos fijos de celda ---
# dist_costa_km es constante dentro de cada celda: colineal con el efecto fijo
VARS_FE = ["wind_speed", "log_ais", "mes_sin", "mes_cos", "t"]
X2 = estandarizar(sub[VARS_FE], ["wind_speed", "log_ais", "t"])
dummies = pd.get_dummies(sub["cell_id"], prefix="c", drop_first=True, dtype=float)
X2 = pd.concat([X2.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
X2 = sm.add_constant(X2)
print(f"\nAjustando con {dummies.shape[1]:,} efectos fijos de celda...")
m2 = sm.GLM(y, X2, family=sm.families.Poisson(), offset=offset).fit()
rr2 = np.exp(m2.params["t"])
print("=" * 62)
print("MODELO 2 — con efectos fijos de celda")
print("=" * 62)
print(f"  Coeficiente de tendencia: {m2.params['t']:+.4f}  ->  rate ratio {rr2:.4f}")
print(f"  p = {m2.pvalues['t']:.3e}")

print("\n" + "=" * 62)
print("COMPARACION")
print("=" * 62)
print(f"  Rate ratio de tendencia sin efectos fijos: {rr1:.4f}")
print(f"  Rate ratio de tendencia con efectos fijos: {rr2:.4f}")
reduccion = (rr1 - rr2) / (rr1 - 1) if rr1 > 1 else np.nan
print(f"  Reduccion del efecto: {reduccion:.1%}")
if rr2 < rr1:
    print("  -> parte de la tendencia procedia de diferencias ENTRE celdas,")
    print("     no de un aumento DENTRO de ellas")

# Como cambian las demas variables
print("\n  Otras variables (rate ratio):")
for v in ["wind_speed", "log_ais"]:
    print(f"    {v:16} {np.exp(m1.params[v]):.3f} -> {np.exp(m2.params[v]):.3f}")

comp = pd.DataFrame([
    {"Modelo": "Sin efectos fijos", "RateRatioTendencia": rr1,
     "p": m1.pvalues["t"], "Viento": np.exp(m1.params["wind_speed"]),
     "TraficoAIS": np.exp(m1.params["log_ais"])},
    {"Modelo": "Con efectos fijos de celda", "RateRatioTendencia": rr2,
     "p": m2.pvalues["t"], "Viento": np.exp(m2.params["wind_speed"]),
     "TraficoAIS": np.exp(m2.params["log_ais"])},
])
comp.to_csv(OUT / "tendencia_efectos_fijos.csv", index=False)

with open(OUT / "tendencia_resumen.txt", "a", encoding="utf-8") as f:
    f.write("\n\nH. EFECTOS FIJOS DE CELDA\n")
    f.write(f"  Submuestra: {len(celdas)} celdas con >=5 detecciones, {len(sub)} filas\n")
    f.write(comp.round(4).to_string(index=False))
    f.write(f"\n  Reduccion del efecto de tendencia: {reduccion:.1%}\n")

import json
with open(OUT / "tendencia_efectos_fijos.json", "w", encoding="utf-8") as f:
    json.dump({"rr_sin_fe": float(rr1), "rr_con_fe": float(rr2),
               "reduccion": float(reduccion), "n_celdas": len(celdas),
               "n_filas": len(sub)}, f, ensure_ascii=False, indent=2)
print(f"\nGuardado en {OUT}")
