"""
Fase 6, version 2: sustituye dist_ruta_maritima_km (proxy OSM) por AIS real
(GFW), log1p(ais_hours) por la fuerte asimetria de la variable (65% ceros,
maximo ~175000 horas-buque en una celda-mes).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

df = pd.read_csv(OUT_DIR / "panel_modelo_fase6_con_ais.csv")
print(f"Filas totales: {len(df)}")

df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais_hours"] = np.log1p(df["ais_hours"])

df = df[df["n_pases"] > 0].copy()
feature_cols = ["wind_speed", "dist_costa_km", "log_ais_hours", "mes_sin", "mes_cos", "anios_desde_2023"]
df = df.dropna(subset=feature_cols + ["n_events", "n_pases"])
print(f"Filas tras dropna: {len(df)}")

X = df[feature_cols].copy()
for col in ["wind_speed", "dist_costa_km", "log_ais_hours", "anios_desde_2023"]:
    X[col] = (X[col] - X[col].mean()) / X[col].std()
X = sm.add_constant(X)

y = df["n_events"].values
offset = np.log(df["n_pases"].values)

print("\n=== Poisson GLM (con AIS real) ===")
poisson_res = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
print(poisson_res.summary())
dispersion = poisson_res.pearson_chi2 / poisson_res.df_resid
print(f"\nDispersion (Pearson chi2/df): {dispersion:.2f}")

print("\n=== Binomial Negativa (con AIS real) ===")
nb_res = NegativeBinomial(y, X, exposure=df["n_pases"].values).fit(method="bfgs", maxiter=200, disp=False)
print(nb_res.summary())

rr = pd.DataFrame({"coef": nb_res.params, "rate_ratio": np.exp(nb_res.params), "p_value": nb_res.pvalues})
print("\n=== Rate ratios (binomial negativa, con AIS real) ===")
print(rr.round(4).to_string())
rr.to_csv(OUT_DIR / "glm_fase6_con_ais_rate_ratios.csv")

with open(OUT_DIR / "glm_fase6_con_ais_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Poisson GLM (con AIS real) ===\n")
    f.write(str(poisson_res.summary()))
    f.write(f"\n\nDispersion (Pearson chi2/df): {dispersion:.2f}\n")
    f.write("\n\n=== Binomial Negativa (con AIS real) ===\n")
    f.write(str(nb_res.summary()))

print(f"\nGuardado en {OUT_DIR}")
