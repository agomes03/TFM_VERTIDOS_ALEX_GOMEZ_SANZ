"""
Fase 6: modelo explicativo con offset de esfuerzo.
n_events ~ viento + distancia a costa + distancia a ruta maritima + estacionalidad + tendencia anual
offset = log(n_pases)  -> el modelo estima la TASA de deteccion, no el conteo bruto,
controlando explicitamente por cuanto se observo cada celda-mes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

df = pd.read_csv(OUT_DIR / "panel_modelo_fase6.csv")
print(f"Filas totales: {len(df)}")

df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0

# Excluir filas sin pasadas (no deberia haber, pero por seguridad) y tipar
df = df[df["n_pases"] > 0].copy()

feature_cols = ["wind_speed", "dist_costa_km", "dist_ruta_maritima_km", "mes_sin", "mes_cos", "anios_desde_2023"]
df = df.dropna(subset=feature_cols + ["n_events", "n_pases"])
print(f"Filas tras dropna: {len(df)}")

# Estandarizar las covariables continuas de escala grande para que los
# coeficientes sean comparables entre si (no afecta al ajuste, solo a la lectura)
X = df[feature_cols].copy()
for col in ["wind_speed", "dist_costa_km", "dist_ruta_maritima_km", "anios_desde_2023"]:
    X[col] = (X[col] - X[col].mean()) / X[col].std()
X = sm.add_constant(X)

y = df["n_events"].values
offset = np.log(df["n_pases"].values)

print("\n=== Poisson GLM (con offset de esfuerzo) ===")
poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()
print(poisson_res.summary())

# Chequeo de sobredispersion: Pearson chi2 / grados de libertad
pearson_chi2 = poisson_res.pearson_chi2
dof = poisson_res.df_resid
dispersion = pearson_chi2 / dof
print(f"\nEstadistico de sobredispersion (Pearson chi2 / df): {dispersion:.2f}")
print("(>1.5-2 indica sobredispersion -> los errores estandar de Poisson son optimistas; "
      "se ajusta tambien un binomial negativo mas abajo, que es mas robusto en ese caso)")

print("\n=== Binomial Negativa (mismas variables, mas robusta a sobredispersion) ===")
nb_model = NegativeBinomial(y, X, exposure=df["n_pases"].values)
nb_res = nb_model.fit(method="bfgs", maxiter=200, disp=False)
print(nb_res.summary())

# Tabla resumen con rate ratios (exp(coef)) para lectura directa
print("\n=== Rate ratios (exp(coef)), modelo binomial negativa ===")
rr = pd.DataFrame({
    "coef": nb_res.params,
    "rate_ratio": np.exp(nb_res.params),
    "p_value": nb_res.pvalues,
})
print(rr.round(4).to_string())

rr.to_csv(OUT_DIR / "glm_fase6_rate_ratios.csv")
with open(OUT_DIR / "glm_fase6_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Poisson GLM ===\n")
    f.write(str(poisson_res.summary()))
    f.write(f"\n\nDispersion (Pearson chi2/df): {dispersion:.2f}\n")
    f.write("\n\n=== Binomial Negativa ===\n")
    f.write(str(nb_res.summary()))

print(f"\nGuardado: glm_fase6_rate_ratios.csv y glm_fase6_resumen.txt en {OUT_DIR}")
