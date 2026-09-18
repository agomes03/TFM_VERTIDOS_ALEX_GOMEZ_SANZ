"""
Autocorrelacion espacial de los residuos del GLM y correccion de los errores
estandar.

El GLM trata las 308.838 filas celda-mes como independientes, pero celdas
vecinas comparten trafico, viento y batimetria. Si queda autocorrelacion en
los residuos, los errores estandar estan subestimados y los p-valores son
optimistas. Aqui se mide (I de Moran sobre los residuos agregados por celda) y
se recalculan los errores estandar agrupando por celda.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
rng = np.random.default_rng(42)

df = pd.read_csv(OUT_DIR / "panel_modelo_fase6_era5.csv")
df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais_hours"] = np.log1p(df["ais_hours"])

FEATURES = ["wind_speed", "dist_costa_km", "log_ais_hours", "mes_sin", "mes_cos", "anios_desde_2023"]
df = df[df["n_pases"] > 0].dropna(subset=FEATURES + ["n_events", "n_pases"]).copy()

X = df[FEATURES].copy()
for c in ["wind_speed", "dist_costa_km", "log_ais_hours", "anios_desde_2023"]:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
X = sm.add_constant(X)
y = df["n_events"].values
offset = np.log(df["n_pases"].values)

print("Ajustando Poisson GLM (errores estandar clasicos)...")
base = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

print("Ajustando con errores estandar agrupados por celda...")
grupos = pd.Categorical(df["cell_id"]).codes
robusto = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(
    cov_type="cluster", cov_kwds={"groups": grupos})

comp = pd.DataFrame({
    "coef": base.params,
    "se_clasico": base.bse,
    "se_cluster": robusto.bse,
    "p_clasico": base.pvalues,
    "p_cluster": robusto.pvalues,
})
comp["inflacion_se"] = comp["se_cluster"] / comp["se_clasico"]
print("\n=== Errores estandar: clasicos vs agrupados por celda ===")
print(comp.round(4).to_string())
comp.to_csv(OUT_DIR / "autocorr_errores_estandar.csv")

sigue_sig = (comp["p_cluster"] < 0.05).drop("const", errors="ignore")
print(f"\nVariables que siguen siendo significativas (p<0.05) con SE agrupados: "
      f"{sigue_sig.sum()} de {len(sigue_sig)}")

# --- I de Moran sobre los residuos medios por celda ---
df["resid"] = base.resid_pearson
res_celda = df.groupby("cell_id")["resid"].mean().reset_index()

import geopandas as gpd
grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
grid["cell_id"] = grid["cell_id"].astype(str)
res_celda["cell_id"] = res_celda["cell_id"].astype(str)
g = grid.merge(res_celda, on="cell_id", how="inner")
print(f"\nCeldas con residuo: {len(g)}")

# Vecindad tipo torre sobre los indices i_j de la rejilla
ij = g["cell_id"].str.split("_", expand=True).astype(int)
g["i"], g["j"] = ij[0], ij[1]
pos = {(r.i, r.j): k for k, r in enumerate(g.itertuples())}
vals = g["resid"].values
n = len(vals)
media = vals.mean()
dev = vals - media

num = 0.0
w_total = 0
for k, r in enumerate(g.itertuples()):
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        v = pos.get((r.i + di, r.j + dj))
        if v is not None:
            num += dev[k] * dev[v]
            w_total += 1

moran = (n / w_total) * (num / (dev ** 2).sum())
esperado = -1.0 / (n - 1)
print(f"\nI de Moran de los residuos: {moran:.4f}  (esperado bajo aleatoriedad: {esperado:.4f})")

# Significacion por permutacion
perms = 499
mayores = 0
for _ in range(perms):
    p_vals = rng.permutation(vals)
    p_dev = p_vals - p_vals.mean()
    p_num = 0.0
    for k, r in enumerate(g.itertuples()):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = pos.get((r.i + di, r.j + dj))
            if v is not None:
                p_num += p_dev[k] * p_dev[v]
    p_moran = (n / w_total) * (p_num / (p_dev ** 2).sum())
    if p_moran >= moran:
        mayores += 1
p_moran_val = (mayores + 1) / (perms + 1)
print(f"p-valor por permutacion ({perms} permutaciones): {p_moran_val:.4f}")

with open(OUT_DIR / "autocorr_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Autocorrelacion espacial y correccion de errores estandar ===\n\n")
    f.write(f"I de Moran de los residuos por celda: {moran:.4f} (esperado {esperado:.4f})\n")
    f.write(f"p-valor por permutacion ({perms} perms): {p_moran_val:.4f}\n\n")
    f.write("Errores estandar clasicos vs agrupados por celda:\n")
    f.write(comp.round(4).to_string())
    f.write(f"\n\nFactor medio de inflacion de los SE: {comp['inflacion_se'].drop('const', errors='ignore').mean():.2f}x\n")
    f.write(f"Variables significativas (p<0.05) con SE agrupados: {sigue_sig.sum()} de {len(sigue_sig)}\n")

print(f"\nGuardado en {OUT_DIR}")
