"""
Diagnostico de sobredispersion para justificar la binomial negativa.

La razon de Pearson del Poisson (0,87) no senala sobredispersion, pero es un
diagnostico poco fiable con recuentos tan escasos: la mayoria de las celdas-mes
tienen una media esperada muy inferior a 1 y su contribucion al estadistico es
inestable. Se contrasta con tres diagnosticos mas adecuados:

  1. Contraste de Cameron y Trivedi (1990) para sobredispersion NB2.
  2. Ceros y cola observados frente a los esperados bajo Poisson.
  3. Razon de verosimilitudes y AIC entre Poisson y binomial negativa.

Misma especificacion que glm_fase6_era5.py (panel ERA5 + AIS, offset de esfuerzo).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial

BASE = Path(__file__).resolve().parent
DATOS = BASE / "salida_golfo_persico"
RES = BASE / "salida_golfo_persico"

df = pd.read_csv(DATOS / "panel_modelo_fase6_era5.csv")
df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais_hours"] = np.log1p(df["ais_hours"])
df = df[df["n_pases"] > 0].copy()
VARS = ["wind_speed", "dist_costa_km", "log_ais_hours", "mes_sin", "mes_cos", "anios_desde_2023"]
df = df.dropna(subset=VARS + ["n_events", "n_pases"])

X = df[VARS].copy()
for c in ["wind_speed", "dist_costa_km", "log_ais_hours", "anios_desde_2023"]:
    X[c] = (X[c] - X[c].mean()) / X[c].std()
X = sm.add_constant(X)
y = df["n_events"].to_numpy()
expo = df["n_pases"].to_numpy()

# --- Poisson ---
pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(expo)).fit()
mu = pois.fittedvalues.to_numpy()
pearson = float(pois.pearson_chi2 / pois.df_resid)
print(f"Filas: {len(y):,} | media de y: {y.mean():.4f} | media esperada: {mu.mean():.4f}")
print(f"Celdas-mes con media esperada < 1: {np.mean(mu < 1):.1%}")
print(f"Razon de Pearson (Poisson): {pearson:.3f}")

# --- 1. Cameron y Trivedi (1990): ((y - mu)^2 - y) / mu = alpha * mu + e ---
aux_y = ((y - mu) ** 2 - y) / mu
aux = sm.OLS(aux_y, mu).fit()
ct_alpha, ct_t = float(aux.params[0]), float(aux.tvalues[0])
ct_p = float(stats.norm.sf(ct_t))          # unilateral: H1 alpha > 0
aux_rob = sm.OLS(aux_y, mu).fit(cov_type="HC1")
ct_t_rob = float(aux_rob.tvalues[0])
print(f"Cameron-Trivedi: alpha = {ct_alpha:.3f}, t = {ct_t:.2f} (robusto {ct_t_rob:.2f}), p = {ct_p:.2e}")

# --- 2. Ceros y cola: observado frente a esperado bajo Poisson ---
obs_cero = int((y == 0).sum())
esp_cero = float(np.exp(-mu).sum())
cola = {}
for k in (3, 5, 10):
    obs = int((y >= k).sum())
    esp = float(stats.poisson.sf(k - 1, mu).sum())
    cola[k] = (obs, esp)
    print(f"  y >= {k:2d}: observadas {obs:6,} | esperadas bajo Poisson {esp:9.1f} | razon {obs / esp:6.1f}")
print(f"  ceros:   observados {obs_cero:,} | esperados {esp_cero:,.0f} | exceso {obs_cero - esp_cero:,.0f}")
print(f"  maximo observado: {int(y.max())} | varianza/media de y bruta: {y.var() / y.mean():.2f}")

# --- 3. Binomial negativa NB2, razon de verosimilitudes y AIC ---
nb = NegativeBinomial(y, X, exposure=expo).fit(method="bfgs", maxiter=300, disp=False)
alpha = float(nb.params["alpha"])
lr = 2 * (nb.llf - pois.llf)
lr_p = 0.5 * stats.chi2.sf(lr, 1)          # alpha en la frontera del espacio parametrico
aic_p, aic_nb = float(pois.aic), float(nb.aic)
print(f"NB2: alpha = {alpha:.3f} (EE {nb.bse['alpha']:.3f})")
print(f"LL Poisson {pois.llf:,.1f} | LL NB {nb.llf:,.1f} | LR = {lr:,.1f} (p = {lr_p:.1e})")
print(f"AIC Poisson {aic_p:,.1f} | AIC NB {aic_nb:,.1f} | diferencia {aic_p - aic_nb:,.1f}")

# --- coeficientes comparados ---
comp = pd.DataFrame({"rr_poisson": np.exp(pois.params), "rr_nb": np.exp(nb.params.drop("alpha"))})
print(comp.round(3).to_string())

res = {
    "n": int(len(y)), "media_y": float(y.mean()), "prop_mu_menor_1": float(np.mean(mu < 1)),
    "pearson_poisson": pearson,
    "ct_alpha": ct_alpha, "ct_t": ct_t, "ct_t_robusto": ct_t_rob, "ct_p": ct_p,
    "ceros_obs": obs_cero, "ceros_esp": esp_cero,
    "cola": {str(k): {"obs": v[0], "esp": v[1]} for k, v in cola.items()},
    "max_y": int(y.max()),
    "nb_alpha": alpha, "nb_alpha_ee": float(nb.bse["alpha"]),
    "ll_poisson": float(pois.llf), "ll_nb": float(nb.llf), "lr": float(lr), "lr_p": float(lr_p),
    "aic_poisson": aic_p, "aic_nb": aic_nb,
    "rr": comp.round(4).to_dict(),
}
RES.mkdir(exist_ok=True)
(RES / "sobredispersion.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nGuardado en {RES / 'sobredispersion.json'}")
