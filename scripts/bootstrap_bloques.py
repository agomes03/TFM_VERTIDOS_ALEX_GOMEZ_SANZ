"""
Sensibilidad del intervalo bootstrap de la diferencia de PR-AUC (XGBoost - RF)
a la dependencia entre observaciones.

El bootstrap original (bootstrap_diferencia.py) remuestrea celdas-mes como si
fueran independientes. Las celdas-mes de una misma celda comparten historia y
las celdas vecinas comparten condiciones, asi que ese intervalo puede ser
demasiado estrecho. Se comparan tres esquemas de remuestreo pareado:

  1. Observaciones celda-mes independientes (esquema original).
  2. Celdas completas, con todos sus meses (dependencia temporal).
  3. Bloques espaciales de 5 x 5 celdas, unos 50 km, con todos sus meses
     (dependencia espacial y temporal a esa escala).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path(__file__).resolve().parent
DATOS = BASE / "salida_golfo_persico"
RES = BASE / "salida_golfo_persico"
N_BOOT = 1000
LADO_BLOQUE = 5

pred = pd.read_csv(DATOS / "fase7b_predicciones_2026_ambos.csv")
y = pred["has_event"].to_numpy()
p_rf = pred["proba_rf"].to_numpy()
p_xgb = pred["proba_xgb"].to_numpy()
ij = pred["cell_id"].str.split("_", expand=True).astype(int)
pred["bloque"] = (ij[0] // LADO_BLOQUE).astype(str) + "_" + (ij[1] // LADO_BLOQUE).astype(str)

obs = average_precision_score(y, p_xgb) - average_precision_score(y, p_rf)
print(f"Diferencia observada (XGB - RF): {obs:+.4f} | n = {len(y):,}")


def indices_por(grupo):
    return [np.asarray(ix) for ix in pred.groupby(grupo).indices.values()]


def remuestrear(esquema, rng):
    if esquema == "observaciones":
        return rng.integers(0, len(y), len(y))
    grupos = GRUPOS[esquema]
    elegidos = rng.integers(0, len(grupos), len(grupos))
    return np.concatenate([grupos[k] for k in elegidos])


GRUPOS = {"celdas": indices_por("cell_id"), "bloques_50km": indices_por("bloque")}
print(f"Celdas: {len(GRUPOS['celdas']):,} | bloques de {LADO_BLOQUE}x{LADO_BLOQUE} celdas: {len(GRUPOS['bloques_50km']):,}")

resumen = {}
for esquema in ("observaciones", "celdas", "bloques_50km"):
    rng = np.random.default_rng(42)
    difs = []
    for _ in range(N_BOOT):
        ix = remuestrear(esquema, rng)
        if y[ix].sum() == 0:
            continue
        difs.append(average_precision_score(y[ix], p_xgb[ix]) - average_precision_score(y[ix], p_rf[ix]))
    difs = np.array(difs)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    resumen[esquema] = {"ic95_bajo": float(lo), "ic95_alto": float(hi), "amplitud": float(hi - lo),
                        "prop_xgb_mejor": float((difs > 0).mean()), "replicas": int(len(difs))}
    print(f"  {esquema:14s} IC95 [{lo:+.4f}, {hi:+.4f}]  amplitud {hi - lo:.4f}  "
          f"XGB mejor en {np.mean(difs > 0):.1%}")

base_amp = resumen["observaciones"]["amplitud"]
for k in resumen:
    resumen[k]["amplitud_relativa"] = resumen[k]["amplitud"] / base_amp
    print(f"  amplitud relativa {k}: x{resumen[k]['amplitud_relativa']:.2f}")

RES.mkdir(exist_ok=True)
(RES / "bootstrap_bloques.json").write_text(
    json.dumps({"diferencia_observada": float(obs), "n_boot": N_BOOT, "esquemas": resumen},
               ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Guardado en {RES / 'bootstrap_bloques.json'}")
