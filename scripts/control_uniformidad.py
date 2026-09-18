"""
Control: el incremento es multiplicativo y uniforme, o hay focos?

El analisis anterior ordena las celdas por EXCESO ABSOLUTO, una medida que
favorece a las que ya tenian tasa alta aunque el aumento fuese proporcional en
todas partes. Para distinguirlo hay que mirar el RATIO de tasas por celda:

  - Si el aumento es multiplicativo uniforme, los ratios se reparten en torno
    a un mismo valor y su dispersion es la que cabe esperar del ruido de
    Poisson.
  - Si hay focos, aparece una cola de celdas con ratios muy por encima de lo
    que el ruido explica.

Se contrasta con un test de Poisson por celda: bajo la hipotesis de aumento
uniforme, los eventos del segundo tramo siguen una Poisson de media
tasa_previa x factor_global x pasadas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = BASE / "salida_golfo_persico"

piv = pd.read_csv(OUT / "tendencia_por_celda.csv")
piv["cell_id"] = piv["cell_id"].astype(str)

# Factor global de aumento entre tramos
tasa_g_antes = piv["ev_Antes"].sum() / piv["pa_Antes"].sum()
tasa_g_despues = piv["ev_Despues"].sum() / piv["pa_Despues"].sum()
factor = tasa_g_despues / tasa_g_antes
print(f"Tasa global: {tasa_g_antes * 1000:.2f} -> {tasa_g_despues * 1000:.2f} por 1.000 pasadas")
print(f"Factor global de aumento: x{factor:.3f}\n")

# Solo celdas con senal suficiente para que el ratio sea informativo
sub = piv[(piv["ev_Antes"] >= 5) & (piv["pa_Despues"] > 0)].copy()
sub["ratio"] = sub["tasa_despues"] / sub["tasa_antes"]
print(f"Celdas con >=5 detecciones previas: {len(sub):,}")
print(f"Ratio de tasas — mediana {sub['ratio'].median():.3f} | "
      f"P25 {sub['ratio'].quantile(.25):.3f} | P75 {sub['ratio'].quantile(.75):.3f}")

# Test de Poisson por celda bajo aumento uniforme
esperado = sub["tasa_antes"] * factor * sub["pa_Despues"]
sub["esperado_unif"] = esperado
sub["p_exceso"] = 1 - stats.poisson.cdf(sub["ev_Despues"] - 1, esperado)

# Correccion por comparaciones multiples (Benjamini-Hochberg)
p = np.sort(sub["p_exceso"].values)
m = len(p)
umbral_bh = None
for i in range(m - 1, -1, -1):
    if p[i] <= (i + 1) / m * 0.05:
        umbral_bh = p[i]
        break
sub["foco"] = sub["p_exceso"] <= (umbral_bh if umbral_bh is not None else -1)

n_focos = int(sub["foco"].sum())
print(f"\nCeldas con exceso significativo sobre el aumento uniforme (BH 5 %): "
      f"{n_focos} de {len(sub)} ({n_focos / len(sub):.1%})")

if n_focos:
    focos = sub[sub["foco"]].sort_values("ev_Despues", ascending=False)
    exceso_focos = (focos["ev_Despues"] - focos["esperado_unif"]).sum()
    exceso_total = (sub["ev_Despues"] - sub["esperado_unif"]).clip(lower=0).sum()
    print(f"Esas celdas aportan {exceso_focos:.0f} detecciones por encima de lo que "
          f"predice el aumento uniforme")
    print(f"({exceso_focos / exceso_total:.1%} del exceso total sobre el modelo uniforme)")
    print("\nDiez celdas con mayor exceso sobre el aumento uniforme:")
    cols = ["cell_id", "ev_Antes", "ev_Despues", "esperado_unif", "ratio", "p_exceso"]
    print(focos[cols].head(10).round(3).to_string(index=False))

# Cuanta de la varianza de los ratios explica el ruido de Poisson?
# Bajo aumento uniforme, Var(ratio) ~ tasa_despues / pasadas_despues
sub["var_esperada"] = sub["tasa_antes"] * factor / sub["pa_Despues"] / (sub["tasa_antes"] ** 2)
disp = sub["ratio"].var() / sub["var_esperada"].mean()
print(f"\nDispersion de los ratios frente a la esperada por ruido de Poisson: x{disp:.2f}")
print("  (~1 = compatible con aumento uniforme; >>1 = heterogeneidad real)")

sub.to_csv(OUT / "tendencia_control_uniformidad.csv", index=False)

with open(OUT / "tendencia_resumen.txt", "a", encoding="utf-8") as f:
    f.write("\n\nF. CONTROL DE UNIFORMIDAD (ratio de tasas, no exceso absoluto)\n")
    f.write(f"  Factor global de aumento: x{factor:.3f}\n")
    f.write(f"  Celdas con >=5 detecciones previas: {len(sub)}\n")
    f.write(f"  Ratio mediano por celda: {sub['ratio'].median():.3f} "
            f"(P25 {sub['ratio'].quantile(.25):.3f}, P75 {sub['ratio'].quantile(.75):.3f})\n")
    f.write(f"  Celdas con exceso significativo sobre el aumento uniforme (BH 5%): "
            f"{n_focos} ({n_focos / len(sub):.1%})\n")
    f.write(f"  Sobredispersion de los ratios: x{disp:.2f}\n")

import json
with open(OUT / "tendencia_uniformidad.json", "w", encoding="utf-8") as f:
    json.dump({"factor_global": float(factor), "n_celdas": len(sub),
               "ratio_mediano": float(sub["ratio"].median()),
               "n_focos": n_focos, "pct_focos": float(n_focos / len(sub)),
               "sobredispersion": float(disp)}, f, ensure_ascii=False, indent=2)
print(f"\nGuardado en {OUT}")
