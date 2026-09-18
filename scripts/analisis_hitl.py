"""
Validacion parcial con las etiquetas revisadas por humanos (hitl_cls_name) del
catalogo Cerulean.

Aviso de sesgo de muestreo: las 777 detecciones revisadas NO son una muestra
aleatoria del catalogo -- Cerulean revisa manualmente los casos que le
interesan o los dudosos. Las proporciones observadas aqui NO se extrapolan al
catalogo completo; sirven para caracterizar la relacion entre la confianza del
modelo y la clase verificada, no para estimar una tasa global de falsos
positivos.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

df = pd.read_csv(OUT_DIR / "cerulean_golfo_persico_limpio.csv",
                 usecols=["id", "hitl_cls_name", "cls", "machine_confidence",
                          "area_km2_aprox", "slick_timestamp", "centroid_lon", "centroid_lat"])
h = df[df["hitl_cls_name"].notna()].copy()
print(f"Detecciones revisadas por humanos: {len(h)} de {len(df)} ({len(h)/len(df)*100:.1f}%)")

ANTROPO = ["Infrastructure", "Anthropogenic", "Vessel", "Vessel, old",
           "Vessel, recent", "Vessel, coincident"]


def grupo(x):
    if x in ANTROPO:
        return "antropogenico"
    if x == "Natural":
        return "natural"
    return "ambiguo"


h["grupo"] = h["hitl_cls_name"].apply(grupo)

# --- 1. Composicion ---
comp = h["hitl_cls_name"].value_counts().reset_index()
comp.columns = ["clase", "n"]
comp["pct"] = comp["n"] / comp["n"].sum() * 100
print("\n=== Composicion de las revisadas ===")
print(comp.round(1).to_string(index=False))

comp_g = h["grupo"].value_counts().reset_index()
comp_g.columns = ["grupo", "n"]
comp_g["pct"] = comp_g["n"] / comp_g["n"].sum() * 100
print("\n=== Agrupado ===")
print(comp_g.round(1).to_string(index=False))
comp.to_csv(OUT_DIR / "hitl_composicion.csv", index=False)
comp_g.to_csv(OUT_DIR / "hitl_composicion_grupo.csv", index=False)

# --- 2. Confianza por clase ---
conf = h.groupby("hitl_cls_name")["machine_confidence"].agg(["count", "mean", "median", "std"])
conf = conf.sort_values("mean", ascending=False)
print("\n=== machine_confidence por clase verificada ===")
print(conf.round(4).to_string())
conf.to_csv(OUT_DIR / "hitl_confianza_por_clase.csv")

a = h.loc[h["grupo"] == "antropogenico", "machine_confidence"]
n = h.loc[h["grupo"] == "natural", "machine_confidence"]
u, p = stats.mannwhitneyu(a, n, alternative="two-sided")
# Tamano del efecto: correlacion rango-biserial
r_rb = 1 - (2 * u) / (len(a) * len(n))
print(f"\nAntropogenico (n={len(a)}, mediana {a.median():.4f}) vs "
      f"Natural (n={len(n)}, mediana {n.median():.4f})")
print(f"Mann-Whitney U={u:.0f}, p={p:.3e}, r_rango-biserial={r_rb:.3f}")

# --- 3. Poder discriminante de la confianza ---
# Si machine_confidence sirviera para separar antropogenico de natural, un
# clasificador basado solo en ella tendria AUC > 0.5. Se calcula sobre las
# revisadas, excluyendo los ambiguos.
from sklearn.metrics import roc_auc_score
sub = h[h["grupo"].isin(["antropogenico", "natural"])].copy()
y = (sub["grupo"] == "antropogenico").astype(int)
auc = roc_auc_score(y, sub["machine_confidence"])
print(f"\nAUC de machine_confidence para separar antropogenico de natural: {auc:.3f}")
print("(0.5 = sin poder discriminante; <0.5 = discrimina en sentido inverso)")

# --- 4. Efecto de filtrar por confianza alta ---
print("\n=== Efecto de filtrar por umbral de confianza (sobre las revisadas) ===")
filas = []
for umbral in [0.0, 0.7, 0.8, 0.85, 0.9, 0.95]:
    s = h[h["machine_confidence"] >= umbral]
    if len(s) == 0:
        continue
    filas.append({
        "umbral": umbral,
        "n": len(s),
        "pct_antropogenico": (s["grupo"] == "antropogenico").mean() * 100,
        "pct_natural": (s["grupo"] == "natural").mean() * 100,
        "pct_ambiguo": (s["grupo"] == "ambiguo").mean() * 100,
    })
umb = pd.DataFrame(filas)
# round(1) colapsaba 0.85 y 0.95 en la columna de umbral al imprimir
print(umb.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
umb.to_csv(OUT_DIR / "hitl_efecto_umbral.csv", index=False)

# --- 5. Area por grupo ---
print("\n=== Area (km2) por grupo ===")
print(h.groupby("grupo")["area_km2_aprox"].agg(["count", "mean", "median"]).round(2).to_string())

with open(OUT_DIR / "hitl_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Validacion parcial con etiquetas revisadas por humanos ===\n\n")
    f.write(f"Revisadas: {len(h)} de {len(df)} ({len(h)/len(df)*100:.1f}% del catalogo)\n")
    f.write("AVISO: no es una muestra aleatoria; las proporciones no se extrapolan.\n\n")
    f.write("Composicion:\n")
    f.write(comp.round(1).to_string(index=False))
    f.write("\n\nAgrupado:\n")
    f.write(comp_g.round(1).to_string(index=False))
    f.write("\n\nConfianza del modelo por clase verificada:\n")
    f.write(conf.round(4).to_string())
    f.write(f"\n\nMann-Whitney antropogenico vs natural: U={u:.0f}, p={p:.3e}, r={r_rb:.3f}\n")
    f.write(f"AUC de machine_confidence (antropogenico vs natural): {auc:.3f}\n")
    f.write("\nEfecto del umbral de confianza:\n")
    f.write(umb.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print(f"\nGuardado en {OUT_DIR}")
