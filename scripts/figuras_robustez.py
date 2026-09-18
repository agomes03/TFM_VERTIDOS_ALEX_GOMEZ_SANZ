"""Figuras de los tres analisis de robustez: etiquetas humanas,
autocorrelacion espacial y sensibilidad al tamano de celda."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
FIG_DIR = Path(__file__).parent / "figuras"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
WARN = "#c98500"
CRIT = "#d03b3b"
INK = "#12201f"
MUTED = "#6b7a7b"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": GRID, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ============ 1. Etiquetas humanas ============
comp = pd.read_csv(OUT_DIR / "hitl_composicion.csv")
conf = pd.read_csv(OUT_DIR / "hitl_confianza_por_clase.csv")
umb = pd.read_csv(OUT_DIR / "hitl_efecto_umbral.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# 1a. Confianza por clase verificada
conf = conf[conf["count"] >= 5].sort_values("mean")
colores = [WARN if c == "Natural" else (MUTED if c == "Ambiguous" else BLUE)
           for c in conf["hitl_cls_name"]]
ax1.barh(conf["hitl_cls_name"], conf["mean"], color=colores, height=0.6,
         xerr=conf["std"] / np.sqrt(conf["count"]), error_kw=dict(ecolor="#9aa8a9", lw=1.2))
ax1.set_xlabel("Confianza media del modelo (machine_confidence)")
ax1.set_title("El modelo está MÁS seguro cuando la mancha es natural",
              fontsize=11, fontweight="bold", loc="left")
ax1.set_xlim(0.78, 0.95)
clean(ax1); ax1.spines["left"].set_visible(False)

# 1b. Efecto del umbral
ax2.plot(umb["umbral"], umb["pct_natural"], color=WARN, marker="o", linewidth=2.2,
         label="Filtraciones naturales")
ax2.plot(umb["umbral"], umb["pct_antropogenico"], color=BLUE, marker="s", linewidth=2.2,
         label="Antropogénicas")
ax2.set_xlabel("Umbral mínimo de confianza aplicado")
ax2.set_ylabel("% de las detecciones verificadas")
ax2.set_title("Filtrar por confianza alta aumenta la contaminación",
              fontsize=11, fontweight="bold", loc="left")
ax2.legend(frameon=False, fontsize=9)
clean(ax2)

fig.suptitle("Validación parcial con las 777 detecciones revisadas por humanos",
             fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.03)
fig.text(0.02, -0.04,
         "Las detecciones que un revisor humano clasificó como filtración natural tienen mayor confianza del modelo que las antropogénicas (Mann-Whitney p < 10⁻⁸; AUC 0,33, es decir,\n"
         "discrimina en sentido inverso). En consecuencia, endurecer el umbral de confianza —lo que intuitivamente depuraría la muestra— eleva la proporción de fenómenos naturales del 15,6 % al 26,0 %.\n"
         "Aviso: las revisadas no son una muestra aleatoria del catálogo, por lo que estas proporciones no son extrapolables al conjunto.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(FIG_DIR / "fig_hitl.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_hitl.png OK")

# ============ 2. Autocorrelacion: inflacion de errores estandar ============
ac = pd.read_csv(OUT_DIR / "autocorr_errores_estandar.csv", index_col=0)
ac = ac.drop(index="const", errors="ignore")
NAMES = {"wind_speed": "Viento (ERA5)", "dist_costa_km": "Distancia a costa",
         "log_ais_hours": "Tráfico AIS (log)", "mes_sin": "Estacionalidad (sen)",
         "mes_cos": "Estacionalidad (cos)", "anios_desde_2023": "Tendencia anual"}
ac["label"] = [NAMES.get(i, i) for i in ac.index]
ac = ac.sort_values("inflacion_se")

fig, ax = plt.subplots(figsize=(7.6, 3.6))
b = ax.barh(ac["label"], ac["inflacion_se"], color=ORANGE, height=0.58)
ax.axvline(1, color=INK, linewidth=1.4)
for bi, v in zip(b, ac["inflacion_se"]):
    ax.text(v + 0.04, bi.get_y() + bi.get_height() / 2, f"×{v:.2f}", va="center",
            fontsize=9.5, color=INK, fontweight="bold")
ax.set_xlabel("Factor de inflación del error estándar al agrupar por celda")
ax.set_title("Corrección por autocorrelación espacial (I de Moran = 0,54; p = 0,002)",
             fontsize=11.5, fontweight="bold", loc="left")
ax.set_xlim(0, ac["inflacion_se"].max() * 1.18)
clean(ax); ax.spines["left"].set_visible(False)
fig.text(0.02, -0.09,
         "Los errores estándar clásicos del GLM asumen independencia entre celdas y resultan optimistas: al agrupar por celda se inflan entre 1,26 y 2,87 veces.\n"
         "Las seis variables mantienen no obstante su significación (p < 0,05) tras la corrección.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG_DIR / "fig_autocorrelacion.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_autocorrelacion.png OK")

# ============ 3. MAUP ============
maup_path = OUT_DIR / "maup_resultados.csv"
if maup_path.exists():
    m = pd.read_csv(maup_path)
    m["label"] = m["variable"].map(NAMES)
    piv = m.pivot(index="label", columns="cell_km", values="rate_ratio")
    orden = piv[10].sort_values().index
    piv = piv.loc[orden]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(piv))
    h = 0.26
    for k, (cell, color, mk) in enumerate(zip([5, 10, 20], [AQUA, BLUE, ORANGE], ["o", "s", "^"])):
        if cell in piv.columns:
            ax.barh(y + (1 - k) * h, piv[cell], height=h, color=color, label=f"{cell} km")
    ax.axvline(1, color=INK, linewidth=1.3)
    ax.set_yticks(y); ax.set_yticklabels(piv.index)
    ax.set_xlabel("Rate ratio (la línea vertical marca la ausencia de efecto)")
    ax.set_title("Sensibilidad al tamaño de celda (MAUP)", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, title="Resolución", title_fontsize=9)
    clean(ax); ax.spines["left"].set_visible(False)
    fig.text(0.02, -0.07,
             "El mismo GLM reajustado sobre rejillas de 5, 10 y 20 km. El signo y la ordenación de los efectos se mantienen en las tres resoluciones, lo que indica que las\n"
             "conclusiones no dependen de la elección —por lo demás arbitraria— del tamaño de celda.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIG_DIR / "fig_maup.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig_maup.png OK")
else:
    print("[aviso] maup_resultados.csv aun no existe")
