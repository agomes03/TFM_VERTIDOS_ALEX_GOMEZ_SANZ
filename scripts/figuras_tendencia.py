"""Figuras de la descomposicion de la tendencia temporal."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "salida_golfo_persico"
FIG = BASE / "figuras"

AZUL = "#2a78d6"; NARANJA = "#eb6834"; AQUA = "#1baf7a"; AMBAR = "#c98500"
INK = "#12201f"; MUTED = "#6b7a7b"; GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": GRID, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ---------- 1. Kitagawa + redistribucion del esfuerzo ----------
kit = json.load(open(OUT / "tendencia_kitagawa.json", encoding="utf-8"))
mov = pd.read_csv(OUT / "tendencia_redistribucion_esfuerzo.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.35]})

# Cascada
partes = [("Tasa\nantes", kit["tasa_antes_por_mil"], MUTED),
          ("Efecto\ncomposición", kit["efecto_composicion_por_mil"], NARANJA),
          ("Efecto\ntasa", kit["efecto_tasa_por_mil"], AQUA),
          ("Tasa\ndespués", kit["tasa_despues_por_mil"], MUTED)]
base = 0
for i, (etq, val, col) in enumerate(partes):
    if i == 0 or i == 3:
        ax1.bar(i, val, color=col, width=.6)
        ax1.text(i, val + .15, f"{val:.2f}", ha="center", fontsize=9.5, fontweight="bold")
        base = val if i == 0 else base
    else:
        ax1.bar(i, val, bottom=base, color=col, width=.6)
        ax1.text(i, base + val + .15, f"+{val:.2f}", ha="center", fontsize=9.5,
                 fontweight="bold", color=col)
        base += val
ax1.set_xticks(range(4)); ax1.set_xticklabels([p[0] for p in partes], fontsize=9)
ax1.set_ylabel("Detecciones por 1.000 pasadas")
ax1.set_title("De dónde sale el aumento", fontsize=11.5, fontweight="bold", loc="left")
ax1.set_ylim(0, kit["tasa_despues_por_mil"] * 1.18)
clean(ax1)
ax1.text(.5, .93, f"{kit['pct_composicion']:.0%} composición · {kit['pct_tasa']:.0%} tasa",
         transform=ax1.transAxes, ha="center", fontsize=9.5, color=MUTED)

# Redistribucion por decil
colores = [NARANJA if v < 0 else AZUL for v in mov["cambio_peso_pct"]]
ax2.bar(mov["decil_tasa_previa"], mov["cambio_peso_pct"], color=colores, width=.65)
ax2.axhline(0, color=INK, linewidth=1)
ax2.set_ylabel("Cambio en la cuota de esfuerzo (%)")
ax2.set_xlabel("Decil de tasa de detección previa  (D1 = menor, D10 = mayor)")
ax2.set_title("El satélite se desplazó hacia las celdas que ya detectaban más",
              fontsize=11.5, fontweight="bold", loc="left")
clean(ax2)

fig.suptitle("Descomposición del aumento de la tasa de detección (2023–jun 2025 frente a jul 2025–2026)",
             fontsize=12.5, fontweight="bold", x=.02, ha="left", y=1.04)
fig.text(.02, -.06,
         "La tasa agregada pasa de 7,71 a 9,73 detecciones por 1.000 pasadas (×1,26). La descomposición de Kitagawa atribuye el 62 % de esa subida a un cambio\n"
         "en el reparto del esfuerzo de observación —los deciles D8–D10 ganan entre un 15 % y un 16 % de cuota, mientras D1–D7 la pierden— y solo el 38 % a que\n"
         "las celdas detecten más de lo que detectaban.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, .02, 1, .97])
fig.savefig(FIG / "fig_tendencia_kitagawa.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_tendencia_kitagawa.png OK")

# ---------- 2. Concentracion, firma y ratios ----------
piv = pd.read_csv(OUT / "tendencia_por_celda.csv")
firma = pd.read_csv(OUT / "tendencia_firma_detector.csv")
unif = pd.read_csv(OUT / "tendencia_control_uniformidad.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

# (a) curva de concentracion
ex = np.sort(piv.loc[piv["exceso"] > 0, "exceso"].values)[::-1]
acum = np.cumsum(ex) / ex.sum()
x = np.arange(1, len(ex) + 1) / len(piv)
axes[0].plot(x * 100, acum * 100, color=AZUL, linewidth=2.2)
axes[0].plot([0, 100], [0, 100], color=MUTED, linestyle="--", linewidth=1)
for pct, col in ((5, AMBAR), (10, NARANJA)):
    idx = min(len(acum) - 1, int(len(piv) * pct / 100))
    axes[0].plot([pct, pct], [0, acum[idx] * 100], color=col, linewidth=1, linestyle=":")
    axes[0].plot(pct, acum[idx] * 100, "o", color=col, markersize=6)
    axes[0].annotate(f"{acum[idx]:.0%}", (pct, acum[idx] * 100),
                     textcoords="offset points", xytext=(8, -4), fontsize=9,
                     color=col, fontweight="bold")
axes[0].set_xlabel("% de celdas (ordenadas por exceso)")
axes[0].set_ylabel("% del incremento acumulado")
axes[0].set_title("El incremento se concentra", fontsize=11, fontweight="bold", loc="left")
axes[0].set_xlim(0, 100); axes[0].set_ylim(0, 102)
clean(axes[0])

# (b) firma del detector
f = firma.copy()
f["fecha"] = pd.to_datetime(f["mes"] + "-01")
ax = axes[1]
ax.plot(f["fecha"], f["conf_mediana"], color=AZUL, linewidth=2, label="Confianza")
ax.axvline(pd.Timestamp("2025-07-01"), color=MUTED, linestyle="--", linewidth=1)
ax.set_ylabel("Confianza mediana", color=AZUL)
ax.tick_params(axis="y", colors=AZUL)
ax2b = ax.twinx()
ax2b.plot(f["fecha"], f["area_mediana"], color=NARANJA, linewidth=2)
ax2b.set_ylabel("Área mediana (km²)", color=NARANJA)
ax2b.tick_params(axis="y", colors=NARANJA)
ax2b.grid(False)
ax.set_title("La firma de las detecciones no cambia", fontsize=11, fontweight="bold", loc="left")
clean(ax)
ax.text(.03, .06, "línea: julio 2025", transform=ax.transAxes, fontsize=8, color=MUTED)

# (c) distribucion de ratios
r = unif["ratio"].clip(upper=4)
axes[2].hist(r, bins=40, color=AQUA, alpha=.8, edgecolor="white", linewidth=.5)
axes[2].axvline(1, color=INK, linewidth=1.6, label="sin cambio")
axes[2].axvline(unif["ratio"].median(), color=NARANJA, linewidth=2,
                label=f"mediana {unif['ratio'].median():.2f}")
axes[2].axvline(1.261, color=AZUL, linewidth=2, linestyle="--", label="global ×1,26")
axes[2].set_xlabel("Ratio de tasa (después / antes) por celda")
axes[2].set_ylabel("Celdas")
axes[2].set_title("La celda típica no aumentó", fontsize=11, fontweight="bold", loc="left")
axes[2].legend(frameon=False, fontsize=8.5)
clean(axes[2])

fig.suptitle("Contrastes sobre el origen de la tendencia", fontsize=12.5,
             fontweight="bold", x=.02, ha="left", y=1.05)
fig.text(.02, -.07,
         "Izquierda: el 10 % de celdas con mayor exceso concentra el 87 % del incremento, y esas celdas están agrupadas (448 pares contiguos frente a 139 esperados al azar).\n"
         "Centro: ni la confianza ni el tamaño de las manchas cambian de forma apreciable, lo que descarta una modificación del detector. Derecha: entre las celdas con señal\n"
         "suficiente, la mediana del ratio es 0,84 —la celda típica detectó menos, no más—, de modo que el aumento agregado no procede de un incremento generalizado.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, .03, 1, .96])
fig.savefig(FIG / "fig_tendencia_contrastes.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_tendencia_contrastes.png OK")
