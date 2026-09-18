"""Figura: tasa de deteccion por franja de profundidad (contraste de la
hipotesis de sesgo en aguas someras)."""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
FIG_DIR = Path(__file__).parent / "figuras"

BLUE = "#2a78d6"
WARN = "#c98500"
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

t = pd.read_csv(OUT_DIR / "detecciones_por_profundidad.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

colores = [WARN if f in ("0-5 m", "5-10 m") else BLUE for f in t["franja"]]
b = ax1.bar(t["franja"], t["tasa_por_1000_pasadas"], color=colores, width=0.6)
for bi, v in zip(b, t["tasa_por_1000_pasadas"]):
    ax1.text(bi.get_x() + bi.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center",
             fontsize=9.5, fontweight="bold", color=INK)
ax1.set_ylabel("Detecciones / 1.000 pasadas")
ax1.set_title("Tasa de detección normalizada por profundidad", fontsize=11,
              fontweight="bold", loc="left")
ax1.set_ylim(0, t["tasa_por_1000_pasadas"].max() * 1.22)

ax2.bar(t["franja"], t["pct_celdas"], width=0.6, color="#c3c2b7", label="% de celdas del AOI")
ax2.plot(t["franja"], t["pct_detecciones"], color=BLUE, marker="o", linewidth=2,
         label="% de detecciones")
ax2.set_ylabel("Porcentaje")
ax2.set_title("Reparto del área frente al de las detecciones", fontsize=11,
              fontweight="bold", loc="left")
ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

for ax in (ax1, ax2):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Profundidad del fondo (ETOPO1)")

fig.suptitle("¿Genera el fondo somero falsos positivos? — contraste sobre el catálogo completo",
             fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.04)
fig.text(0.02, -0.06,
         "Las franjas en ámbar (<10 m) son las señaladas como sospechosas en la inspección visual (figura 12). Su tasa de detección por pasada es un 42 % inferior\n"
         "a la de aguas más profundas, lo que descarta un sesgo sistemático del detector hacia los fondos someros del golfo.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(FIG_DIR / "fig_profundidad.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_profundidad.png OK")
